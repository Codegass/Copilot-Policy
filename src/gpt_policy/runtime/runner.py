"""Preserve the established observation/execute sequence for every provider."""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from typing import Any

from ..harness.contract import AgentSession
from ..harness.errors import AgentOverloadedError, AgentTimeoutError
from ..harness.models import AgentTurn
from ..harness.protocol import observation
from ..harness.waiting import monitor_health, wait_with_health
from ..input.request import RunInput
from ..hardware.motion_control import MotionFault
from ..motion.coordination import TrajectoryIKError
from ..settings import RuntimeConfig
from ..interrupts import HomeInterrupted
from .console import RunConsole


# Keep the original exponential backoff for the first three retries, then
# cap the delay so a transient provider overload can be retried up to twenty
# times without creating an unbounded wait between attempts.
MODEL_RETRY_DELAYS_S = tuple(min(2.0 * (2 ** attempt), 8.0) for attempt in range(20))
MODEL_RECOVERY_TIMEOUT_S = 300.0


def run_loop(
    runtime: RuntimeConfig, run_input: RunInput, robot: Any, cameras: Any,
    video: Any, agent: AgentSession, tool_executor: Any, recorder: Any,
    observation_fn=observation, health_check=None, display=None,
) -> str:
    display = display if display is not None else RunConsole()
    previous: str | None = None
    last_state: dict[str, Any] | None = None
    observation_history: dict[int, dict[str, Any]] = {}
    for step in range(runtime.max_decisions):
        retries = 0
        recovery_deadline = None
        while True:
            try:
                check_recovery = (lambda: _check_recovery_health(robot, health_check)) if retries else health_check
                with monitor_health(check_recovery, deadline=recovery_deadline) as check:
                    if retries:
                        with display.waiting(f"Waiting {delay_s:.1f}s before model retry {retries}"):
                            wait_with_health(delay_s)
                        check()
                    prepare_started = time.perf_counter()
                    capture_after = time.time()
                    images = video.snapshot(after=capture_after)
                    state, state_error = _state_with_recovery(robot, last_state, step, recorder, display)
                    if state_error is not None:
                        if retries:
                            raise RuntimeError("Fresh robot state unavailable during model recovery")
                        previous = json.dumps(state_error, ensure_ascii=False)
                    camera_context = cameras.describe(images)
                    if not runtime.right_interface:
                        state["interface"] = runtime.interface
                    observation_text = observation_fn(
                        run_input.instruction, state, camera_context, previous, step
                    )
                    observation_history[step] = state
                    last_state = deepcopy(state)
                    observation_prepare_s = time.perf_counter() - prepare_started
                    record_started = time.perf_counter()
                    recorder.observation(step, observation_text, state, camera_context, images,
                                         **({"attempt": retries} if retries else {}))
                    observation_record_s = time.perf_counter() - record_started
                    check()
                    with display.waiting(f"Waiting for model decision {step + 1} / {runtime.max_decisions} (step {step})"):
                        decide_started = time.perf_counter()
                        try:
                            decision = agent.decide(AgentTurn(
                                observation_text, images, run_input.content if step == 0 else None,
                            ))
                        finally:
                            timing = {
                                "step": step, "observation_prepare_s": observation_prepare_s,
                                "observation_record_s": observation_record_s,
                                "agent_decide_s": time.perf_counter() - decide_started,
                                **({"attempt": retries} if retries else {}),
                            }
                            # These nested client intervals are not pure server inference timings.
                            provider_timing = getattr(agent, "last_decision_timing", None)
                            if isinstance(provider_timing, dict):
                                timing["provider"] = dict(provider_timing)
                            recorder.write("decision_timing", timing)
                    check()  # A late answer or hardware fault must never reach the tool executor.
                break
            except AgentOverloadedError as exc:
                if retries >= len(MODEL_RETRY_DELAYS_S):
                    recorder.write("model_retry_exhausted", {"step": step, "retries": retries,
                        "reason": "retry_limit", "error": str(exc)})
                    display.message("Model retry limit reached. Stopping and saving the run.", "red")
                    raise
                if recovery_deadline is None:
                    recovery_deadline = time.monotonic() + MODEL_RECOVERY_TIMEOUT_S
                delay_s = MODEL_RETRY_DELAYS_S[retries] + random.uniform(0, .5)
                retries += 1
                recorder.write("model_retry", {"step": step, "retry": retries,
                    "max_retries": len(MODEL_RETRY_DELAYS_S), "delay_s": delay_s,
                    "recovery_timeout_s": MODEL_RECOVERY_TIMEOUT_S,
                    "provider": exc.provider, "code": exc.code, "error": str(exc)})
                display.message(f"Model overloaded ({exc.code}). Retrying decision at step {step} "
                    f"in {delay_s:.1f}s ({retries}/{len(MODEL_RETRY_DELAYS_S)}); "
                    "no robot action will be replayed.", "yellow")
            except AgentTimeoutError as exc:
                if recovery_deadline is not None:
                    recorder.write("model_retry_exhausted", {"step": step, "retries": retries,
                        "reason": "recovery_deadline", "error": str(exc)})
                    display.message("Model recovery timed out. Stopping and saving the run.", "red")
                raise
        refresh = getattr(agent, "last_context_refresh", None)
        if isinstance(refresh, dict):
            recorder.write("model_context_refreshed", {"step": step, **refresh})
        if health_check is not None:
            health_check()
        recorder.write("model_decision", {"step": step, "decision": decision})
        action = str(decision.get("name"))
        arguments = decision.get("arguments", {})
        if not isinstance(arguments, dict):
            raise RuntimeError("工具 arguments 必须是对象")
        display.decision(step, action, arguments)
        if tool_executor.is_terminal(action):
            # Keep the model's conclusion even if homing fails. Final physical
            # success/failure is assigned by a human after device shutdown.
            recorder.write("terminal", {"step": step, "name": action, "arguments": arguments})
            _return_home(robot, recorder, step, action, display)
            return "give_up" if action == "give_up" else "completed"
        execute_started = time.perf_counter()
        execute_status = "failed"
        try:
            try:
                result = tool_executor.execute(action, arguments, state, observation_history)
                execute_status = "completed"
            except KeyboardInterrupt:
                execute_status = "interrupted"
                raise
            finally:
                # Includes local planning, command submission and settling.
                recorder.write("tool_timing", {"step": step, "name": action,
                    "execute_s": time.perf_counter() - execute_started, "status": execute_status})
        except TrajectoryIKError as exc:
            # Planning fails before CAN submission; continue the same session
            # from unchanged measured state without altering the requested path.
            error = {
                "tool": action, "error": "motion_not_executed",
                "requested_motion": arguments, **exc.details,
            }
            recorder.write("tool_error", {"step": step, **error})
            previous = json.dumps(error, ensure_ascii=False)
            display.error(step, action, error)
            continue
        except ValueError as exc:
            error = {"tool": action, "error": f"tool_rejected: {exc}"}
            recorder.write("tool_error", {"step": step, **error})
            previous = json.dumps(error, ensure_ascii=False)
            display.error(step, action, error)
            continue
        recorder.write("execution_result", {"step": step, "name": action, "result": result})
        display.returned(step, action)
        model_result = compact_execution_result(result)
        previous = json.dumps({"tool": action, "result": model_result}, ensure_ascii=False)
    recorder.write("budget_exhausted", {"decisions_used": runtime.max_decisions,
                                        "max_decisions": runtime.max_decisions})
    display.message(f"Decision budget exhausted ({runtime.max_decisions}). Returning home and saving recordings.", "yellow")
    _return_home(robot, recorder, runtime.max_decisions - 1, "budget_exhausted", display)
    return "budget_exhausted"


def _check_recovery_health(robot, health_check):
    if health_check is not None:
        health_check()
    # The recorder's callback may be a no-op. Do not extend a hold through a
    # stopped driver, stale feedback, or a reported temperature fault.
    state = robot.state()
    for side, arm in state.get("arms", {"arm": state}).items():
        if arm.get("temperature_over_limit"):
            raise MotionFault({"reason": "motor_overtemperature", "side": side,
                "source": "model_recovery", "temperature": arm["temperature_over_limit"]})


def _state_with_recovery(robot, last_state, step, recorder, display):
    while True:
        try:
            return robot.state(), None
        except RuntimeError as exc:
            error = {
                "tool": "state",
                "error": f"state_unavailable: {exc}",
                "recovering": True,
            }
            recorder.write("state_observation_error", {"step": step, **error})
            display.error(step, "state", error)
            if last_state is not None:
                state = deepcopy(last_state)
                state["state_observation_error"] = error
                return state, error
            display.message(f"State unavailable before first observation; retrying: {exc}", "yellow")
            time.sleep(0.5)


def _return_home(robot, recorder, step, trigger, display):
    home_started = time.perf_counter()
    home_status = "failed"
    try:
        display.message("Returning home...")
        result = robot.return_home()
        recorder.write("return_home", {"step": step, "trigger": trigger, "result": result})
        display.home(without_trace(result))
        require_home_settled(result)
        display.message("Home complete.")
        home_status = "completed"
    except KeyboardInterrupt as exc:
        home_status = "interrupted"
        raise HomeInterrupted from exc
    finally:
        # Includes home execution, completion checks and its result recording.
        recorder.write("home_timing", {"step": step, "trigger": trigger,
            "home_s": time.perf_counter() - home_started, "status": home_status})


def require_home_settled(result):
    states = result.get("arms", {"arm": result})
    for side, state in states.items():
        home = state.get("home", {})
        if isinstance(home, dict) and home.get("settle", {}).get("settled") is False:
            raise RuntimeError(f"回正等待超时，{side} 机械臂未确认到位：{home['settle']}")


def compact_execution_result(result: Any) -> Any:
    """Project robot feedback for the next turn without changing the recorded result."""
    if not isinstance(result, dict) or not result.get("execution_feedback"):
        # Localization, custom results and results without feedback retain their
        # information; only known robot feedback replaces a full state snapshot.
        return without_trace(result)

    compact = {"execution_feedback": without_trace(result["execution_feedback"])}
    for key in ("timestamp_s", "status", "error", "reason"):
        if key in result:
            compact[key] = result[key]
    if "arms" in result:
        timestamps = {
            side: state["timestamp_s"]
            for side, state in result["arms"].items()
            if "timestamp_s" in state
        }
        if timestamps:
            # Each arm's SDK uses its own clock. These are result-state sample
            # times, not a shared wall-clock time or the next observation time.
            compact["state_timestamps_s"] = timestamps
    if "trajectory" in result:
        compact["trajectory"] = _trajectory_summary(result["trajectory"])
    return compact


def _trajectory_summary(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Keep planned geometry and timing, separately from measured execution feedback."""
    summary = {
        key: trajectory[key]
        for key in (
            "planned_duration_s", "concurrent", "segment_synchronized",
            "common_segment_durations_s", "coordinated_start_delay_s",
            "ik_execution_tolerance_m_rad", "gripper_during_motion",
        )
        if key in trajectory
    }
    trace = trajectory.get("_trace", {})
    if "model_tcp_points_xyzrpy" in trace:
        # Includes the planning start and every requested waypoint, including
        # repeated points for held segments. These are not measured samples.
        summary["planned_tcp_points_xyzrpy"] = trace["model_tcp_points_xyzrpy"]
    if "segments" in trace:
        summary["segments"] = [
            {key: segment[key] for key in (
                "segment", "duration_s",
                "endpoint_fk_translation_error_m", "endpoint_fk_rotation_error_rad",
            ) if key in segment}
            for segment in trace["segments"]
        ]
    if "arms" in trajectory:
        summary["arms"] = {
            side: _trajectory_summary(plan) for side, plan in trajectory["arms"].items()
        }
    return summary


def without_trace(value: Any) -> Any:
    """Keep model results compact while the recorder retains full plans."""
    if isinstance(value, dict):
        return {key: without_trace(item) for key, item in value.items() if key != "_trace"}
    if isinstance(value, list):
        return [without_trace(item) for item in value]
    return value
