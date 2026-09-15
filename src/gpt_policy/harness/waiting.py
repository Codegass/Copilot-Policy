"""Allow host health faults to interrupt provider waits on the calling thread."""

from contextlib import contextmanager
from contextvars import ContextVar
import queue
import time

from .errors import AgentTimeoutError


_health_check = ContextVar("agent_wait_health_check", default=None)


@contextmanager
def monitor_health(check, *, deadline=None):
    # Context-local: no callback becomes model input or survives into another run.
    if deadline is not None:
        health_check = check

        def check():
            if health_check is not None:
                health_check()
            if time.monotonic() >= deadline:
                raise AgentTimeoutError("Model overload recovery deadline exceeded")

    token = _health_check.set(check)
    try:
        if check is not None:
            check()
        yield check if check is not None else lambda: None
    finally:
        _health_check.reset(token)


def wait_with_health(delay_s):
    """Interruptible backoff using the same health polling as provider I/O."""
    try:
        receive_event(queue.Queue(), deadline=time.monotonic() + delay_s)
    except queue.Empty:
        pass


def receive_event(events, *, deadline=None):
    check = _health_check.get()
    while True:
        if check is not None:
            check()
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise queue.Empty
        timeout = remaining
        if check is not None:
            timeout = .1 if remaining is None else min(.1, remaining)
        try:
            event = events.get(timeout=timeout)
        except queue.Empty:
            if check is None:
                raise
            continue
        if check is not None:
            check()
        return event
