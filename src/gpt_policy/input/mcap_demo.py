"""Read exported YAM/AnyRobo episodes without importing their runtime or SDK."""

from bisect import bisect_left, bisect_right
import math
from pathlib import Path

from .action_sampling import event_times
from .recorded_demo import nearest_index, read_json
from .video_cache import _sha256_file


TOPICS = tuple(f"/{side}-{part}-{kind}" for side in ("left", "right")
               for part in ("arm", "ee") for kind in ("state", "action"))
MAX_DELTA_S = .1


def is_mcap_episode(root):
    return all((Path(root) / name).is_file() for name in
               ("camera_top.mp4", "session_summary.json", "episode.mcap"))


class McapDemo:
    def __init__(self, root, extractor, with_actions):
        self.root = Path(root)
        self.video = self.root / "camera_top.mp4"
        names = ["session_summary.json"]
        if (self.root / "finalization.json").is_file():
            names.append("finalization.json")
        if with_actions:
            names.append("episode.mcap")
        self.source_hashes = {name: _sha256_file(self.root / name) for name in names}
        media = read_json(self.root / "session_summary.json")
        if media.get("schema") != "any_robo_episode_media_v1" or not media.get("ok"):
            raise ValueError("YAM episode media export is not complete")
        self.pts = extractor.frame_times(self.video)
        if len(self.pts) != media["cameras"]["top"]["frames"]:
            raise ValueError("YAM video and media index have different frame counts")
        self.end_pts = self.pts[-1]
        self.views = {"top": (self.video, self.pts, self.pts)}
        for camera in ("left_wrist", "right_wrist"):
            path = self.root / f"camera_{camera}.mp4"
            if not path.is_file():
                continue
            info = media["cameras"][camera]
            if (media.get("timing_mode") != "recorded_timestamps"
                    or info.get("timing_mode") != "recorded_timestamps"
                    or info.get("timeline_origin_ns") != media.get("timeline_origin_ns")):
                raise ValueError("Multi-view YAM videos require a shared recorded timestamp origin")
            pts = extractor.frame_times(path)
            if len(pts) != info["frames"]:
                raise ValueError(f"YAM {camera} video and media index have different frame counts")
            self.views[camera] = (path, pts, pts)
        self.metadata = {
            "title": self.root.name, "source": str(self.root.resolve()),
            "demonstrator": "recorded_episode", "coverage": "sampled", "outcome": "unverified",
            "finalization_state": read_json(self.root / "finalization.json").get("state")
                                  if "finalization.json" in names else None,
            "source_robot": {"model": "YAM", "format": "episode_mcap_v1"},
            "coordinate_frame": "Source per-arm I2RT model/world base frames and recorded EEF site. "
                "No verified transform to the current robot. Metres, radians, quaternion xyzw "
                "(converted from source wxyz). Targets are historical commands/commanded-joint FK; "
                "measured fields are follower feedback. Gripper: 0 closed, 1 open. "
                "t_s and source_times_s are seconds since the media timeline origin; image t_s is MP4 PTS. "
                "Channels are independently joined by nearest host event time within 0.1 s; "
                "missing channels remain absent. Capture times do not prove simultaneous exposure. "
                "Completed recording finalization does not establish task success.",
        }
        self.samples = []
        if with_actions:
            self._read_actions(media)

    def _read_actions(self, media):
        # Only historical action import needs the standard MCAP/Protobuf reader.
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory

        if media.get("timing_mode") != "recorded_timestamps":
            raise ValueError("YAM action alignment requires recorded video timestamps")
        origin = int(media["timeline_origin_ns"])
        self.streams = {topic: [] for topic in TOPICS}
        self.times = {topic: [] for topic in (*TOPICS, "/top-camera", "/sync")}
        decoder = DecoderFactory()
        with (self.root / "episode.mcap").open("rb") as stream:
            reader = make_reader(stream)
            metadata = {k: v for entry in reader.iter_metadata() if entry.name == "session-metadata"
                        for k, v in entry.metadata.items()}
            if metadata.get("recording_format") != "episode_mcap_v1":
                raise ValueError("Expected episode_mcap_v1 session metadata")
            for schema, channel, message in reader.iter_messages(topics=list(self.times), log_time_order=False):
                topic = channel.topic
                times = self.times[topic]
                t = (message.log_time - origin) / 1e9
                if times and t <= times[-1]:
                    raise ValueError(f"YAM timestamps must strictly increase: {topic}")
                times.append(t)
                if topic not in self.streams:
                    continue
                decode = decoder.decoder_for(channel.message_encoding, schema)
                if decode is None:
                    raise ValueError(f"Missing protobuf schema: {topic}")
                self.streams[topic].append(_values(topic, decode(message.data), metadata))
        if any(not times for times in self.times.values()):
            raise ValueError(f"Missing YAM recording streams: {[k for k, v in self.times.items() if not v]}")
        self.capture_times = self.times["/top-camera"]
        if len(self.capture_times) != len(self.pts):
            raise ValueError("YAM video and MCAP camera have different frame counts")
        # Direct-remux MP4 timestamps are rounded to milliseconds. Verify every
        # frame, so a re-encoded constant-FPS video cannot silently shift actions.
        if any(abs(a - b) > .005 for a, b in zip(self.capture_times, self.pts)):
            raise ValueError("YAM video timestamps do not match MCAP capture times")
        self.state_times = self.times["/sync"]
        self.samples = [self._at(t)[0] for t in self.state_times]

    def _at(self, t, measured_only=False):
        sample = {"t_s": t, "source_times_s": {}}
        deltas, missing = {}, []
        for topic, rows in self.streams.items():
            if measured_only and not topic.endswith("state"):
                continue
            i = nearest_index(self.times[topic], t)
            observed = self.times[topic][i]
            deltas[topic] = observed - t
            if abs(observed - t) > MAX_DELTA_S:
                missing.append(topic)
                continue
            side = topic.split("-")[0][1:]
            sample.setdefault(side, {}).update(rows[i])
            sample["source_times_s"][topic] = observed
        if missing:
            sample["missing_streams"] = missing
        return sample, {"method": "nearest_mcap_host_event_time", "capture_t_s": t,
                        "delta_s_by_topic": deltas, "max_delta_s": MAX_DELTA_S,
                        "missing_streams": missing, "exposure_synchronized": False}

    def attach(self, keyframes):
        for i, frame in enumerate(keyframes):
            captured = self.capture_times[frame["video_frame_index"]]
            state, frame["alignment"] = self._at(captured, measured_only=True)
            frame["state"] = {side: state[side] for side in ("left", "right") if side in state} or None
            if i + 1 == len(keyframes):
                continue
            end = self.capture_times[keyframes[i + 1]["video_frame_index"]]
            samples = self.samples[bisect_left(self.state_times, captured):bisect_right(self.state_times, end)]
            frame["action"] = {"kind": "recorded_bimanual_segment", "time_basis": "media_origin_elapsed_s",
                               "samples": samples,
                               "max_sample_gap_s": max((b["t_s"] - a["t_s"] for a, b in zip(samples, samples[1:])), default=0)}

    def event_pts(self):
        return sorted({self.pts[nearest_index(self.capture_times, t)] for t in event_times(self.samples)
                       if self.capture_times[0] <= t <= self.capture_times[-1]})

    def verify_sources(self):
        if any(_sha256_file(self.root / name) != digest for name, digest in self.source_hashes.items()):
            raise RuntimeError("Recording changed during demonstration preparation")


def _values(topic, message, metadata):
    position = list(message.position)
    pose = list(getattr(message, "pose", []))
    arm = "-arm-" in topic
    measured = topic.endswith("state")
    widths = (6, 7) if arm and measured else (6,) if arm else (1,)
    if len(position) not in widths or len(pose) not in (0, 7):
        raise ValueError(f"Unexpected YAM position/pose dimensions: {topic}")
    if any(not math.isfinite(x) for x in position + pose):
        raise ValueError(f"Nonfinite YAM recording data: {topic}")
    if not arm:
        return {"gripper_measured" if measured else "gripper_command": position[0]}
    result = {"joint_measured_rad" if measured else "joint_target_rad": position[:6]}
    if pose:
        if metadata.get("robot_state_pose_format") != "ee_pos_quat=[x,y,z,qw,qx,qy,qz]":
            raise ValueError("Unknown YAM pose convention; quaternion order is unverified")
        if sum(x * x for x in pose[3:]) < 1e-12:
            raise ValueError(f"Zero quaternion in YAM recording: {topic}")
        result["eef_measured_xyz_xyzw" if measured else "eef_target_xyz_xyzw"] = pose[:3] + pose[4:] + pose[3:4]
    return result
