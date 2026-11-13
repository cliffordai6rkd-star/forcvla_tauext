"""Local LeRobot v3 reader for ForceVLA, without changing the legacy LeRobot install."""

from collections import OrderedDict
import dataclasses
import json
import operator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclasses.dataclass(frozen=True)
class LeRobotV3Options:
    root: str
    state_key: str = "observation.joint"
    action_key: str = "action.joint"
    tau_ext_key: str = "observation.tau_ext"
    image_keys: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "image": "observation.images.front",
        "right_wrist_image": "observation.images.right_wrist",
    })
    # Override incorrect source task labels, or use the episode/task metadata when None.
    prompt: str | None = None


class LeRobotV3Dataset:
    """Read numeric Parquet columns and seek the matching frames in v3 video shards.

    Action chunks start at the current frame and repeat the final action at episode
    boundaries. Only the current tau_ext is exposed; future torque is never read.
    Decoders are CPU-only, cached per worker, and excluded from pickled datasets.
    """

    def __init__(self, options: LeRobotV3Options, action_horizon: int, *, load_images: bool = True):
        self.options = options
        self.root = Path(options.root).expanduser().resolve()
        self.action_horizon = action_horizon
        self.load_images = load_images
        self._decoders = OrderedDict()
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        with (self.root / "meta/info.json").open() as f:
            self.info = json.load(f)
        if not str(self.info["codebase_version"]).startswith("v3"):
            raise ValueError(f"Expected LeRobot v3 at {self.root}")
        self.fps = float(self.info["fps"])
        for key in (options.state_key, options.action_key, options.tau_ext_key):
            if self.info["features"].get(key, {}).get("shape") != [7]:
                raise ValueError(f"{key} must be a 7-dimensional feature at {self.root}")
        for key in options.image_keys.values():
            if self.info["features"].get(key, {}).get("dtype") != "video":
                raise ValueError(f"Missing video feature {key} at {self.root}")

        episode_files = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No v3 episode metadata at {self.root}")
        episode_columns = [
            "episode_index", "length", "dataset_from_index", "dataset_to_index", "tasks",
            *[f"videos/{key}/{field}" for key in options.image_keys.values()
              for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")],
        ]
        self.episodes = sorted(
            [row for path in episode_files
             for row in pq.read_table(path, columns=episode_columns).to_pylist()],
            key=lambda row: row["dataset_from_index"],
        )
        self._starts = np.asarray([row["dataset_from_index"] for row in self.episodes])
        self._ends = np.asarray([row["dataset_to_index"] for row in self.episodes])
        if (self._starts[0] != 0 or np.any(self._ends <= self._starts)
                or np.any(self._starts[1:] != self._ends[:-1])):
            raise ValueError("Episode frame ranges must be nonempty and contiguous")
        if any(row["length"] != end - start
               for row, start, end in zip(self.episodes, self._starts, self._ends, strict=True)):
            raise ValueError("Episode lengths disagree with their frame ranges")

        data_files = sorted((self.root / "data").rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No v3 data Parquet files at {self.root}")
        columns = list(dict.fromkeys([
            "index", "episode_index", "timestamp", "task_index",
            options.state_key, options.action_key, options.tau_ext_key,
        ]))
        table = pa.concat_tables([pq.read_table(path, columns=columns) for path in data_files])
        table = table.sort_by("index")
        expected_frames = int(self.info["total_frames"])
        if (len(table) != expected_frames or self._ends[-1] != expected_frames
                or not np.array_equal(table["index"].to_numpy(), np.arange(expected_frames))):
            raise ValueError("Data indices, episode ranges, and total_frames disagree")
        if len(self.episodes) != int(self.info["total_episodes"]):
            raise ValueError("Episode count disagrees with info.json")
        episode_ids = table["episode_index"].to_numpy()
        for row, start, end in zip(self.episodes, self._starts, self._ends, strict=True):
            if np.any(episode_ids[start:end] != row["episode_index"]):
                raise ValueError("Data episode_index disagrees with episode metadata")
        self._states = np.asarray(table[options.state_key].to_pylist(), dtype=np.float32)
        self._actions = np.asarray(table[options.action_key].to_pylist(), dtype=np.float32)
        self._tau_ext = np.asarray(table[options.tau_ext_key].to_pylist(), dtype=np.float32)
        self._timestamps = table["timestamp"].to_numpy()
        self._task_indices = table["task_index"].to_numpy()
        for key, values in ((options.state_key, self._states), (options.action_key, self._actions),
                            (options.tau_ext_key, self._tau_ext)):
            if values.shape != (expected_frames, 7) or not np.isfinite(values).all():
                raise ValueError(f"Invalid shape or nonfinite values in {key}")
        self._tasks = {}
        tasks_path = self.root / "meta/tasks.parquet"
        if tasks_path.exists():
            for row in pq.read_table(tasks_path).to_pylist():
                self._tasks[int(row["task_index"])] = row.get("task", row.get("__index_level_0__", ""))

    def __len__(self):
        return len(self._states)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_decoders"] = OrderedDict()
        return state

    def _decode_image(self, key: str, episode: dict, timestamp: float) -> np.ndarray:
        if not self.load_images:
            # Normalization only needs state/actions, so avoid all video IO and large RGB batches.
            return np.zeros((1, 1, 3), dtype=np.uint8)
        from torchcodec.decoders import VideoDecoder

        prefix = f"videos/{key}"
        path = self.root / self.info["video_path"].format(
            video_key=key,
            chunk_index=episode[f"{prefix}/chunk_index"],
            file_index=episode[f"{prefix}/file_index"],
        )
        if path not in self._decoders:
            self._decoders[path] = VideoDecoder(
                path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=1,
            )
        self._decoders.move_to_end(path)
        if len(self._decoders) > 8:
            self._decoders.popitem(last=False)
        # Timestamps in the data restart per episode, while videos concatenate episodes.
        # A small epsilon prevents float32 timestamps selecting the preceding 25 Hz frame.
        video_time = float(episode[f"{prefix}/from_timestamp"]) + timestamp + 1e-6
        return self._decoders[path].get_frame_played_at(video_time).data.numpy()

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = int(np.searchsorted(self._ends, index, side="right"))
        episode = self.episodes[episode_position]
        indices = np.minimum(np.arange(index, index + self.action_horizon), self._ends[episode_position] - 1)
        prompt = self.options.prompt
        if prompt is None:
            prompt = self._tasks.get(int(self._task_indices[index]))
            if not prompt:
                prompt = next(iter(episode["tasks"]), "")
        return {
            "state": self._states[index].copy(),
            "tau_ext": self._tau_ext[index].copy(),
            "actions": self._actions[indices].copy(),
            "prompt": str(prompt),
            **{target: self._decode_image(source, episode, float(self._timestamps[index]))
               for target, source in self.options.image_keys.items()},
        }
