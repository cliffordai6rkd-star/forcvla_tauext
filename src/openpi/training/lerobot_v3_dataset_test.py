import dataclasses
import json
import pickle

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.policies import forcevla_policy
from openpi.training.lerobot_v3_dataset import LeRobotV3Dataset, LeRobotV3Options


@pytest.fixture
def v3_root(tmp_path):
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    video_dir = tmp_path / "videos/observation.images.front/chunk-000"
    video_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0", "fps": 25, "total_frames": 4, "total_episodes": 2,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.joint": {"dtype": "float32", "shape": [7]},
            "action.joint": {"dtype": "float32", "shape": [7]},
            "observation.tau_ext": {"dtype": "float32", "shape": [7]},
            "observation.images.front": {"dtype": "video", "shape": [16, 16, 3]},
        },
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    rows = [{
        "index": i, "episode_index": i // 2, "timestamp": (i % 2) / 25, "task_index": 0,
        "observation.joint": [float(i)] * 7,
        "action.joint": [float(i + 10)] * 7,
        "observation.tau_ext": [float(i + 20)] * 7,
    } for i in range(4)]
    # Reversed physical shard order exercises indexing by the global data index.
    pq.write_table(pa.Table.from_pylist(rows[2:]), tmp_path / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:2]), tmp_path / "data/chunk-000/file-001.parquet")
    episodes = [{
        "episode_index": i, "length": 2, "dataset_from_index": 2 * i, "dataset_to_index": 2 * i + 2,
        "tasks": ["source task"],
        "videos/observation.images.front/chunk_index": 0,
        "videos/observation.images.front/file_index": 0,
        "videos/observation.images.front/from_timestamp": 2 * i / 25,
        "videos/observation.images.front/to_timestamp": (2 * i + 2) / 25,
    } for i in range(2)]
    pq.write_table(pa.Table.from_pylist(episodes), tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "__index_level_0__": "source task"}]),
                   tmp_path / "meta/tasks.parquet")
    with av.open(str(video_dir / "file-000.mp4"), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for intensity in [20, 70, 120, 170]:
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), intensity, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return tmp_path


def _options(root, **kwargs):
    return LeRobotV3Options(str(root), image_keys={"image": "observation.images.front"}, **kwargs)


def test_episode_padding_current_torque_and_all_joint_deltas(v3_root):
    dataset = LeRobotV3Dataset(_options(v3_root), action_horizon=3, load_images=False)
    assert len(dataset) == 4
    np.testing.assert_array_equal(dataset[1]["actions"], np.full((3, 7), 11))
    np.testing.assert_array_equal(dataset[1]["tau_ext"], np.full(7, 21))
    assert dataset[1]["prompt"] == "source task"
    inputs = forcevla_policy.Forcevla_inputs(32)(dataset[1])
    inputs = transforms.DeltaActions(transforms.make_bool_mask(7))(inputs)
    np.testing.assert_array_equal(inputs["actions"][:, :7], 10)
    np.testing.assert_array_equal(inputs["state"][7:14], 21)
    assert not inputs["image_mask"]["left_wrist_0_rgb"]
    assert not inputs["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_array_equal(dataset._actions[1], 11)  # transforms must not modify cached arrays


def test_shared_video_offset_frame_precision_and_pickling(v3_root):
    dataset = LeRobotV3Dataset(_options(v3_root), action_horizon=3)
    # Episode 1 starts inside the shared video, not at its first frame.
    for index, expected in enumerate([20, 70, 120, 170]):
        image = dataset[index]["image"]
        assert image.shape == (16, 16, 3)
        assert abs(float(image.mean()) - expected) < 6
    assert len(dataset._decoders) == 1
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._decoders
    assert abs(float(restored[2]["image"].mean()) - 120) < 6


def test_prompt_override_and_no_video_io_for_statistics(v3_root):
    for path in (v3_root / "videos").rglob("*.mp4"):
        path.unlink()
    dataset = LeRobotV3Dataset(_options(v3_root, prompt="Wipe the board."), 3, load_images=False)
    assert dataset[0]["prompt"] == "Wipe the board."
    assert dataset[0]["image"].shape == (1, 1, 3)
    assert not dataset._decoders


def test_reject_missing_tau_ext(v3_root):
    options = dataclasses.replace(_options(v3_root), tau_ext_key="observation.wrench")
    with pytest.raises(ValueError, match="must be a 7-dimensional feature"):
        LeRobotV3Dataset(options, 3, load_images=False)


def test_real_right_wrist_camera_is_used():
    example = forcevla_policy.make_forcevla_example()
    del example["wrist_image"]
    example["right_wrist_image"] = np.full((224, 224, 3), 123, dtype=np.uint8)
    inputs = forcevla_policy.Forcevla_inputs(32)(example)
    assert inputs["image_mask"]["right_wrist_0_rgb"]
    assert not inputs["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(inputs["image"]["right_wrist_0_rgb"], 123)
