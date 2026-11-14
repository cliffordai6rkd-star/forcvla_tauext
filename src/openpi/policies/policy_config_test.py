from types import SimpleNamespace

import flax.nnx as nnx
import jax
import numpy as np
import pytest

from openpi import transforms
from openpi.policies import forcevla_policy
from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.training import config


class _Model(nnx.Module):
    def sample_actions(self, rng, observation):
        return jax.random.normal(rng, (observation.state.shape[0], 2, 32))


@pytest.mark.parametrize("prompt,expected_prompt", [(None, "Wipe the board."), ("Custom task.", "Custom task.")])
def test_local_policy_uses_checkpoint_statistics_prompt_and_contact_gate(monkeypatch, tmp_path, prompt, expected_prompt):
    cfg = config.get_config("pi0_force_lora_board")
    monkeypatch.setattr(type(cfg.model), "load", lambda *args: _Model())
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    data_cfg = SimpleNamespace(
        asset_id="fixture",
        lerobot_v3=SimpleNamespace(image_keys={"image": "front", "right_wrist_image": "right"}),
        data_transforms=transforms.Group(inputs=[forcevla_policy.Forcevla_inputs(32)]),
        model_transforms=transforms.Group(),
        use_quantile_norm=False,
    )
    monkeypatch.setattr(type(cfg.data), "create", lambda *args: data_cfg)
    loaded_paths = []

    def load_stats(path, asset_id):
        loaded_paths.append((str(path), asset_id))
        return {"state": transforms.NormStats(mean=np.arange(32), std=np.full(32, 2.0))}

    monkeypatch.setattr(policy_config._checkpoints, "load_norm_stats", load_stats)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt=prompt)
    observation = {
        "state": np.zeros(7, dtype=np.float32),
        "tau_ext": np.ones(7, dtype=np.float32),
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "right_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    transformed = policy._input_transform(observation)
    assert transformed["prompt"] == expected_prompt
    np.testing.assert_allclose(transformed["state"][7:14], -np.arange(7, 14) / (2 + 1e-6))
    assert loaded_paths == [(str(checkpoint / "assets"), "fixture")]
    assert policy.metadata["tau_ext_l1_threshold_nm"] == 14.0
    assert policy.metadata["action_representation"] == "absolute_joint_positions"
    explicit = policy._input_transform({**observation, "prompt": "Client task."})
    assert explicit["prompt"] == "Client task."


def test_warmup_preserves_first_live_prediction_rng():
    observation = {
        "state": np.zeros(32, dtype=np.float32),
        "image": {"base_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.True_},
    }
    policy = policy_lib.Policy(_Model())
    before = np.asarray(jax.random.key_data(policy._rng)).copy()
    policy.warmup(observation)
    np.testing.assert_array_equal(jax.random.key_data(policy._rng), before)
    prediction = policy.infer(observation)["actions"]
    fresh_prediction = policy_lib.Policy(_Model()).infer(observation)["actions"]
    np.testing.assert_array_equal(prediction, fresh_prediction)
