import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_force
from openpi.policies import forcevla_policy
from openpi.shared import normalize
from openpi.training import config


def test_tau_ext_dataset_route_and_normalization(tmp_path):
    train_config = config.get_config("forcevla_lora")
    data_config = train_config.data.create(tmp_path, train_config.model)
    example = forcevla_policy.make_forcevla_example()
    proprio = np.arange(7, dtype=np.float32)
    tau_ext = np.arange(7, dtype=np.float32) + 10
    raw = {
        "observation.image": example["image"],
        "observation.wrist_image": example["wrist_image"],
        # A legacy wrench must not leak into either input branch.
        "observation.state": np.concatenate([proprio, np.full(6, 999)]),
        "observation.tau_ext": tau_ext,
        "action": np.ones((2, 7), dtype=np.float32),
        "prompt": "test",
    }
    inputs = transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ])(raw)
    np.testing.assert_array_equal(inputs["state"][:7], proprio)
    np.testing.assert_array_equal(inputs["state"][7:14], tau_ext)
    np.testing.assert_array_equal(inputs["state"][14:], 0)
    # Both proprioception and torques participate in the existing state statistics.
    stats = normalize.RunningStats()
    stats.update(np.stack([inputs["state"], inputs["state"] + 2]))
    normalized = transforms.Normalize({"state": stats.get_statistics()})(inputs)
    np.testing.assert_allclose(normalized["state"][7:14], -1, atol=1e-5)


@pytest.mark.parametrize("size", [6, 8])
def test_reject_wrong_tau_ext_dimension(size):
    example = forcevla_policy.make_forcevla_example()
    example["tau_ext"] = np.ones(size)
    with pytest.raises(ValueError, match="tau_ext must have 7 channels"):
        forcevla_policy.Forcevla_inputs(action_dim=32)(example)


def test_missing_tau_ext_does_not_use_legacy_wrench():
    example = forcevla_policy.make_forcevla_example()
    del example["tau_ext"]
    example["state"] = np.ones(13)
    with pytest.raises(KeyError, match="tau_ext"):
        forcevla_policy.Forcevla_inputs(action_dim=32)(example)


def test_seventh_joint_reaches_guidance_without_changing_proprioception():
    model_config = pi0_force.Pi0_GuidanceConfig(
        paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2,
    )
    abstract_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    assert abstract_model.force_in_proj.kernel.value.shape[0] == 7

    # Exercise the real suffix routing with small projections, without allocating the VLM.
    class SuffixModel(pi0_force.Pi0_Guidance):
        def __init__(self):
            model_lib.BaseModel.__init__(self, 32, 2, 48)

    suffix_model = SuffixModel()
    rngs = nnx.Rngs(0)
    suffix_model.state_proj = nnx.Linear(32, 4, rngs=rngs)
    suffix_model.action_in_proj = nnx.Linear(32, 4, rngs=rngs)
    suffix_model.action_time_mlp_in = nnx.Linear(8, 4, rngs=rngs)
    suffix_model.action_time_mlp_out = nnx.Linear(4, 4, rngs=rngs)
    suffix_model.force_in_proj = nnx.Linear(abstract_model.force_in_proj.in_features, 8, rngs=rngs)
    suffix_model.force_in_proj.kernel.value = jnp.ones((7, 8))
    observation = model_lib.Observation(images={}, image_masks={}, state=jnp.zeros((1, 32)))
    actions = jnp.zeros((1, 2, 32))
    time = jnp.ones((1,))
    original = suffix_model.embed_suffix(observation, actions, time)
    changed = dataclasses.replace(observation, state=observation.state.at[:, 13].set(3))
    updated = suffix_model.embed_suffix(changed, actions, time)
    np.testing.assert_array_equal(updated[0], original[0])
    np.testing.assert_allclose(updated[3] - original[3], 3)
    # All seven torques receive gradients through the injection projection.
    grad = jax.grad(lambda state: suffix_model.embed_suffix(
        dataclasses.replace(observation, state=state), actions, time,
    )[3].sum())(observation.state)
    np.testing.assert_array_equal(grad[:, 7:14], 8)
    np.testing.assert_array_equal(grad[:, :7], 0)
    np.testing.assert_array_equal(grad[:, 14:], 0)
