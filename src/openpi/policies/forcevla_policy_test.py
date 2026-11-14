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


def _deployment_observation():
    return {
        "state": np.zeros(7, dtype=np.float32),
        "tau_ext": np.zeros(7, dtype=np.float32),
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "right_wrist_image": np.full((224, 224, 3), 128, dtype=np.uint8),
    }


@pytest.mark.parametrize("tau_ext,expected", [
    ([1] * 7, [0] * 7),
    ([-2] * 7, [-2] * 7),  # equality is retained, including signs
    ([3, -3, 3, -3, 3, -3, 3], [3, -3, 3, -3, 3, -3, 3]),
])
def test_serving_contact_gate_is_idempotent_and_precedes_normalization(tau_ext, expected):
    validate = forcevla_policy.ForcevlaServingInputs(("image", "right_wrist_image"), 14.0)
    observation = {**_deployment_observation(), "tau_ext": np.asarray(tau_ext, dtype=np.float32)}
    gated = validate(observation)
    np.testing.assert_array_equal(gated["tau_ext"], expected)
    np.testing.assert_array_equal(validate(gated)["tau_ext"], expected)
    np.testing.assert_array_equal(observation["tau_ext"], tau_ext)
    inputs = forcevla_policy.Forcevla_inputs(32)(gated)
    np.testing.assert_array_equal(inputs["state"][7:14], expected)
    assert not inputs["image_mask"]["left_wrist_0_rgb"]
    assert inputs["image_mask"]["right_wrist_0_rgb"]


@pytest.mark.parametrize("bad_input", ["missing_right", "left_present"])
def test_serving_rejects_camera_slot_mismatch(bad_input):
    observation = _deployment_observation()
    observation["wrist_image"] = observation["right_wrist_image"]
    if bad_input == "missing_right":
        del observation["right_wrist_image"]
    with pytest.raises(ValueError, match="right_wrist_image"):
        forcevla_policy.ForcevlaServingInputs(("image", "right_wrist_image"))(observation)


@pytest.mark.parametrize("field", ["state", "tau_ext"])
@pytest.mark.parametrize("values", [np.full(7, np.nan), np.full(7, np.inf), np.ones((1, 7))])
def test_serving_rejects_nonfinite_or_batched_state(field, values):
    observation = {**_deployment_observation(), field: values}
    with pytest.raises(ValueError, match=field):
        forcevla_policy.ForcevlaServingInputs(("image", "right_wrist_image"))(observation)


@pytest.mark.parametrize("image", [
    np.full((8, 8, 3), 255.0),
    np.full((8, 8, 3), -1.0),
    np.full((8, 8, 3), np.nan),
    np.zeros((1, 8, 8, 3), dtype=np.uint8),
    np.zeros((8, 8, 4), dtype=np.uint8),
    np.zeros((8, 8, 3), dtype=np.uint16),
])
def test_reject_invalid_rgb_frame(image):
    observation = {**_deployment_observation(), "image": image}
    with pytest.raises(ValueError, match="[Ii]mages|RGB"):
        forcevla_policy.Forcevla_inputs(32)(observation)


def test_rgb_float_chw_and_uint8_hwc_represent_same_image():
    observation = _deployment_observation()
    image = observation["right_wrist_image"]
    as_float = image.transpose(2, 0, 1).astype(np.float32) / 255
    converted = forcevla_policy.Forcevla_inputs(32)({**observation, "right_wrist_image": as_float})
    np.testing.assert_array_equal(converted["image"]["right_wrist_0_rgb"], image)


def test_reject_nonfinite_output_actions():
    with pytest.raises(ValueError, match="nonfinite"):
        forcevla_policy.Forcevla_outputs()({"actions": np.full((50, 32), np.nan)})
