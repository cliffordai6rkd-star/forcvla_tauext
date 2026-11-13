import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0_force
from openpi.models import siglip
from openpi.shared import nnx_utils


@pytest.fixture
def small_model(monkeypatch):
    # Exercise the real Gemma KV cache, vision encoder and MoE with small
    # parameter sizes. The production checkpoint structure is unchanged.
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 64)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(
            width=16, depth=1, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8,
        ),
    )
    vision_factory = siglip.Module
    monkeypatch.setattr(
        siglip, "Module", lambda **kwargs: vision_factory(**{**kwargs, "variant": "mu/112"}),
    )
    config = pi0_force.Pi0_GuidanceConfig(action_horizon=2, max_token_len=4)
    model = config.create(jax.random.key(0))
    model.eval()
    return config, model


def _uncached_sample(model, rng, observation, num_steps):
    """Reference sampler recomputing prefix and suffix together at each step."""
    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    actions = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    dt = -1.0 / num_steps
    for index in range(num_steps):
        timestep = jnp.full((batch_size,), 1.0 + index * dt)
        suffix, suffix_mask, suffix_ar, force = model.embed_suffix(observation, actions, timestep)
        mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar = jnp.concatenate([prefix_ar, suffix_ar])
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [prefix, suffix], positions=jnp.cumsum(mask, axis=1) - 1,
            mask=pi0_force.make_attn_mask(mask, ar),
        )
        guidance, _ = model.limoe(jnp.concatenate([prefix_out, force], axis=1))
        velocity = model.action_out_proj(
            guidance[:, -model.action_horizon:] + suffix_out[:, -model.action_horizon:],
        )
        actions = actions + dt * velocity
    return actions


@pytest.mark.parametrize("batch_size", [1, 2])
def test_cached_force_sampling_matches_full_forward(small_model, batch_size):
    # Fully valid prefix tokens cannot attend to the suffix, so prefill and
    # full-forward outputs agree. Masked prefix rows have different semantics
    # in the existing Gemma attention and are covered for runtime safety below.
    config, model = small_model
    observation = config.fake_obs(batch_size)
    observation = dataclasses.replace(
        observation,
        state=observation.state.at[:, 7:14].set(jnp.arange(1, 8, dtype=jnp.float32)),
    )
    rng = jax.random.key(1)
    sample = nnx_utils.module_jit(model.sample_actions)
    cached = sample(rng, observation, num_steps=2)
    reference = _uncached_sample(model, rng, observation, num_steps=2)
    assert cached.shape == (batch_size, config.action_horizon, config.action_dim)
    assert np.isfinite(cached).all()
    np.testing.assert_allclose(cached, reference, atol=0.03, rtol=0.03)

    without_force = sample(
        rng, dataclasses.replace(observation, state=observation.state.at[:, 7:14].set(0)), num_steps=2,
    )
    assert np.max(np.abs(cached - without_force)) > 1e-4


def test_single_sample_with_missing_camera_and_prompt_padding(small_model):
    config, model = small_model
    observation = config.fake_obs()
    observation = dataclasses.replace(
        observation,
        image_masks={**observation.image_masks, "left_wrist_0_rgb": jnp.zeros(1, dtype=bool)},
        tokenized_prompt_mask=observation.tokenized_prompt_mask.at[:, -1].set(False),
    )
    actions = nnx_utils.module_jit(model.sample_actions)(jax.random.key(3), observation, num_steps=2)
    assert actions.shape == (1, config.action_horizon, config.action_dim)
    assert np.isfinite(actions).all()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_force_loss_accepts_small_batches(small_model, batch_size):
    config, model = small_model
    loss = nnx_utils.module_jit(model.compute_loss)(
        jax.random.key(2), config.fake_obs(batch_size), config.fake_act(batch_size),
    )
    assert loss.shape == (batch_size, config.action_horizon)
    assert np.isfinite(loss).all()
