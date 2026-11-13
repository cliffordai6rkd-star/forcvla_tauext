from flaxformer.architectures.moe import routing
from flaxformer.components import dense
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import limoe_simple


def _make_moe():
    return limoe_simple.MoeLayer(
        num_experts=4,
        max_group_size=16,
        train_capacity_factor=1.0,
        eval_capacity_factor=1.0,
        expert=dense.MlpBlock(
            intermediate_dim=8,
            use_bias=True,
            intermediate_dropout_rate=0.0,
            final_dropout_rate=0.0,
        ),
        router=routing.TokensChooseScatterRouter(
            router_weights=routing.RouterWeights(),
            jitter_noise=0.0,
            dtype=jnp.float32,
            ignore_padding_tokens=True,
            num_selected_experts=1,
            batch_prioritized_routing=True,
        ),
        dtype=jnp.float32,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_moe_817_tokens_forward_and_gradients(batch_size):
    # Three image grids, prompt, and the force token produce 817 routing tokens.
    layer = _make_moe()
    inputs = jax.random.normal(jax.random.key(0), (batch_size, 817, 4))
    variables = layer.init(jax.random.key(1), inputs, enable_dropout=False)

    def loss_fn(params, values):
        outputs = layer.apply({"params": params}, values, enable_dropout=False)
        return jnp.mean(jnp.square(outputs)), outputs

    (loss, outputs), gradients = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(
        variables["params"], inputs
    )
    assert outputs.shape == inputs.shape
    assert np.isfinite(loss)
    assert np.isfinite(outputs).all()
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradients))
    assert np.any(np.asarray(gradients[1]) != 0)

    if batch_size == 4:
        # Compare the divisible batch against the existing dispatch path, with
        # its original groups and capacities and no added routing tokens.
        grouped = inputs.reshape((3268, 1, 4))
        expected = layer.apply(
            variables,
            grouped,
            False,
            4,
            decode=False,
            prefill=False,
            prefill_lengths=None,
            method=layer._scatter_to_experts,
        ).reshape(inputs.shape)
        uncompiled_outputs = layer.apply(variables, inputs, enable_dropout=False)
        np.testing.assert_array_equal(uncompiled_outputs, expected)


def test_limoe_single_sample_preserves_sequence_length():
    block = limoe_simple.LIMoEBlock(mlp_dim=8, num_heads=2, out_dim=4)
    inputs = jax.random.normal(jax.random.key(2), (1, 817, 8))
    variables = block.init(jax.random.key(3), inputs, deterministic=True)
    outputs, _ = block.apply(variables, inputs, deterministic=True)
    assert outputs.shape == (1, 817, 4)
    assert np.isfinite(outputs).all()


def test_single_sample_keeps_all_tokens_routed_to_same_expert():
    layer = _make_moe()
    # Equal nonzero tokens choose the same expert. Five-token groups with
    # capacity four would silently drop one token from every full group.
    inputs = jnp.broadcast_to(jnp.array([1.0, 0.5, -0.2, 0.8]), (1, 817, 4))
    variables = layer.init(jax.random.key(4), inputs, enable_dropout=False)
    outputs = layer.apply(variables, inputs, enable_dropout=False)

    # The original batch-four training path dispatches one token per group,
    # so the same input must keep that capacity-independent routing behavior.
    padded = jnp.pad(inputs.reshape((817, 4)), ((0, 3), (0, 0)))
    expected = layer.apply(
        variables,
        padded.reshape((820, 1, 4)),
        False,
        4,
        decode=False,
        prefill=False,
        prefill_lengths=None,
        method=layer._scatter_to_experts,
    ).reshape((820, 4))[:817].reshape(inputs.shape)
    assert np.all(np.any(np.asarray(expected) != 0, axis=-1))
    np.testing.assert_array_equal(outputs, expected)


def test_moe_inference_is_consistent_between_batch_one_and_four():
    layer = _make_moe()
    inputs = jnp.broadcast_to(jax.random.normal(jax.random.key(5), (1, 1, 4)), (1, 817, 4))
    variables = layer.init(jax.random.key(6), inputs, enable_dropout=False)
    single = layer.apply(variables, inputs, enable_dropout=False)
    batched = layer.apply(variables, jnp.repeat(inputs, 4, axis=0), enable_dropout=False)
    assert np.all(np.any(np.asarray(batched) != 0, axis=-1))
    for output in batched:
        np.testing.assert_array_equal(single[0], output)


def test_batch_four_training_retains_original_dispatch():
    layer = _make_moe()
    inputs = jax.random.normal(jax.random.key(7), (4, 817, 4))
    variables = layer.init(jax.random.key(8), inputs, enable_dropout=True)
    outputs = layer.apply(variables, inputs, enable_dropout=True)
    expected = layer.apply(
        variables,
        inputs.reshape((3268, 1, 4)),
        True,
        4,
        decode=False,
        prefill=False,
        prefill_lengths=None,
        method=layer._scatter_to_experts,
    ).reshape(inputs.shape)
    np.testing.assert_array_equal(outputs, expected)


@pytest.mark.parametrize(
    ("num_tokens", "max_group_size", "num_experts", "expected_groups"),
    [(72, 16, 4, 8), (24, 6, 4, 4), (8, 16, 4, 4), (820, 16, 4, 164), (3268, 16, 4, 3268)],
)
def test_num_groups_respects_maximum_size(num_tokens, max_group_size, num_experts, expected_groups):
    groups = limoe_simple._num_groups(num_tokens, max_group_size, num_experts)
    assert groups == expected_groups
    assert num_tokens % groups == 0
    assert groups % num_experts == 0
    assert num_tokens // groups <= max_group_size


@pytest.mark.parametrize("num_tokens", [1, 817, 1634])
def test_num_groups_rejects_nondivisible_token_counts(num_tokens):
    with pytest.raises(ValueError, match="number of experts"):
        limoe_simple._num_groups(num_tokens, 16, 4)


@pytest.mark.parametrize("arguments", [(0, 16, 4), (4, 0, 4), (4, 16, 0)])
def test_num_groups_rejects_nonpositive_sizes(arguments):
    with pytest.raises(ValueError, match="must be positive"):
        limoe_simple._num_groups(*arguments)
