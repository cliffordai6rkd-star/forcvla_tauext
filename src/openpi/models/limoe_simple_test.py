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
