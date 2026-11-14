import dataclasses
from unittest import mock

import numpy as np
import pytest

from scripts import serve_policy


@pytest.mark.parametrize("task,prompt", [("board", "Wipe the board."), ("huanggua", "Peel the cucumber.")])
def test_local_policy_warms_before_returning_with_training_camera_slots(task, prompt):
    policy = mock.Mock()
    args = serve_policy.Args(policy=serve_policy.Checkpoint(f"pi0_force_lora_{task}", "/checkpoint"))
    with mock.patch.object(serve_policy._policy_config, "create_trained_policy", return_value=policy):
        assert serve_policy.create_policy(args) is policy
    observation = policy.warmup.call_args.args[0]
    assert observation["prompt"] == prompt
    assert "wrist_image" not in observation
    assert observation["right_wrist_image"].shape == (224, 224, 3)
    assert observation["right_wrist_image"].dtype == np.uint8
    assert observation["state"].shape == observation["tau_ext"].shape == (7,)


def test_warmup_can_be_disabled_and_explicit_prompt_is_used():
    policy = mock.Mock()
    args = serve_policy.Args(
        policy=serve_policy.Checkpoint("pi0_force_lora_board", "/checkpoint"),
        default_prompt="Custom task.",
    )
    with mock.patch.object(serve_policy._policy_config, "create_trained_policy", return_value=policy):
        serve_policy.create_policy(args)
        assert policy.warmup.call_args.args[0]["prompt"] == "Custom task."
        policy.reset_mock()
        serve_policy.create_policy(dataclasses.replace(args, warmup=False))
        policy.warmup.assert_not_called()


def test_nonlocal_policy_does_not_receive_local_camera_warmup():
    policy = mock.Mock()
    args = serve_policy.Args(policy=serve_policy.Checkpoint("pi0_aloha_sim", "/checkpoint"))
    with mock.patch.object(serve_policy._policy_config, "create_trained_policy", return_value=policy):
        serve_policy.create_policy(args)
    policy.warmup.assert_not_called()
