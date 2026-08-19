import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_rtc_prefix_weights_never_guide_padding():
    weights = _policy._rtc_prefix_weights(  # noqa: SLF001
        30,
        prefix_len=8,
        decay_steps=12,
        schedule="exp",
        target_len=8,
    )

    np.testing.assert_array_equal(weights[:8], np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(weights[8:], np.zeros(22, dtype=np.float32))


def test_rtc_prefix_weights_decay_only_over_real_remaining_targets():
    weights = _policy._rtc_prefix_weights(  # noqa: SLF001
        30,
        prefix_len=8,
        decay_steps=4,
        schedule="exp",
        target_len=12,
    )

    np.testing.assert_allclose(weights[:8], 1.0)
    np.testing.assert_allclose(weights[8:12], np.exp(-np.arange(4, dtype=np.float32)), rtol=1e-6)
    np.testing.assert_array_equal(weights[12:], np.zeros(18, dtype=np.float32))


def test_prepare_rtc_prefix_preserves_targets_beyond_lock():
    remaining = np.arange(15 * 16, dtype=np.float32).reshape(15, 16)

    prefix, weights, lock_len, target_len = _policy._prepare_rtc_prefix(  # noqa: SLF001
        remaining,
        action_horizon=30,
        action_dim=32,
        lock_len=8,
        decay_steps=4,
        schedule="exp",
    )

    assert lock_len == 8
    assert target_len == 15
    np.testing.assert_array_equal(prefix[:15, :16], remaining)
    np.testing.assert_array_equal(prefix[:15, 16:], 0.0)
    np.testing.assert_array_equal(prefix[15:], 0.0)
    assert np.all(weights[:12] > 0.0)
    np.testing.assert_array_equal(weights[12:], 0.0)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
