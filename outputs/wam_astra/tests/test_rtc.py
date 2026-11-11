"""Boundary tests against the unmodified, pinned LeRobot queue (CPU only)."""
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from threading import Thread

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "work/wam_demo_library/vendor"))
SOURCE = WORKSPACE / "work/upstream/lerobot"


def test_adapter_module_exists():
    assert importlib.util.find_spec("wam_astra.rtc") is not None, "RTC adapter must exist"


@pytest.fixture
def rtc():
    assert importlib.util.find_spec("wam_astra.rtc") is not None, "RTC adapter must exist"
    return importlib.import_module("wam_astra.rtc")


@pytest.fixture
def queue(rtc):
    return rtc.ActionQueue(source_root=SOURCE)


@dataclass(frozen=True)
class Source:
    task: str = "pick"
    request_id: str = "review-1"
    target_at_s: float = 100.0


def actions(value=0.0, horizon=3):
    result = np.full((horizon, 7), value, dtype=np.float32)
    result[:, -1] = 0
    return result


def seed(queue, value=0.0):
    assert queue.merge(actions(value), actions(value + 0.1), 0,
                       snapshot=queue.snapshot(), task="pick", provenance=Source())


def test_adapter_executes_real_verified_upstream_classes(rtc, queue):
    assert queue.native.__class__.__name__ == "ActionQueue"
    assert Path(importlib.import_module(queue.native.__class__.__module__).__file__).resolve() == (
        SOURCE / "src/lerobot/policies/rtc/action_queue.py").resolve()
    assert rtc.UPSTREAM_COMMIT == "d490e472a5d1a207cfe0255caaaf74b6d66e4656"
    assert rtc.load_upstream(SOURCE).RTCConfig.__module__.endswith("configuration_rtc")


def test_upstream_snapshot_preserves_model_and_canonical_units(queue):
    seed(queue)
    snapshot = queue.snapshot()
    np.testing.assert_allclose(snapshot.model_actions.numpy(), actions())
    np.testing.assert_allclose(snapshot.canonical_actions.numpy(), actions(0.1))
    snapshot.model_actions[0, 0] = 999
    assert queue.snapshot().model_actions[0, 0] == 0


def test_interleaved_consumer_makes_native_replacement_cas_fail(queue):
    seed(queue)
    snapshot = queue.snapshot()
    thread = Thread(target=queue.get)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not queue.replace_future(actions(0.2), [Source()] * 3, snapshot=snapshot)
    assert queue.snapshot().cursor == 1
    assert queue.qsize() == 2


def test_clear_preempts_pending_native_merge_and_review(queue):
    seed(queue)
    old = queue.snapshot()
    queue.invalidate()
    assert queue.snapshot().generation == old.generation + 1
    assert not queue.merge(actions(), actions(), 0, snapshot=old)
    assert not queue.replace_future(actions(), [Source()] * 3, snapshot=old)
    assert queue.empty()


def test_facade_rejects_null_or_non_native_snapshot_after_preemption(queue):
    queue.invalidate()
    for snapshot in (None, object(), {"generation": queue.snapshot().generation}):
        with pytest.raises(ValueError, match="snapshot"):
            queue.merge(actions(), actions(), 0, snapshot=snapshot)
        with pytest.raises(ValueError, match="snapshot"):
            queue.replace_future(actions(), [Source()] * 3, snapshot=snapshot)
    assert queue.empty()


@pytest.mark.parametrize("delay", [None, float("nan"), float("inf"), -1, 1.5, True])
def test_facade_rejects_invalid_delay_before_native_mutation(queue, delay):
    with pytest.raises(ValueError, match="real_delay"):
        queue.merge(actions(), actions(), delay, snapshot=queue.snapshot())
    assert queue.empty()


def test_native_merge_drops_only_consumed_prefix_and_keeps_task_provenance(queue):
    seed(queue)
    before = queue.snapshot()
    queue.get()
    assert queue.merge(actions(0.2), actions(0.3), 2, snapshot=before,
                       task="recover", provenance=Source(task="recover"))
    assert queue.qsize() == 2  # native cap uses one action actually committed
    action, task, source = queue.get_with_provenance()
    assert task == source.task == "recover"
    np.testing.assert_allclose(action.numpy(), actions(0.3)[1])


def test_native_replacement_preserves_per_action_timestamp_objects(queue):
    sources = [Source(target_at_s=100 + i * 0.1) for i in range(3)]
    assert queue.replace_future(actions(), sources, snapshot=queue.snapshot())
    assert queue.get_with_provenance()[2] is sources[0]


@pytest.mark.parametrize("bad", [np.zeros((3, 6)), np.zeros((7,)), actions(np.nan)])
def test_adapter_rejects_invalid_canonical_inputs(queue, bad):
    with pytest.raises(ValueError):
        queue.merge(actions(), bad, 0, snapshot=queue.snapshot())
    assert queue.empty()


def test_adapter_rejects_misaligned_native_and_robot_horizons(queue):
    with pytest.raises(ValueError, match="horizon"):
        queue.merge(np.zeros((4, 12)), actions(), 0, snapshot=queue.snapshot())


def test_adapter_accepts_different_native_action_dimension(queue):
    assert queue.merge(np.zeros((3, 12)), actions(), 0, snapshot=queue.snapshot())
    assert tuple(queue.snapshot().model_actions.shape) == (3, 12)


def test_adapter_preserves_native_tensor_dtype_and_device(queue):
    import torch
    model = torch.arange(36, dtype=torch.float32).reshape(3, 12)
    assert queue.merge(model, actions(), 0, snapshot=queue.snapshot())
    prefix = queue.get_left_over()
    assert prefix.dtype == model.dtype
    assert prefix.device == model.device
    assert torch.equal(prefix, model)


def test_adapter_accepts_native_bfloat16_without_changing_its_dtype(queue):
    import torch
    model = torch.zeros((3, 12), dtype=torch.bfloat16)
    assert queue.merge(model, actions(), 0, snapshot=queue.snapshot())
    assert queue.get_left_over().dtype == torch.bfloat16


def test_append_mode_keeps_canonical_prefix_guard_until_clear(rtc):
    queue = rtc.ActionQueue(SOURCE, enabled=False)
    assert queue.replace_future(actions(), [Source()] * 3, snapshot=queue.snapshot())
    assert queue.merge(actions(), actions(), 0, snapshot=queue.snapshot())
    with pytest.raises(ValueError, match="model.space"):
        queue.get_left_over()
    queue.clear()
    assert queue.get_left_over() is None


def test_timestamp_validation_reports_expired_prefix_without_redating(rtc):
    assert rtc.validate_proposal_timestamp(100.0, 100.15, 0.1, 4) == 2
    assert rtc.validate_proposal_timestamp(100.0, 100.0, 0.1, 4) == 0
    with pytest.raises(ValueError, match="expired"):
        rtc.validate_proposal_timestamp(100.0, 101.0, 0.1, 4)
    for args in [(float("nan"), 1, 0.1, 4), (1, 1, 0, 4), (1, 1, 0.1, 0)]:
        with pytest.raises(ValueError):
            rtc.validate_proposal_timestamp(*args)


def test_validation_rejects_first_point_velocity_jump_and_joint_bounds(rtc):
    with pytest.raises(ValueError, match="velocity"):
        rtc.validate_trajectory(actions(1), step_s=0.1, previous_action=actions()[0], max_velocity=np.ones(6))
    with pytest.raises(ValueError, match="bounds"):
        rtc.validate_trajectory(actions(1), step_s=0.1, joint_lower=-np.ones(6), joint_upper=np.ones(6) * 0.5)


def test_validation_checks_acceleration_and_discrete_gripper(rtc):
    trajectory = actions()
    trajectory[2, :6] = 1
    with pytest.raises(ValueError, match="acceleration"):
        rtc.validate_trajectory(trajectory, step_s=0.1, max_acceleration=np.ones(6))
    trajectory = actions()
    trajectory[1, -1] = 0.5
    with pytest.raises(ValueError, match="gripper"):
        rtc.validate_trajectory(trajectory, step_s=0.1)
    trajectory[1:, -1] = 1
    np.testing.assert_equal(rtc.validate_trajectory(trajectory, step_s=0.1), trajectory)


def test_policy_adapter_rejects_black_box_and_passes_native_prefix(rtc, queue):
    class BlackBox:
        def predict_action_chunk(self, observation):
            return actions()

    with pytest.raises(ValueError, match="supports_rtc"):
        rtc.RTCPolicyAdapter(BlackBox(), queue)

    class Backend:
        supports_rtc = True

        def predict_action_chunk(self, observation, *, prev_chunk_left_over, inference_delay):
            assert observation == "observation"
            assert inference_delay == 2
            np.testing.assert_equal(prev_chunk_left_over.numpy(), np.zeros((3, 12)))
            return prev_chunk_left_over

    queue.merge(np.zeros((3, 12)), actions(0.3), 0, snapshot=queue.snapshot())
    result = rtc.RTCPolicyAdapter(Backend(), queue).predict("observation", inference_delay=2)
    assert result.snapshot.generation == queue.snapshot().generation
    assert tuple(result.model_actions.shape) == (3, 12)


def test_canonical_only_reviewer_replacement_cannot_become_native_prefix(rtc, queue):
    class Backend:
        supports_rtc = True

        def predict_action_chunk(self, observation, *, prev_chunk_left_over, inference_delay):
            pytest.fail("canonical replacement must be rejected before policy invocation")

    assert queue.replace_future(actions(), [Source()] * 3, snapshot=queue.snapshot())
    with pytest.raises(ValueError, match="model.space"):
        rtc.RTCPolicyAdapter(Backend(), queue).predict({}, inference_delay=1)


def test_verified_loader_rejects_changed_upstream_bytes(rtc, tmp_path):
    from shutil import copytree
    copytree(SOURCE / "src", tmp_path / "src")
    path = tmp_path / "src/lerobot/policies/rtc/action_queue.py"
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="hash"):
        rtc.load_upstream(tmp_path)
