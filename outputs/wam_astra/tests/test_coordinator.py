"""Shadow validation against the actual pinned native queue; no robot calls."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / 'work/wam_demo_library/vendor'))


def observation(identifier='obs-1', at=100.):
    return {'observation_id': identifier, 'observed_at_s': at, 'state': [0.] * 7,
            'state_joint_units': 'rad', 'state_gripper_units': 'm',
            'previous_gripper_normalized': 0., 'images': []}


@pytest.fixture
def coordinator():
    assert importlib.util.find_spec('wam_astra.coordinator') is not None, 'Shadow coordinator must exist'
    from wam_astra.coordinator import CoordinatorConfig, ShadowCoordinator
    from wam_astra.rtc import ActionQueue
    queue = ActionQueue()
    return ShadowCoordinator(queue, config=CoordinatorConfig(
        max_observation_age_s=1., max_request_age_s=1.,
        joint_lower=(-2.,) * 6, joint_upper=(2.,) * 6,
        max_velocity=(1.,) * 6, max_acceleration=(4.,) * 6,
    ), clock=lambda: 100.)


def request(coordinator):
    coordinator.observe(observation())
    return coordinator.make_review_request('Move toward target', demonstrations=[
        {'demo_json': 'library/demo.json', 'input_json': 'native/input.json',
         'app_server_items_json': 'native/app_server_items.json'}
    ], history=[{'observation_id': 'older', 'receipt': {'acknowledged': None}}])


def response(req, **changes):
    result = {
        'schema_version': 1, 'request_id': req['request_id'], 'observation_id': req['observation_id'],
        'queue_generation': req['queue_generation'], 'queue_cursor': req['queue_cursor'],
        'decision': 'edit', 'reason': 'External provider proposal',
        'action_joint_units': 'rad', 'action_gripper_units': 'normalized',
        'plan_observed_at_s': req['observed_at_s'], 'target_start_at_s': 100., 'step_s': .1,
        'actions': [[.01 * i] * 6 + [0.] for i in range(3)],
    }
    result.update(changes)
    return result


def test_valid_review_atomically_replaces_native_future_and_preserves_target_times(coordinator):
    req = request(coordinator)
    assert req['shadow_only'] is True
    assert req['demonstrations'][0]['input_json'] == 'native/input.json'
    assert req['history'][0]['receipt']['acknowledged'] is None
    result = coordinator.apply_review(response(req))
    assert result['status'] == 'accepted'
    assert result['shadow_only'] is True
    assert coordinator.queue.qsize() == 3
    action, task, provenance = coordinator.queue.get_with_provenance()
    assert task == 'Move toward target'
    assert provenance.request_id == req['request_id']
    assert provenance.target_at_s == 100.
    np.testing.assert_array_equal(action.numpy(), [0.] * 7)
    assert coordinator.apply_review(response(req))['status'] == 'rejected'


@pytest.mark.parametrize('changes,reason', [
    ({'queue_generation': 999}, 'generation'),
    ({'queue_cursor': 999}, 'cursor'),
    ({'observation_id': 'foreign'}, 'observation'),
    ({'request_id': 'foreign'}, 'request'),
    ({'schema_version': 2}, 'schema'),
    ({'schema_version': True}, 'schema'),
    ({'plan_observed_at_s': 99.9}, 'origin'),
    ({'action_joint_units': 'degree'}, 'units'),
    ({'action_gripper_units': 'm'}, 'units'),
    ({'actions': [[float('nan')] * 7]}, 'finite'),
    ({'actions': [[0.] * 6]}, 'shape'),
    ({'actions': [[.5] * 6 + [0.]]}, 'velocity'),
    ({'actions': [[0.] * 6 + [.5]]}, 'gripper'),
    ({'actions': [[3.] * 6 + [0.]]}, 'bounds'),
    ({'actions': [[0.] * 7, [.01] * 6 + [0.], [.09] * 6 + [0.]]}, 'acceleration'),
    ({'target_start_at_s': 98.}, 'expired'),
])
def test_invalid_return_cannot_mutate_native_queue(coordinator, changes, reason):
    req = request(coordinator)
    result = coordinator.apply_review(response(req, **changes))
    assert result['status'] == 'rejected'
    assert reason in result['reason'].lower()
    assert coordinator.queue.empty()


def test_stale_request_and_stale_live_observation_are_rejected(coordinator):
    req = request(coordinator)
    result = coordinator.apply_review(response(req), now_s=101.01)
    assert result['status'] == 'rejected'
    assert 'stale' in result['reason']


def test_new_observation_preempts_review_even_when_prior_observation_is_recent(coordinator):
    req = request(coordinator)
    coordinator.observe(observation('obs-2', 100.01), now_s=100.01)
    result = coordinator.apply_review(response(req), now_s=100.02)
    assert result['status'] == 'rejected'
    assert 'observation' in result['reason']
    assert coordinator.queue.empty()


def test_out_of_order_review_is_rejected(coordinator):
    old = request(coordinator)
    new = coordinator.make_review_request('New goal')
    assert coordinator.apply_review(response(old))['status'] == 'rejected'
    assert coordinator.apply_review(response(new))['status'] == 'accepted'


def test_native_generation_change_preempts_review(coordinator):
    req = request(coordinator)
    coordinator.queue.invalidate()
    result = coordinator.apply_review(response(req))
    assert result['status'] == 'rejected'
    assert 'generation' in result['reason']


def test_native_consumer_cursor_change_preempts_review(coordinator):
    from dataclasses import make_dataclass
    source = make_dataclass('Source', ['task'])('old')
    assert coordinator.queue.replace_future([[0.] * 7] * 3, [source] * 3, snapshot=coordinator.queue.snapshot())
    req = request(coordinator)
    coordinator.queue.get()
    result = coordinator.apply_review(response(req))
    assert result['status'] == 'rejected'
    assert 'cursor' in result['reason']
    assert coordinator.queue.qsize() == 2


def test_partial_expiry_is_rejected_without_redating(coordinator):
    req = request(coordinator)
    result = coordinator.apply_review(response(req), now_s=100.05)
    assert result['status'] == 'rejected'
    assert 'expired' in result['reason']
    assert coordinator.queue.empty()


def test_hold_invalidates_future_and_never_enqueues_recorded_targets(coordinator):
    req = request(coordinator)
    reply = {k: response(req)[k] for k in (
        'schema_version', 'request_id', 'observation_id', 'queue_generation', 'queue_cursor', 'decision', 'reason')}
    reply['decision'] = 'hold'
    assert coordinator.apply_review(reply)['status'] == 'hold'
    assert coordinator.queue.empty()
    assert coordinator.queue.snapshot().generation == req['queue_generation'] + 1


def test_retrieval_callback_is_caller_supplied_and_request_is_defensive_copy(coordinator):
    data = [{'input_json': 'existing/native/input.json', 'recorded_joint_target': [999.] * 7}]
    coordinator.retrieval_callback = lambda obs: data
    coordinator.observe(observation())
    req = coordinator.make_review_request('Observe')
    req['demonstrations'][0]['input_json'] = 'changed'
    assert data[0]['input_json'] == 'existing/native/input.json'
    assert coordinator.queue.empty()


def test_native_tracking_event_invalidates_future(coordinator):
    from wam_astra.monitor import MonitorConfig, NativeTrackingMonitor
    coordinator.monitor = NativeTrackingMonitor(config=MonitorConfig(consecutive=1))
    event = coordinator.observe_tracking(observation(),
        {'left_arm_joint_state': [1.] * 6},
        {'left_arm_joint_state': [0.] * 6})
    assert event['triggered'] is True
    assert event['evidence']['model_identity'] == 'RoboICL.TrackingGuard'
    assert coordinator.queue.snapshot().generation == 1


@pytest.mark.parametrize('value', ['0', False])
def test_string_and_boolean_actions_are_not_numeric_json_targets(coordinator, value):
    req = request(coordinator)
    result = coordinator.apply_review(response(req, actions=[[value] * 7]))
    assert result['status'] == 'rejected'
    assert 'number' in result['reason']
    assert coordinator.queue.empty()


def test_unknown_prior_gripper_is_not_filled_from_measured_metres(coordinator):
    obs = observation()
    obs['state'][6] = .035
    obs.pop('previous_gripper_normalized')
    coordinator.observe(obs)
    req = coordinator.make_review_request('Observe')
    assert req['observation']['state'][6] == .035
    result = coordinator.apply_review(response(req))
    assert result['status'] == 'rejected'
    assert 'previous_gripper_normalized' in result['reason']


def test_future_observation_is_rejected_without_becoming_current(coordinator):
    with pytest.raises(ValueError, match='future'):
        coordinator.observe(observation(at=100.01))
    with pytest.raises(ValueError, match='No observation'):
        coordinator.make_review_request('Observe')


def test_foreign_monitor_receipt_cannot_invalidate_current_queue(coordinator):
    coordinator.observe(observation())
    before = coordinator.queue.snapshot().generation
    with pytest.raises(ValueError, match='observation'):
        coordinator.observe(observation(), monitor_event={'observation_id': 'foreign', 'triggered': True})
    assert coordinator.queue.snapshot().generation == before


def test_actual_native_cas_rejects_an_interleaved_invalidation(coordinator, monkeypatch):
    req = request(coordinator)
    original = coordinator.queue.replace_future
    def interleave(actions, provenance, *, snapshot):
        coordinator.queue.invalidate()
        return original(actions, provenance, snapshot=snapshot)
    monkeypatch.setattr(coordinator.queue, 'replace_future', interleave)
    result = coordinator.apply_review(response(req))
    assert result['status'] == 'rejected'
    assert 'atomic' in result['reason']
    assert coordinator.queue.empty()


def test_request_age_limit_is_independent_of_observation_age_limit(coordinator):
    from dataclasses import replace
    coordinator.config = replace(coordinator.config, max_request_age_s=.1)
    req = request(coordinator)
    result = coordinator.apply_review(response(req, target_start_at_s=100.2), now_s=100.2)
    assert result['status'] == 'rejected'
    assert result['reason'] == 'Review request is stale'


def test_unconfigured_physical_limits_export_context_but_reject_proposals(coordinator):
    from wam_astra.coordinator import CoordinatorConfig
    coordinator.config = CoordinatorConfig()
    req = request(coordinator)
    result = coordinator.apply_review(response(req))
    assert result['status'] == 'rejected'
    assert 'limits' in result['reason']
