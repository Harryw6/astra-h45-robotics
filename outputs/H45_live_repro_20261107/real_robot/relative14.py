"""Shared, hardware-free contract for the V3 chunk-relative state14 policy."""
import numpy as np

SCHEMA_ID = 'heterovla_v3_state14_chunk_relative_joint_velocity_v1'
HORIZON = 15
STATE_INDICES = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27)
STATE_NAMES = (
    'go2.vx_heading', 'go2.vy_heading', 'go2.yaw_rate',
    'piper.q1', 'piper.q2', 'piper.q3', 'piper.q4', 'piper.q5', 'piper.q6', 'piper.gripper',
    'uav.vx_heading', 'uav.vy_heading', 'uav.vz', 'uav.yaw_rate',
)
UNITS = ('m/s', 'm/s', 'rad/s', *(['rad'] * 6), 'm', 'm/s', 'm/s', 'm/s', 'rad/s')
DELTA_MASK = (True,) * 9 + (False,) + (True,) * 4
OUTPUT_ENCODING = 'physical_joint_targets_and_heading_velocities'


def finite_width(value, width, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != width or not np.isfinite(array).all():
        raise ValueError('%s must have finite trailing width %s; got %s' % (name, width, array.shape))
    return array


def state14_from_state30(state):
    """Project measured velocity/joint feedback; never fill or reorder implicitly."""
    return finite_width(state, 30, 'state30')[..., list(STATE_INDICES)].copy()


def chunk_delta(state, actions):
    """Numerical equivalent of OpenPI DeltaActions, with a single chunk anchor."""
    state = finite_width(state, 14, 'state14')
    actions = finite_width(actions, 14, 'actions')
    if actions.shape != state.shape[:-1] + (HORIZON, 14):
        raise ValueError('Expected one state14 for each H15 action chunk')
    return actions.copy() - np.expand_dims(np.where(DELTA_MASK, state, 0), -2)


def contract_metadata():
    return dict(schema_id=SCHEMA_ID, physical_state_dim=14, physical_action_dim=14,
                action_horizon=HORIZON, fps=15, state_names=list(STATE_NAMES),
                source_state30_indices=list(STATE_INDICES), units=list(UNITS),
                delta_mask=list(DELTA_MASK), action_encoding='chunk_relative_joint_velocity',
                returned_action_encoding=OUTPUT_ENCODING, anchor='observation_at_chunk_start',
                gripper_encoding='absolute_opening_m', discrete_state_input=True,
                model_action_dim=32,
                velocity_frame='heading components; subtract/add the same start vector numerically, not world-frame acceleration')


def check_policy_metadata(metadata):
    required = contract_metadata()
    for key in ('schema_id', 'physical_state_dim', 'physical_action_dim', 'action_horizon',
                'fps', 'state_names', 'delta_mask', 'action_encoding',
                'returned_action_encoding', 'anchor', 'gripper_encoding', 'discrete_state_input'):
        if metadata.get(key) != required[key]:
            raise ValueError('Relative14 policy contract mismatch: '+key)


def policy_observation(observation, metadata):
    """Client-side projection. The new server itself requires exactly 14 values."""
    width = metadata.get('physical_state_dim', 30)
    if width == 30:
        return observation
    if width != 14:
        raise ValueError('Unsupported policy state dimension: '+str(width))
    check_policy_metadata(metadata)
    state = np.asarray(observation['state'])
    if state.shape == (30,):
        state = state14_from_state30(state)
    elif state.shape == (14,):
        state = finite_width(state, 14, 'state14').copy()
    else:
        raise ValueError('Expected unbatched state30 or state14 at the policy client')
    return {**observation, 'state': state}
