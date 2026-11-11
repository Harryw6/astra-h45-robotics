"""Go2 per-step displacement policy contract; gateways still consume velocities."""
import numpy as np

from .relative14 import HORIZON, STATE_INDICES, STATE_NAMES, UNITS as STATE_UNITS
from .relative14 import finite_width, state14_from_state30

FPS = 15
DT = 1.0 / FPS
SCHEMA_ID = 'heterovla_v3_state14_go2_step_displacement_v1'
ACTION_NAMES = ('go2.dx_heading', 'go2.dy_heading', 'go2.delta_yaw', *STATE_NAMES[3:])
ACTION_UNITS = ('m', 'm', 'rad', *STATE_UNITS[3:])
# A displacement must never be offset by a measured velocity.
DELTA_MASK = (False,) * 3 + (True,) * 6 + (False,) + (True,) * 4
OUTPUT_ENCODING = 'go2_step_displacements_joint_targets_and_uav_heading_velocities'


def displacement_actions(velocity_actions):
    """Convert original V3 motion proxies to signed displacement over one DT."""
    actions = finite_width(velocity_actions, 14, 'physical velocity actions').copy()
    actions[..., :3] *= DT
    return actions


def chunk_targets(state, velocity_actions):
    state = finite_width(state, 14, 'state14')
    actions = displacement_actions(velocity_actions)
    if actions.shape != state.shape[:-1] + (HORIZON, 14):
        raise ValueError('Expected one state14 for each H15 action chunk')
    return actions - np.expand_dims(np.where(DELTA_MASK, state, 0), -2)


def contract_metadata():
    return dict(schema_id=SCHEMA_ID, physical_state_dim=14, physical_action_dim=14,
                model_action_dim=32, action_horizon=HORIZON, fps=FPS, action_dt_s=DT,
                state_names=list(STATE_NAMES), source_state30_indices=list(STATE_INDICES),
                state_units=list(STATE_UNITS), action_names=list(ACTION_NAMES),
                units=list(ACTION_UNITS), delta_mask=list(DELTA_MASK),
                action_encoding='go2_step_displacement_other_chunk_relative',
                returned_action_encoding=OUTPUT_ENCODING,
                anchor='go2_each_step_start_other_observation_at_chunk_start',
                go2_displacement_frame='heading_at_each_step_start_forward_left',
                gripper_encoding='absolute_opening_m', discrete_state_input=True)


def check_policy_metadata(metadata):
    for key, expected in contract_metadata().items():
        if metadata.get(key) != expected:
            raise ValueError('Displacement14 policy contract mismatch: ' + key)


def policy_observation(observation, metadata):
    check_policy_metadata(metadata)
    state = np.asarray(observation['state'])
    if state.shape == (30,):
        state = state14_from_state30(state)
    elif state.shape == (14,):
        state = finite_width(state, 14, 'state14').copy()
    else:
        raise ValueError('Expected unbatched state30 or state14')
    return {**observation, 'state': state}


def execution_actions(actions, metadata):
    """Decode once at the policy client boundary, before hardware limit checks."""
    check_policy_metadata(metadata)
    result = finite_width(actions, 14, 'displacement actions').astype(np.float64)
    if result.shape != (HORIZON, 14):
        raise ValueError('Expected H15 displacement actions')
    result[:, :3] /= DT
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite decoded velocity')
    return result
