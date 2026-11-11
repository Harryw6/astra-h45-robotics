"""Thin, shadow-only boundary around a pinned, unmodified LeRobot RTC queue.

Queue storage, cursor/generation arbitration and delay trimming execute upstream
code. This module supplies import isolation, input checks and a policy contract;
it does not implement interpolation, a trajectory planner or a guided sampler.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import inspect
import math
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

UPSTREAM_COMMIT = "d490e472a5d1a207cfe0255caaaf74b6d66e4656"
UPSTREAM_REPOSITORY = "https://github.com/huggingface/lerobot"
UPSTREAM_HASHES = {
    "src/lerobot/configs/types.py": "1354e7d24b19d5d4970064104f76d8c85570c204bf2d09e2b4402f93a994ad99",
    "src/lerobot/policies/rtc/configuration_rtc.py": "21398dee6197714b7b1dafea66e32da05ce1c02ef1487999af65b4522a3cf223",
    "src/lerobot/policies/rtc/action_queue.py": "9cf6207c92fb71efd4753003f330b7c72cca7c6f1d561180420584b53add36d0",
}
DEFAULT_UPSTREAM_ROOT = Path(__file__).resolve().parents[4] / "work/upstream/lerobot"
_IMPORT_LOCK = RLock()
_UPSTREAM_CACHE: dict[str, SimpleNamespace] = {}


def load_upstream(source_root: str | Path = DEFAULT_UPSTREAM_ROOT) -> SimpleNamespace:
    """Verify pinned source bytes and execute only native types/config/queue.

    LeRobot package initializers also import unrelated dataset, policy and
    training dependencies. The private package namespace avoids those imports.
    The temporary ``lerobot.configs`` bridge exports the *native* enum required
    by native configuration_rtc.py, then restores any existing module binding.
    No upstream source is edited and no Torch substitute is installed.
    """
    root = Path(source_root).resolve()
    for relative, expected in UPSTREAM_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned LeRobot source: {path}")
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"LeRobot source hash mismatch: {path}")
    with _IMPORT_LOCK:
        key = str(root)
        if key in _UPSTREAM_CACHE:
            return _UPSTREAM_CACHE[key]
        namespace = "_wam_lerobot_" + sha256(key.encode()).hexdigest()[:12]
        paths = {"": "src/lerobot", ".policies": "src/lerobot/policies",
                 ".policies.rtc": "src/lerobot/policies/rtc"}
        for suffix, relative in paths.items():
            package = ModuleType(namespace + suffix)
            package.__path__ = [str(root / relative)]
            sys.modules[package.__name__] = package

        def execute(name: str, relative: str):
            spec = importlib.util.spec_from_file_location(name, root / relative)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {relative}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            return module

        native_types = execute(namespace + ".types", "src/lerobot/configs/types.py")
        bridge = ModuleType("lerobot.configs")
        bridge.__file__ = native_types.__file__
        bridge.RTCAttentionSchedule = native_types.RTCAttentionSchedule
        previous_configs = sys.modules.get("lerobot.configs")
        sys.modules["lerobot.configs"] = bridge
        try:
            config = execute(namespace + ".policies.rtc.configuration_rtc",
                             "src/lerobot/policies/rtc/configuration_rtc.py")
        finally:
            if previous_configs is None:
                sys.modules.pop("lerobot.configs", None)
            else:
                sys.modules["lerobot.configs"] = previous_configs
        native_queue = execute(namespace + ".policies.rtc.action_queue",
                               "src/lerobot/policies/rtc/action_queue.py")
        bundle = SimpleNamespace(ActionQueue=native_queue.ActionQueue,
                                 QueueSnapshot=native_queue.QueueSnapshot,
                                 RTCConfig=config.RTCConfig,
                                 RTCAttentionSchedule=native_types.RTCAttentionSchedule,
                                 source_root=root, commit=UPSTREAM_COMMIT)
        _UPSTREAM_CACHE[key] = bundle
        return bundle


def _array(value: Any, *, canonical: bool) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().double().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Actions must have shape H x D with a positive horizon")
    if canonical and array.shape[1] != 7:
        raise ValueError("Canonical Piper actions must have shape H x 7")
    if not np.isfinite(array).all():
        raise ValueError("Actions must be finite")
    return array


def validate_proposal_timestamp(origin_at_s: float, now_s: float, step_s: float, horizon: int) -> int:
    """Return count of targets strictly before now; reject an expired chunk.

    ``origin_at_s`` is the original first target time, never an arrival time.
    The caller decides whether to reject or explicitly drop the reported prefix;
    this validator never changes target times or mutates the queue.
    """
    if not all(math.isfinite(x) for x in (origin_at_s, now_s, step_s)) or step_s <= 0:
        raise ValueError("Times must be finite and step_s must be positive")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    targets = origin_at_s + np.arange(horizon, dtype=np.float64) * step_s
    if not np.isfinite(targets).all() or (horizon > 1 and not np.all(np.diff(targets) > 0)):
        raise ValueError("Target timestamps must be finite and strictly increasing")
    expired = int(np.count_nonzero(targets < now_s))
    if expired == horizon:
        raise ValueError("Entire proposal is expired")
    return expired


def validate_trajectory(actions: Any, *, step_s: float, joint_lower=None, joint_upper=None,
                        max_velocity=None, max_acceleration=None, previous_action=None,
                        previous_velocity=None, gripper_values=(0.0, 1.0)) -> np.ndarray:
    """Reject invalid canonical targets without clipping or planning.

    First six columns are joint positions; column seven is a discrete gripper
    command. Limits are supplied by the caller in the same joint units. The
    finite-difference checks are diagnostics, not continuous-time guarantees.
    Supplying previous_action checks the first point across the chunk boundary.
    """
    array = _array(actions, canonical=True)
    if not math.isfinite(step_s) or step_s <= 0:
        raise ValueError("step_s must be finite and positive")

    def six(value, name, *, positive=False):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (6,) or not np.isfinite(vector).all() or (positive and np.any(vector <= 0)):
            raise ValueError(f"{name} must contain six finite{' positive' if positive else ''} values")
        return vector

    if (joint_lower is None) != (joint_upper is None):
        raise ValueError("Both joint bounds are required")
    if joint_lower is not None:
        lower, upper = six(joint_lower, "joint_lower"), six(joint_upper, "joint_upper")
        if np.any(lower >= upper) or np.any(array[:, :6] < lower) or np.any(array[:, :6] > upper):
            raise ValueError("Joint bounds violated")
    if gripper_values is not None:
        values = np.asarray(gripper_values, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Discrete gripper values must be finite and nonempty")
        if not np.isin(array[:, 6], values).all():
            raise ValueError("Discrete gripper command is invalid")
    joint_positions = array[:, :6]
    if previous_action is not None:
        previous = np.asarray(previous_action, dtype=np.float64)
        if previous.shape != (7,) or not np.isfinite(previous).all():
            raise ValueError("previous_action must contain seven finite values")
        joint_positions = np.vstack((previous[:6], joint_positions))
    velocities = np.diff(joint_positions, axis=0) / step_s
    if max_velocity is not None and np.any(np.abs(velocities) > six(max_velocity, "max_velocity", positive=True)):
        raise ValueError("Joint velocity limit violated")
    if previous_velocity is not None:
        if previous_action is None:
            raise ValueError("previous_velocity requires previous_action")
        velocities = np.vstack((six(previous_velocity, "previous_velocity"), velocities))
    accelerations = np.diff(velocities, axis=0) / step_s
    if max_acceleration is not None and np.any(
            np.abs(accelerations) > six(max_acceleration, "max_acceleration", positive=True)):
        raise ValueError("Joint acceleration limit violated")
    return array.copy()


class ActionQueue:
    """Native queue facade; no replacement queue implementation lives here.

    All mutation must use this facade if RTCPolicyAdapter is used, because the
    facade marks canonical-only review playback as lacking a model-space prefix.
    ``native`` is exposed for inspection, not for bypassing that contract.
    """

    def __init__(self, source_root: str | Path = DEFAULT_UPSTREAM_ROOT, *, enabled=True, cfg=None):
        upstream = load_upstream(source_root)
        self.native = upstream.ActionQueue(cfg if cfg is not None else upstream.RTCConfig(enabled=enabled))
        self._snapshot_type = upstream.QueueSnapshot
        self._boundary_lock = RLock()
        self._canonical_only = False

    def snapshot(self):
        with self._boundary_lock:
            return self.native.snapshot()

    def get(self):
        with self._boundary_lock:
            return self.native.get()

    def get_with_provenance(self):
        with self._boundary_lock:
            return self.native.get_with_provenance()

    def get_with_task(self):
        with self._boundary_lock:
            return self.native.get_with_task()

    def clear(self):
        with self._boundary_lock:
            self.native.clear()
            self._canonical_only = False

    invalidate = clear

    def qsize(self):
        return self.native.qsize()

    def empty(self):
        return self.native.empty()

    def get_left_over(self):
        with self._boundary_lock:
            if self._canonical_only and not self.native.empty():
                raise ValueError("Canonical-only review playback has no native model-space prefix")
            return self.native.get_left_over()

    def get_processed_left_over(self):
        return self.native.get_processed_left_over()

    def get_action_index(self):
        return self.native.get_action_index()

    def merge(self, original_actions, processed_actions, real_delay, action_index_before_inference=None,
              *, snapshot, task=None, provenance=None):
        import torch
        if not isinstance(snapshot, self._snapshot_type):
            raise ValueError("snapshot must be a non-null native QueueSnapshot from this adapter's upstream loader")
        original = _array(original_actions, canonical=False)
        processed = _array(processed_actions, canonical=True)
        if len(original) != len(processed):
            raise ValueError("Model and canonical actions must have the same horizon")
        if isinstance(real_delay, bool) or not isinstance(real_delay, (int, np.integer)) or real_delay < 0:
            raise ValueError("real_delay must be a nonnegative integer")
        with self._boundary_lock:
            # Keep a policy tensor's native dtype/device as well as its units.
            # Validation uses CPU views; the tensors passed to upstream do not.
            model_tensor = (original_actions.detach().clone() if isinstance(original_actions, torch.Tensor)
                            else torch.as_tensor(np.asarray(original_actions)).clone())
            canonical_tensor = (processed_actions.detach().clone() if isinstance(processed_actions, torch.Tensor)
                                else torch.as_tensor(np.asarray(processed_actions)).clone())
            accepted = self.native.merge(model_tensor, canonical_tensor,
                                         int(real_delay), action_index_before_inference,
                                         snapshot=snapshot, task=task, provenance=provenance)
            if accepted and self.native.cfg.enabled:
                self._canonical_only = False
            return accepted

    def replace_future(self, actions, provenance, *, snapshot):
        import torch
        if not isinstance(snapshot, self._snapshot_type):
            raise ValueError("snapshot must be a non-null native QueueSnapshot from this adapter's upstream loader")
        canonical = _array(actions, canonical=True)
        sources = list(provenance)
        if len(sources) != len(canonical) or any(not hasattr(source, "task") for source in sources):
            raise ValueError("Every replacement action requires provenance with a task attribute")
        with self._boundary_lock:
            accepted = self.native.replace_future(torch.as_tensor(canonical).clone(), sources, snapshot=snapshot)
            if accepted:
                self._canonical_only = True
            return accepted

    def _prediction_snapshot(self):
        with self._boundary_lock:
            self.get_left_over()  # explicit guard for canonical-only review playback
            return self.native.snapshot()


@dataclass(frozen=True)
class PolicyPrediction:
    snapshot: Any
    model_actions: Any


class RTCPolicyAdapter:
    """Pass native prefixes to an explicitly RTC-capable policy backend.

    Capability declaration is an integration contract, not proof that a backend
    implements guided diffusion correctly. No policy/model is loaded here.
    """

    def __init__(self, backend, queue: ActionQueue):
        method = getattr(backend, "predict_action_chunk", None)
        if getattr(backend, "supports_rtc", False) is not True or not callable(method):
            raise ValueError("Backend must explicitly declare supports_rtc=True")
        parameters = inspect.signature(method).parameters
        if not {"prev_chunk_left_over", "inference_delay"}.issubset(parameters):
            raise ValueError("Backend must explicitly accept prev_chunk_left_over and inference_delay")
        self.backend, self.queue = backend, queue

    def predict(self, observation, *, inference_delay: int) -> PolicyPrediction:
        if isinstance(inference_delay, bool) or not isinstance(inference_delay, (int, np.integer)) or inference_delay < 0:
            raise ValueError("inference_delay must be a nonnegative integer number of steps")
        snapshot = self.queue._prediction_snapshot()
        prefix = None if snapshot.model_actions is None else snapshot.model_actions.clone()
        result = self.backend.predict_action_chunk(observation, prev_chunk_left_over=prefix,
                                                   inference_delay=int(inference_delay))
        _array(result, canonical=False)
        return PolicyPrediction(snapshot=snapshot, model_actions=result)
