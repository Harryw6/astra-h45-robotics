"""Thin adapters to pinned, unchanged upstream event guards.

TrackingGuard is a proprioceptive tracking guard, not visual task verification.
NativeVisionMonitor requires externally computed predicted/observed features;
it does not load or manufacture a trained latent dynamics model.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys


ROBOICL_COMMIT = "629290545af0d139a955a6555dedd651a60f3db9"
ROBOICL_SHA256 = "0dead10a31340b114a8ad449157934777482d5efcfd86ce5398efb8c41dae474"
VLA_COMMIT = "927a06da8f00d7df483459e9039219e076c846de"
VLA_SHA256 = "0a88c882fe8f9f6977cbe1957d6d988b1f2b91927a53086d864e2bbf7423b0fe"


def _dependency_root(name: str) -> Path:
    return Path(__file__).resolve().parents[4] / "work" / "upstream" / name


def _load_pinned(root, relative_path, expected_hash, name):
    path = Path(root).resolve() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"{name} dependency missing: {path}; provide the pinned upstream_dir")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"{name} source hash mismatch; expected unchanged pinned source at {path}")
    module_name = "_wam_native_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return sys.modules[module_name], path


@dataclass(frozen=True)
class MonitorConfig:
    """Parameters forwarded verbatim to RoboICL's original TrackingGuard.

    Position is metres; orientation and joints are radians. ``consecutive``
    counts calls containing valid measurements, not elapsed seconds.
    """
    position_m: float = 0.05
    orientation_rad: float = 0.5
    joint_rad: float = 0.5
    consecutive: int = 2

    def __post_init__(self):
        try:
            valid = all(math.isfinite(value) and value > 0
                        for value in (self.position_m, self.orientation_rad, self.joint_rad))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("tracking thresholds must be positive and finite")
        if isinstance(self.consecutive, bool) or not isinstance(self.consecutive, int) or self.consecutive < 1:
            raise ValueError("consecutive must be a positive integer")


class NativeTrackingMonitor:
    """Execute original RoboICL tracking_receipt and TrackingGuard.observe.

    Caller must align command targets with measured state, with radians/metres
    and native left/right keys. This adapter does not infer observation latency,
    hardware acknowledgements, grasp success or task progress.
    """

    def __init__(self, upstream_dir=None, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self._native, self._source = _load_pinned(
            upstream_dir if upstream_dir is not None else _dependency_root("RoboICL"),
            "roboicl/policy/action_feedback.py", ROBOICL_SHA256, "RoboICL",
        )
        self.reset()

    def reset(self):
        """Create a fresh original guard; no custom counter or state machine."""
        self.guard = self._native.TrackingGuard(**asdict(self.config))

    def _evidence(self):
        return {
            "model_identity": "RoboICL.TrackingGuard",
            "capability": "proprioceptive_tracking_only",
            "commit": ROBOICL_COMMIT,
            "source_sha256": ROBOICL_SHA256,
            "source_path": str(self._source),
            "source_url": f"https://github.com/Mosi-AI/RoboICL/blob/{ROBOICL_COMMIT}/roboicl/policy/action_feedback.py",
            "thresholds": dict(self.guard.limits),
            "trained_model": False,
        }

    def update(self, target: Mapping, observed: Mapping, *, observation_id: str = "") -> dict:
        """Return an auditable native receipt and guard result.

        Invalid/missing arm measurements return unavailable and interrupt the
        consecutive run by recreating the original guard. They are never
        interpreted as healthy tracking. observation_id is metadata only.
        """
        receipt = {}
        try:
            if not isinstance(target, Mapping) or not isinstance(observed, Mapping):
                raise ValueError("target_and_observed_must_be_mappings")
            receipt = self._native.tracking_receipt(target, observed)
            if not receipt:
                raise ValueError("no_comparable_arm_measurements")
            if not all(math.isfinite(value) for arm in receipt.values() for value in arm.values()):
                raise ValueError("nonfinite_tracking_receipt")
        except (ValueError, TypeError, OverflowError) as exc:
            self.reset()
            return {
                "triggered": False, "status": "unavailable", "reason": str(exc),
                "receipt": {}, "count": 0, "required": self.guard.required,
                "observation_id": observation_id, "evidence": self._evidence(),
            }
        triggered = bool(self.guard.observe(receipt))
        return {
            "triggered": triggered,
            "status": "triggered" if triggered else (
                "tracking_error_pending" if self.guard.count else "within_tracking_limits"),
            "reason": "native_tracking_guard",
            "receipt": receipt, "count": self.guard.count, "required": self.guard.required,
            "observation_id": observation_id, "evidence": self._evidence(),
        }


EventMonitor = NativeTrackingMonitor


class NativeVisionMonitor:
    """Original VLA-Corrector CircuitBreaker over externally supplied features.

    No LVM checkpoint or encoder is bundled. model_identity records caller
    provenance, not verification that a particular trained model was executed.
    PyTorch is required only when this optional adapter is instantiated.
    """

    def __init__(self, upstream_dir=None, *, model_identity: str,
                 k_mad: float = 3.0, window_size: int = 100, cooldown_steps: int = 5):
        if not isinstance(model_identity, str) or not model_identity.strip():
            raise ValueError("model_identity must identify the supplied feature prediction model")
        if (isinstance(k_mad, bool) or not isinstance(k_mad, (int, float))
                or not math.isfinite(k_mad) or k_mad <= 0):
            raise ValueError("k_mad must be finite and positive")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 10:
            raise ValueError("window_size must be an integer at least 10 for native warmup")
        if isinstance(cooldown_steps, bool) or not isinstance(cooldown_steps, int) or cooldown_steps < 0:
            raise ValueError("cooldown_steps must be a nonnegative integer")
        self.model_identity = model_identity
        self._native, self._source = _load_pinned(
            upstream_dir if upstream_dir is not None else _dependency_root("vla-corrector"),
            "src/siglip_dynamics/inference/circuit_breaker.py", VLA_SHA256, "VLA-Corrector",
        )
        self.breaker = self._native.CircuitBreaker(
            k_mad=k_mad, window_size=window_size, cooldown_steps=cooldown_steps)

    def reset(self):
        self.breaker.reset()

    def update(self, predicted_delta, observed_delta, *, observation_id: str = "") -> dict:
        """Call native compute_error then check; missing/invalid features raise.

        No added persistence, zero-vector policy, cooldown or score calibration.
        The native warmup's infinite threshold is encoded as None for JSON.
        """
        if predicted_delta is None or observed_delta is None:
            raise ValueError("predicted and observed features are required")
        torch = self._native.torch
        try:
            predicted = torch.as_tensor(predicted_delta)
            observed = torch.as_tensor(observed_delta)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise ValueError("features must be numeric tensors") from exc
        if predicted.numel() == 0 or predicted.shape != observed.shape:
            raise ValueError("features must have equal, nonempty shape")
        if not (torch.isfinite(predicted).all() and torch.isfinite(observed).all()):
            raise ValueError("features must be finite")
        if not predicted.is_floating_point():
            predicted = predicted.float()
        if not observed.is_floating_point():
            observed = observed.float()
        score = float(self.breaker.compute_error(predicted, observed))
        if not math.isfinite(score):
            raise ValueError("native feature error must be finite")
        triggered = bool(self.breaker.check(score))
        threshold = float(self.breaker.get_dynamic_threshold())
        warmup = not math.isfinite(threshold)
        return {
            "triggered": triggered, "score": score,
            "threshold": None if warmup else threshold,
            "status": self.breaker.state.value,
            "reason": "native_threshold_warmup" if warmup else "native_circuit_breaker",
            "observation_id": observation_id,
            "evidence": {
                "model_identity": "VLA-Corrector.CircuitBreaker",
                "prediction_model_identity": self.model_identity,
                "prediction_identity_verified": False,
                "capability": "supplied_latent_feature_mismatch_only",
                "commit": VLA_COMMIT, "source_sha256": VLA_SHA256,
                "source_path": str(self._source),
                "source_url": f"https://github.com/ZJU-OmniAI/vla-corrector/blob/{VLA_COMMIT}/src/siglip_dynamics/inference/circuit_breaker.py",
                "checkpoint_loaded_by_adapter": False,
            },
        }
