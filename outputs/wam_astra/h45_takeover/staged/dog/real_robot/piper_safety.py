"""Piper SDK adapter and watchdog. Importing this module never opens hardware."""
import logging
import math
import threading
import time


FEEDBACK_IDS = (0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8)
ENABLE_IDS = tuple(range(0x261, 0x267))


def monitored_piper(base):
    """Track each feedback frame; the SDK aggregate timestamp masks missing pairs."""
    class MonitoredPiper(base):
        def __init__(self, *args, **kwargs):
            self.feedback_received = {}
            self.feedback_lock = threading.Lock()
            super().__init__(*args, **kwargs)

        def ParseCANFrame(self, frame):
            result = super().ParseCANFrame(frame)
            if (frame is not None and not frame.is_error_frame
                    and not frame.is_remote_frame and not frame.is_extended_id
                    and len(frame.data) == 8
                    and frame.arbitration_id in FEEDBACK_IDS + ENABLE_IDS):
                with self.feedback_lock:
                    self.feedback_received[frame.arbitration_id] = time.monotonic()
            return result

        def feedback_fresh(self, ids, timeout):
            with self.feedback_lock:
                # Sample the clock with the timestamps locked: a frame arriving
                # during lock acquisition must not appear to come from the future.
                now = time.monotonic()
                return all(key in self.feedback_received and
                           0 <= now - self.feedback_received[key] <= timeout for key in ids)
    return MonitoredPiper


class SdkErrors(logging.Handler):
    """Installed SDK returns None even when CAN sends fail; observe its error log."""
    def __init__(self):
        super().__init__(logging.ERROR)
        self.generation = 0
        self.message = None

    def emit(self, record):
        # No robot calls here: a stop command may itself log a send failure.
        self.message = record.getMessage()
        self.generation += 1


class SafetyGuard:
    feedback_timeout = .25
    loop_timeout = .25

    def __init__(self, interface, clock=time.monotonic):
        self.interface = interface
        self.clock = clock
        self.lock = threading.RLock()
        self.errors = SdkErrors()
        logger = getattr(interface, 'logger', None)
        self.logger = getattr(logger, 'logger', logger)
        if not isinstance(self.logger, logging.Logger):
            raise RuntimeError('Piper SDK logger unavailable; cannot monitor CAN send failures')
        if not callable(getattr(interface, 'feedback_fresh', None)):
            raise RuntimeError('Piper feedback monitor is required')
        self.logger.addHandler(self.errors)
        self.seen_error = 0
        self.active = False
        self.inhibited = True
        self.fault = '松开所有摇杆、方向键和扳机，按 Home 确认遥操'
        self.last_tick = clock()
        self.stop_attempts = 0
        self.stop_requested = False
        self.stop_confirmed = False
        self.stop_error = None
        self.last_stop = -math.inf
        self.closed = threading.Event()
        self.shutdown_requested = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._watch, name='piper-safety', daemon=True)
        self.thread.start()

    def _watch(self):
        while not self.closed.wait(.02):
            self.poll()

    def call(self, method, *args, allow_false=False, **kwargs):
        """Caller holds lock for a command group. None is not hardware acknowledgement."""
        if method != 'MotionCtrl_1' and self.shutdown_requested.is_set():
            raise RuntimeError('遥操正在退出')
        if method != 'MotionCtrl_1' and self.errors.generation != self.seen_error:
            raise RuntimeError(self.errors.message or 'Unacknowledged SDK error')
        before = self.errors.generation
        result = getattr(self.interface, method)(*args, **kwargs)
        if self.errors.generation != before:
            raise RuntimeError(self.errors.message or 'Piper SDK error')
        if result is False and not allow_false:
            raise RuntimeError(method + ' returned failure')
        return result

    def trip(self, reason):
        with self.lock:
            if not self.inhibited or not self.stop_requested:
                self.fault = str(reason)
            self.inhibited = True
            if self.active:
                self._stop()

    def _stop(self):
        now = self.clock()
        if self.stop_attempts >= 3 or now - self.last_stop < .05:
            return
        self.stop_requested = True
        self.stop_attempts += 1
        self.last_stop = now
        try:
            self.call('MotionCtrl_1', 0x01, 0, 0)
        except Exception as exc:
            self.stop_error = str(exc)

    def poll(self):
        with self.lock:
            if not self.active:
                return
            try:
                if self.shutdown_requested.is_set():
                    self.trip('遥操程序退出，已请求停止')
                if self.errors.generation != self.seen_error:
                    self.seen_error = self.errors.generation
                    self.trip('CAN/SDK 异常：' + str(self.errors.message))
                if self.clock() - self.last_tick > self.loop_timeout:
                    self.trip('遥操循环超时，已请求停止')
                if not self.interface.feedback_fresh(FEEDBACK_IDS + ENABLE_IDS, self.feedback_timeout):
                    self.trip('CAN 反馈超时，已请求停止')
                if self.inhibited and self.stop_requested:
                    self._stop()
                    status = self.interface.GetArmStatus().arm_status
                    self.stop_confirmed = bool(
                        self.interface.feedback_fresh((0x2A1,), self.feedback_timeout)
                        and status.arm_status == 1)
            except Exception as exc:
                self.trip('安全监控异常：' + str(exc))

    def acknowledge(self):
        """Explicit neutral-input acknowledgement; never reset a hardware E-stop."""
        self.seen_error = self.errors.generation
        self.inhibited = False
        self.fault = None
        self.stop_requested = self.stop_confirmed = False
        self.stop_error = None
        self.stop_attempts = 0
        self.last_stop = -math.inf
        self.last_tick = self.clock()

    def close(self):
        self.trip('遥操程序退出，已请求停止')
        # Bounded retries also cover an exception during disk writes or input polling.
        if self.active:
            for _ in range(2):
                self.closed.wait(.06)
                self.poll()
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=.5)
        self.logger.removeHandler(self.errors)
