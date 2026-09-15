"""Serialize command submission against cancellation of an arm's motion."""

import threading


class MotionFault(RuntimeError):
    """A feedback fault latches the arm off for the rest of this process."""

    def __init__(self, details):
        self.details = details
        super().__init__(f"Robot motion fault: {details}")


class MotionControl:
    def __init__(self):
        self.stopped = threading.Event()
        self.lock = threading.RLock()
        self.fault = None

    def check(self):
        if self.fault is not None:
            raise MotionFault(self.fault)
        if self.stopped.is_set():
            raise InterruptedError("Robot motion cancelled")

    def send(self, function, *args):
        with self.lock:
            self.check()
            return function(*args)

    def cancel(self, hold):
        # A worker already sending may finish that command, but cannot submit
        # another target after the measured hold below has replaced it.
        self.stopped.set()
        with self.lock:
            hold()

    def resume(self):
        # Called only after the previous command workers have been joined.
        with self.lock:
            if self.fault is not None:
                raise MotionFault(self.fault)
            self.stopped.clear()
