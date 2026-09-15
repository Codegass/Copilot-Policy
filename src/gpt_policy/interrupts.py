"""Count operator interrupts and defer them while joining command workers."""

from contextlib import contextmanager
import signal
import threading


class HomeInterrupted(KeyboardInterrupt):
    """An interrupt during an already running home must not restart it."""


class Interrupts:
    def __init__(self):
        self.count = 0
        self.ignoring = False
        self._deferred = 0
        self._installed = False

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self._previous = signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        return self

    def __exit__(self, *_):
        if self._installed:
            signal.signal(signal.SIGINT, self._previous)

    def _handle(self, *_):
        self.count += 1
        if not self.ignoring and not self._deferred:
            raise KeyboardInterrupt

    @contextmanager
    def defer(self):
        before = self.count
        self._deferred += 1
        try:
            yield
        finally:
            self._deferred -= 1
            if self.count > before and not self.ignoring and not self._deferred:
                raise KeyboardInterrupt


@contextmanager
def defer_interrupts():
    handler = getattr(signal.getsignal(signal.SIGINT), "__self__", None)
    if isinstance(handler, Interrupts):
        with handler.defer():
            yield
    else:
        yield
