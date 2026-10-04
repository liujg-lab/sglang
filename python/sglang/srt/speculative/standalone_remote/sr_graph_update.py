"""Runner-owned, single-flight graph update thread; no process-global worker."""

from __future__ import annotations

import threading
import weakref
from contextlib import nullcontext

import torch


class SRGraphUpdateWorker:
    def __init__(self, device_id):
        self.device_id = device_id
        self.generation = 0
        self.completed = 0
        self._condition = threading.Condition()
        self._task = None
        self._busy = False
        self._error = None
        self._closing = False
        self._thread = None
        self._poisoned = False
        # CPython runs these callbacks before joining non-daemon threads.
        # Retain the worker until that point, even if the runner was collected.
        threading._register_atexit(self.close)

    def submit(self, update, *, context=None):
        with self._condition:
            if self._closing or self._poisoned or self._busy:
                raise RuntimeError("graph update worker closed, unresolved or busy")
            if self._thread is None:
                thread = threading.Thread(
                    target=self._run, name="SRGraphUpdate", daemon=False
                )
                thread.start()
                self._thread = thread
            self.generation += 1
            self._busy = True
            self._error = None
            self._task = (self.generation, update, context)
            self._condition.notify_all()
            return self.generation

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._task is not None or self._closing
                )
                if self._task is None:
                    return
                generation, update, context = self._task
                self._task = None
            error = None
            try:
                with context if context is not None else nullcontext():
                    update()
            except BaseException as exc:
                error = exc
            finally:
                with self._condition:
                    self.completed = generation
                    self._error = error
                    self._poisoned |= error is not None
                    # _busy stays set until the caller confirms completion.
                    self._condition.notify_all()
                # Do not retain logits, payload or runner closures between tasks.
                del update, context

    def wait(self, generation):
        with self._condition:
            if generation != self.generation or not self._busy:
                raise RuntimeError("graph update generation is stale")
            self._condition.wait_for(lambda: self.completed >= generation)
            error, self._error = self._error, None
            self._busy = False
            self._condition.notify_all()
            return error

    def poison(self):
        with self._condition:
            self._poisoned = True

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def execution_context(self):
        # Resolve the caller's stream before moving to the persistent thread.
        stream = torch.npu.current_stream(self.device_id)
        from contextlib import contextmanager

        @contextmanager
        def bind():
            torch.npu.set_device(self.device_id)
            with torch.npu.stream(stream):
                yield

        return bind()


def runner_update_worker(runner, enabled):
    if not enabled:
        return None
    worker = getattr(runner, "_sr_graph_update_worker", None)
    if worker is None:
        device = getattr(runner, "_npu_graph_device_id", None)
        if device is None:
            raise RuntimeError("SR graph update device has not been prepared")
        worker = runner._sr_graph_update_worker = SRGraphUpdateWorker(device)
        if hasattr(runner, "__weakref__"):
            runner._sr_graph_update_finalizer = weakref.finalize(runner, worker.close)
    elif worker.device_id != runner._npu_graph_device_id:
        raise RuntimeError("SR graph update device changed")
    return worker


def close_runner_update_worker(runner):
    worker = getattr(runner, "_sr_graph_update_worker", None)
    if worker is not None:
        worker.close()
