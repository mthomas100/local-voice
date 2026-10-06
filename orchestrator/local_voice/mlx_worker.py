"""One thread, with its own MLX stream, for every MLX call in the process; and the GPU rule enforced at its door.

Why one thread (09c §9, 04c §6): MLX streams belong to the thread that made them. An array created on one thread
and evaluated on another fails ("There is no Stream(gpu, 0) in current thread"), and `asyncio.to_thread` hops
between pool threads. mlx-audio's own MLXWorkScheduler is the pattern copied here: a one-worker executor that
creates a GPU stream on first use and runs every call inside `with mx.stream(...)`. Model loads, warm-ups, STT
and TTS all go through it, so STT can wait behind one TTS chunk; keep each call to one chunk.

Why the guard lives here (a rule of this Mac since 2026-10-04): no speech inference, model load or warm-up
may run while the hold gate is draining or held. Every guarded call asks the guard on the MLX thread, right before
it runs (a call can sit in the queue while a hold begins), and raises GpuHeldError instead of touching the GPU.
Cleanup calls (closing a generator, resetting decoder state, clearing the cache) only release memory and are
submitted unguarded.

`use_mlx=False` (tests with fake engines) never imports mlx.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")


class GpuHeldError(RuntimeError):
    """The hold gate is not open: GPU work refused. The message says why (the gate's own reason)."""


class MLXWorker:
    def __init__(self, *, guard: Callable[[], str | None] | None = None, use_mlx: bool = True, name: str = "mlx"):
        """guard() returns None when GPU work may run, or the reason it may not."""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._guard = guard
        self._use_mlx = use_mlx
        self._stream = None
        self._thread_id: int | None = None
        self.calls = 0
        self.refused = 0

    def set_guard(self, guard: Callable[[], str | None] | None) -> None:
        self._guard = guard

    def _enter(self):
        if self._thread_id is None:
            self._thread_id = threading.get_ident()
        if self._use_mlx and self._stream is None:
            import mlx.core as mx

            self._stream = mx.new_stream(mx.gpu)
            mx.set_default_stream(self._stream)

    def _call(self, fn: Callable[..., T], guarded: bool) -> T:
        self._enter()
        if guarded and self._guard is not None:
            reason = self._guard()
            if reason:
                self.refused += 1
                raise GpuHeldError(reason)
        self.calls += 1
        if self._use_mlx:
            import mlx.core as mx

            with mx.stream(self._stream):
                return fn()
        return fn()

    async def run(self, fn: Callable[..., T], /, *args: Any, guarded: bool = True, **kwargs: Any) -> T:
        """Run a blocking call on the MLX thread and await it. Raises GpuHeldError when the guard refuses."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._call, partial(fn, *args, **kwargs), guarded)

    def submit(self, fn: Callable[..., T], /, *args: Any, guarded: bool = False, **kwargs: Any) -> Future:
        """Queue a call without awaiting it: cleanup on cancellation paths, which must never block."""
        return self._executor.submit(self._call, partial(fn, *args, **kwargs), guarded)

    def on_thread(self) -> bool:
        return self._thread_id is not None and threading.get_ident() == self._thread_id

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
