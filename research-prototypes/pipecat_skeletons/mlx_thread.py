"""One thread for every MLX call in the orchestrator process.

Why (research note 04c, 2026-10-05):
- MLX keeps a default stream per thread; an array created or realised on one thread and used on another
  fails ("There is no Stream(gpu, 0) in current thread", Ianodad/voice-stack, note 04b).
- Pipecat 1.12's own local services hop threads: WhisperSTTServiceMLX calls asyncio.to_thread
  (pipecat/services/whisper/stt.py:590-596) and PocketTTSService steps its generator with
  asyncio.to_thread(sync_next, stream) (pipecat/services/pocket_tts/tts.py:196-212). asyncio.to_thread uses
  the loop's default multi-thread pool, so consecutive calls can land on different threads. Do not copy that
  for MLX; route every MLX call (STT, TTS, warm-up, model load) through run_mlx() below.
- Smart Turn and Silero are ONNX on CPU and already run on their own one-thread executors
  (base_smart_turn.py:154-183, vad_analyzer.py:92 and 178-191); they do not need this thread.

Cost: STT and TTS share one thread, so a transcription waits behind a TTS chunk that is being generated.
Keep each submitted call short (one TTS chunk per call, never a whole utterance) so the wait stays at one
chunk. A call that is already running cannot be cancelled; cancellation only stops the next submission.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")

MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_mlx_thread_id: int | None = None


def _remember_thread() -> int:
    global _mlx_thread_id
    _mlx_thread_id = threading.get_ident()
    return _mlx_thread_id


async def run_mlx(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a blocking MLX call on the single MLX thread and await its result."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(MLX_EXECUTOR, partial(fn, *args, **kwargs))


def submit_mlx(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
    """Queue an MLX call without awaiting it (cleanup on cancellation paths, which must not block)."""
    return MLX_EXECUTOR.submit(fn, *args, **kwargs)


def on_mlx_thread() -> bool:
    """True when called from the MLX thread (for assertions inside engines)."""
    return _mlx_thread_id is not None and threading.get_ident() == _mlx_thread_id


# Pin the thread identity as soon as the module is imported, without touching MLX itself.
MLX_EXECUTOR.submit(_remember_thread).result()
