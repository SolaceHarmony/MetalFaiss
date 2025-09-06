"""
Stream utilities: background waiters and callbacks (plain-speech helpers)

These helpers let you trigger host-side work when a specific MLX stream
finishes, without blocking other streams or the main/UI thread. They also
support an evaluation-based trigger when you care about specific arrays.

Notes
- Always target a specific `stream` with `mx.synchronize(stream)` rather than a
  global synchronize. This keeps other queues running.
- For device-to-device chaining, prefer passing arrays from stream A into ops on
  stream B; MLX inserts minimal waits. Use these helpers for host callbacks
  (logging, checkpointing, UI updates, etc.).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Iterable, Optional, Sequence

import mlx.core as mx


def on_stream_complete(
    stream: mx.core.Stream,
    callback: Callable[..., Any],
    *args: Any,
    executor: Optional[ThreadPoolExecutor] = None,
    **kwargs: Any,
) -> threading.Thread | Future:
    """Wait for `stream` to complete in the background, then call `callback`.

    This confines the wait to a worker thread and scopes synchronization to the
    given stream, preserving overlap across other streams and devices.

    Parameters
    - stream: MLX Stream to wait on (required; do not pass None).
    - callback: function to run once the stream is done.
    - *args/**kwargs: forwarded to the callback.
    - executor: optional ThreadPoolExecutor; if provided, returns a Future; else
      returns the created daemon Thread.
    """

    def wait_and_call() -> Any:
        mx.synchronize(stream)  # stream-scoped wait
        return callback(*args, **kwargs)

    if executor is not None:
        return executor.submit(wait_and_call)
    t = threading.Thread(target=wait_and_call, daemon=True)
    t.start()
    return t


async def on_stream_complete_async(
    stream: mx.core.Stream,
    callback: Callable[..., Any],
    *args: Any,
    loop=None,
    executor: Optional[ThreadPoolExecutor] = None,
    **kwargs: Any,
) -> Any:
    """Async variant: wait for `stream` off-loop and invoke `callback`.

    Uses the event loop's executor to avoid blocking the event loop while the
    stream synchronizes. Returns the callback's result.
    """
    if loop is None:
        import asyncio

        loop = asyncio.get_running_loop()

    def wait_and_call() -> Any:
        mx.synchronize(stream)
        return callback(*args, **kwargs)

    return await loop.run_in_executor(executor, wait_and_call)


def after_eval(
    arrays: Sequence[mx.core.array],
    callback: Callable[..., Any],
    *args: Any,
    executor: Optional[ThreadPoolExecutor] = None,
    **kwargs: Any,
) -> Future:
    """Evaluate arrays in a worker and then run `callback`.

    Useful when your trigger is "these concrete results are ready" rather than
    "this stream finished". Keeps evaluation off the main/UI thread.
    """
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1)

    def wait_and_call() -> Any:
        mx.eval(*arrays)  # compute concrete results off-thread
        return callback(*args, **kwargs)

    return executor.submit(wait_and_call)

