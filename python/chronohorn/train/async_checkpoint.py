"""Async checkpoint writer — overlap the training-state disk write with training.

The checkpoint covenant mandates frequent periodic saves; each writes the full
resume point (model + optimizer moments + RNG + AMP scale + gear), and on a slow
disk the ``torch.save()`` stalls the training loop for seconds. This offloads the
WRITE to a background thread so training continues — but only after a
SYNCHRONOUS, CONSISTENT snapshot: ``snapshot_training_state`` clones every tensor
to CPU at the checkpoint step, so the async task serializes a frozen copy, never
live-mutating parameters.

COVENANT-SAFE BY CONSTRUCTION. The writer touches no training RNG, no optimizer,
no data stream — it is pure I/O over a snapshot taken by the caller. So enabling
it cannot change the training trajectory or the resumable state; it only changes
WHEN the identical bytes hit disk. Disabled by default (byte-identical to the old
synchronous path); opt in with ``CHRONOHORN_ASYNC_CHECKPOINT=1``.

ORDERING GUARANTEE. A single worker thread serializes writes in submission order,
and the previous training-state file is deleted only AFTER the new one is fully
written (inside the same task) — so a crash never leaves zero resumable states,
the invariant the synchronous write-then-delete held.
"""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Any


def async_checkpoint_enabled() -> bool:
    """Opt-in flag: CHRONOHORN_ASYNC_CHECKPOINT=1 turns on background writes."""
    try:
        return int(os.environ.get("CHRONOHORN_ASYNC_CHECKPOINT", "0")) != 0
    except ValueError:
        return False


def _deep_cpu_clone(obj: Any) -> Any:
    """Recursively detach+clone tensors to CPU; rebuild containers; pass scalars."""
    import torch

    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _deep_cpu_clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_deep_cpu_clone(v) for v in obj)
    return obj


def snapshot_training_state(model, optimizer, *, step: int, gear, grad_scaler,
                            rng_cpu, rng_cuda) -> dict:
    """A consistent CPU snapshot of the resume point, taken synchronously.

    Clones the model and optimizer tensors (which the next optimizer step will
    mutate); the RNG states are already immutable CPU byte tensors and the AMP
    scaler / gear are small python objects. The returned dict is safe to hand to
    a background writer while training continues.
    """
    return {
        "model": _deep_cpu_clone(model.state_dict()),
        "optimizer": _deep_cpu_clone(optimizer.state_dict()),
        "step": int(step),
        "rng_cpu": rng_cpu,
        "rng_cuda": rng_cuda,
        "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
        "gear": gear,
    }


def _write(snapshot: dict, path: str, prev: str | None) -> None:
    import torch

    torch.save(snapshot, path)
    if prev is not None and prev != path:
        Path(prev).unlink(missing_ok=True)


class AsyncCheckpointWriter:
    """Single background thread writing training-state snapshots to disk.

    ``submit(snapshot, path, prev)`` queues a write; the worker ``torch.save``s
    the snapshot then deletes ``prev``. ``close()`` blocks until every queued
    write finishes. When disabled, ``submit`` writes synchronously (identical to
    the former inline path). A failed write is re-raised on the next
    ``submit``/``close`` so it never passes silently.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._q: "queue.Queue" = queue.Queue()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(
                target=self._run, name="ckpt-writer", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                snapshot, path, prev = item
                try:
                    _write(snapshot, path, prev)
                except BaseException as e:  # noqa: BLE001 — surfaced on submit/close
                    if self._error is None:
                        self._error = e
            finally:
                self._q.task_done()

    def _raise_pending(self) -> None:
        if self._error is not None:
            err, self._error = self._error, None
            raise err

    def submit(self, snapshot: dict, path, prev=None) -> None:
        self._raise_pending()
        if not self.enabled:
            _write(snapshot, str(path), str(prev) if prev is not None else None)
            return
        self._q.put((snapshot, str(path), str(prev) if prev is not None else None))

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join()
            self._thread = None
        self._raise_pending()
