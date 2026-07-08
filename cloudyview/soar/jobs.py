"""Small background-job helper for soar's non-blocking overlays."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class JobSnapshot:
    kind: str
    filename: str
    stage: str
    started_at: float
    done: bool = False
    error: str | None = None
    result: object | None = None
    percent: float | None = None
    eta: float | None = None
    note: str | None = None

    @property
    def elapsed(self) -> float:
        return max(0.0, perf_counter() - self.started_at)


class BackgroundJob:
    """Threaded task with queue-based progress handoff to the render loop."""

    def __init__(
        self,
        *,
        kind: str,
        filename: str,
        target: Callable[[Callable[..., None]], object],
        initial_stage: str,
        note: str | None = None,
    ):
        self._messages: Queue[tuple[str, object]] = Queue()
        self._lock = Lock()
        self._snapshot = JobSnapshot(
            kind=kind,
            filename=filename,
            stage=initial_stage,
            started_at=perf_counter(),
            note=note,
        )
        self._target = target
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        self.pump()

    def snapshot(self) -> JobSnapshot:
        self.pump()
        with self._lock:
            return self._snapshot

    def pump(self) -> JobSnapshot:
        while True:
            try:
                kind, payload = self._messages.get_nowait()
            except Empty:
                break
            with self._lock:
                current = self._snapshot
                if kind == "progress":
                    update = dict(payload)
                    self._snapshot = JobSnapshot(
                        kind=current.kind,
                        filename=current.filename,
                        stage=update.get("stage", current.stage),
                        started_at=current.started_at,
                        done=current.done,
                        error=current.error,
                        result=current.result,
                        percent=update.get("percent", current.percent),
                        eta=update.get("eta", current.eta),
                        note=update.get("note", current.note),
                    )
                elif kind == "done":
                    self._snapshot = JobSnapshot(
                        kind=current.kind,
                        filename=current.filename,
                        stage=current.stage,
                        started_at=current.started_at,
                        done=True,
                        result=payload,
                        percent=current.percent,
                        eta=current.eta,
                        note=current.note,
                    )
                elif kind == "error":
                    self._snapshot = JobSnapshot(
                        kind=current.kind,
                        filename=current.filename,
                        stage=current.stage,
                        started_at=current.started_at,
                        done=True,
                        error=str(payload),
                        percent=current.percent,
                        eta=current.eta,
                        note=current.note,
                    )
        with self._lock:
            return self._snapshot

    def _run(self) -> None:
        def report(
            stage: str | None = None,
            *,
            percent: float | None = None,
            eta: float | None = None,
            note: str | None = None,
        ) -> None:
            payload = {}
            if stage is not None:
                payload["stage"] = stage
            if percent is not None:
                payload["percent"] = float(percent)
            if eta is not None:
                payload["eta"] = float(eta)
            if note is not None:
                payload["note"] = note
            self._messages.put(("progress", payload))

        try:
            result = self._target(report)
        except Exception as e:
            self._messages.put(("error", e))
        else:
            self._messages.put(("done", result))
