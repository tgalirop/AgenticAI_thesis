"""Process-level resource telemetry for reproducible experiment stages."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from types import TracebackType

import psutil


@dataclass(frozen=True, slots=True)
class MemoryUsageSummary:
    """Compact RSS evidence captured during one measured code region."""

    start_rss_bytes: int
    peak_rss_bytes: int
    peak_increase_bytes: int
    samples: int

    @property
    def start_rss_megabytes(self) -> float:
        return self.start_rss_bytes / (1024**2)

    @property
    def peak_rss_megabytes(self) -> float:
        return self.peak_rss_bytes / (1024**2)

    @property
    def peak_increase_megabytes(self) -> float:
        return self.peak_increase_bytes / (1024**2)

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-ready byte and MiB values without losing precision."""

        return {
            "start_rss_bytes": self.start_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_increase_bytes": self.peak_increase_bytes,
            "start_rss_megabytes": self.start_rss_megabytes,
            "peak_rss_megabytes": self.peak_rss_megabytes,
            "peak_increase_megabytes": self.peak_increase_megabytes,
            "samples": self.samples,
        }


class ProcessMemoryMonitor:
    """Sample resident memory for this process and all of its child processes.

    RSS is sampled because Python-only allocation trackers omit native memory
    owned by NumPy, pandas, Polars and scikit-learn.  Including child processes
    also captures estimator workers when an implementation uses joblib.
    """

    def __init__(self, sampling_interval_seconds: float = 0.05) -> None:
        if sampling_interval_seconds <= 0:
            raise ValueError("sampling_interval_seconds must be positive")
        self._interval = sampling_interval_seconds
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_rss = 0
        self._peak_rss = 0
        self._samples = 0
        self._summary: MemoryUsageSummary | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "ProcessMemoryMonitor":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        """Start background sampling; a monitor instance is single-use."""

        if self._thread is not None or self._summary is not None:
            raise RuntimeError("ProcessMemoryMonitor instances are single-use")
        initial = self._resident_bytes()
        self._start_rss = initial
        self._peak_rss = initial
        self._samples = 1
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="agentic-memory-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> MemoryUsageSummary:
        """Stop sampling and return the immutable measurement summary."""

        if self._thread is None:
            raise RuntimeError("ProcessMemoryMonitor has not been started")
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self._interval * 4))
        self._sample_once()
        with self._lock:
            self._summary = MemoryUsageSummary(
                start_rss_bytes=self._start_rss,
                peak_rss_bytes=self._peak_rss,
                peak_increase_bytes=max(0, self._peak_rss - self._start_rss),
                samples=self._samples,
            )
        self._thread = None
        return self._summary

    @property
    def summary(self) -> MemoryUsageSummary:
        """Expose results only after the measured block has completed."""

        if self._summary is None:
            raise RuntimeError("Memory summary is available only after stop")
        return self._summary

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._interval):
            self._sample_once()

    def _sample_once(self) -> None:
        resident = self._resident_bytes()
        with self._lock:
            self._peak_rss = max(self._peak_rss, resident)
            self._samples += 1

    def _resident_bytes(self) -> int:
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total
