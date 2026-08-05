"""Tests for process-tree RSS telemetry."""

import time

import pytest

from agenticai_thesis.utils.resource_monitor import ProcessMemoryMonitor


def test_memory_monitor_returns_consistent_byte_and_megabyte_summary() -> None:
    with ProcessMemoryMonitor(sampling_interval_seconds=0.005) as monitor:
        allocation = bytearray(2 * 1024 * 1024)
        allocation[0] = 1
        time.sleep(0.015)

    summary = monitor.summary
    assert summary.start_rss_bytes > 0
    assert summary.peak_rss_bytes >= summary.start_rss_bytes
    assert summary.peak_increase_bytes == summary.peak_rss_bytes - summary.start_rss_bytes
    assert summary.samples >= 2
    assert summary.peak_rss_megabytes == pytest.approx(summary.peak_rss_bytes / (1024**2))


def test_memory_monitor_is_single_use() -> None:
    monitor = ProcessMemoryMonitor()
    with monitor:
        pass

    with pytest.raises(RuntimeError, match="single-use"):
        monitor.start()
