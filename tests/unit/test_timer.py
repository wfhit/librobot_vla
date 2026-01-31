"""
Unit tests for the timer utilities module.

Tests Timer class, TimerRegistry, and timing utilities.
"""

import time
import threading

import pytest

from librobot.utils.timer import (
    Timer,
    TimerRegistry,
    get_global_timer_registry,
)


class TestTimer:
    """Test suite for Timer class."""

    def test_timer_initialization(self):
        """Test Timer initialization."""
        timer = Timer(name="test_timer")

        assert timer.name == "test_timer"
        assert timer.verbose is True
        assert timer.start_time is None
        assert timer.elapsed is None

    def test_timer_with_verbose_false(self):
        """Test Timer with verbose=False."""
        timer = Timer(name="quiet", verbose=False)

        assert timer.verbose is False

    def test_timer_start_stop(self):
        """Test Timer start and stop."""
        timer = Timer(verbose=False)

        timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = timer.stop()

        assert elapsed is not None
        assert elapsed >= 0.01
        assert timer.elapsed == elapsed

    def test_timer_stop_without_start_raises(self):
        """Test that stopping without starting raises error."""
        timer = Timer(verbose=False)

        with pytest.raises(RuntimeError):
            timer.stop()

    def test_timer_context_manager(self):
        """Test Timer as context manager."""
        with Timer(verbose=False) as timer:
            time.sleep(0.01)

        assert timer.elapsed is not None
        assert timer.elapsed >= 0.01

    def test_timer_context_manager_with_name(self, capsys):
        """Test Timer context manager with name (verbose)."""
        with Timer("named_timer", verbose=True):
            time.sleep(0.01)

        captured = capsys.readouterr()
        assert "named_timer" in captured.out
        assert "Elapsed time" in captured.out

    def test_timer_multiple_measurements(self):
        """Test Timer for multiple measurements."""
        timer = Timer(verbose=False)

        timer.start()
        time.sleep(0.01)
        elapsed1 = timer.stop()

        timer.start()
        time.sleep(0.02)
        elapsed2 = timer.stop()

        assert elapsed1 >= 0.01
        assert elapsed2 >= 0.02
        assert elapsed2 > elapsed1

    def test_timer_precision(self):
        """Test Timer precision (uses perf_counter)."""
        timer = Timer(verbose=False)

        timer.start()
        # Very short operation
        _ = 1 + 1
        elapsed = timer.stop()

        # Should be able to measure very small times
        assert elapsed >= 0
        assert elapsed < 1.0  # Should be much less than 1 second


class TestTimerDecorator:
    """Test suite for Timer decorator."""

    def test_timer_decorator(self, capsys):
        """Test Timer.decorator basic usage."""

        @Timer.decorator("decorated_function", verbose=True)
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()

        assert result == "result"
        captured = capsys.readouterr()
        assert "decorated_function" in captured.out

    def test_timer_decorator_uses_function_name(self, capsys):
        """Test Timer.decorator uses function name when not specified."""

        @Timer.decorator(verbose=True)
        def auto_named_function():
            time.sleep(0.01)

        auto_named_function()

        captured = capsys.readouterr()
        assert "auto_named_function" in captured.out

    def test_timer_decorator_with_args(self):
        """Test Timer.decorator with function arguments."""

        @Timer.decorator(verbose=False)
        def add_numbers(a, b):
            return a + b

        result = add_numbers(3, 4)

        assert result == 7

    def test_timer_decorator_with_kwargs(self):
        """Test Timer.decorator with keyword arguments."""

        @Timer.decorator(verbose=False)
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")

        assert result == "Hi, World!"

    def test_timer_decorator_preserves_function_name(self):
        """Test that decorator preserves function name."""

        @Timer.decorator(verbose=False)
        def original_name():
            pass

        assert original_name.__name__ == "original_name"


class TestTimerRegistry:
    """Test suite for TimerRegistry class."""

    def test_registry_initialization(self):
        """Test TimerRegistry initialization."""
        registry = TimerRegistry()

        assert len(registry._timings) == 0

    def test_registry_timer_context(self):
        """Test TimerRegistry.timer context manager."""
        registry = TimerRegistry()

        with registry.timer("operation"):
            time.sleep(0.01)

        timings = registry.get_timings("operation")
        assert len(timings) == 1
        assert timings[0] >= 0.01

    def test_registry_multiple_timings(self):
        """Test recording multiple timings for same name."""
        registry = TimerRegistry()

        for _ in range(3):
            with registry.timer("repeated"):
                time.sleep(0.01)

        timings = registry.get_timings("repeated")
        assert len(timings) == 3

    def test_registry_different_operations(self):
        """Test recording timings for different operations."""
        registry = TimerRegistry()

        with registry.timer("op1"):
            time.sleep(0.01)

        with registry.timer("op2"):
            time.sleep(0.02)

        timings1 = registry.get_timings("op1")
        timings2 = registry.get_timings("op2")

        assert len(timings1) == 1
        assert len(timings2) == 1
        assert timings2[0] > timings1[0]

    def test_registry_record_manual(self):
        """Test TimerRegistry.record for manual timing."""
        registry = TimerRegistry()

        registry.record("manual", 0.5)
        registry.record("manual", 1.0)

        timings = registry.get_timings("manual")
        assert timings == [0.5, 1.0]

    def test_registry_get_timings_empty(self):
        """Test get_timings for nonexistent name."""
        registry = TimerRegistry()

        timings = registry.get_timings("nonexistent")

        assert timings == []

    def test_registry_get_timings_returns_copy(self):
        """Test that get_timings returns a copy."""
        registry = TimerRegistry()
        registry.record("test", 1.0)

        timings = registry.get_timings("test")
        timings.append(999)

        original = registry.get_timings("test")
        assert 999 not in original

    def test_registry_get_stats(self):
        """Test TimerRegistry.get_stats."""
        registry = TimerRegistry()

        for val in [1.0, 2.0, 3.0]:
            registry.record("stats_test", val)

        stats = registry.get_stats()

        assert "stats_test" in stats
        assert stats["stats_test"]["count"] == 3
        assert stats["stats_test"]["mean"] == 2.0
        assert stats["stats_test"]["min"] == 1.0
        assert stats["stats_test"]["max"] == 3.0
        assert stats["stats_test"]["total"] == 6.0

    def test_registry_get_stats_empty(self):
        """Test get_stats with no timings."""
        registry = TimerRegistry()

        stats = registry.get_stats()

        assert stats == {}

    def test_registry_reset(self):
        """Test TimerRegistry.reset."""
        registry = TimerRegistry()
        registry.record("test", 1.0)

        registry.reset()

        assert registry.get_timings("test") == []

    def test_registry_print_stats(self, capsys):
        """Test TimerRegistry.print_stats."""
        registry = TimerRegistry()
        registry.record("operation", 0.5)
        registry.record("operation", 1.5)

        registry.print_stats()

        captured = capsys.readouterr()
        assert "Timing Statistics" in captured.out
        assert "operation" in captured.out

    def test_registry_print_stats_empty(self, capsys):
        """Test print_stats with no timings."""
        registry = TimerRegistry()

        registry.print_stats()

        captured = capsys.readouterr()
        assert "No timings recorded" in captured.out


class TestTimerRegistryThreadSafety:
    """Test thread safety of TimerRegistry."""

    def test_concurrent_recording(self):
        """Test concurrent recording from multiple threads."""
        registry = TimerRegistry()
        num_threads = 10
        num_records = 100

        def record_timings(thread_id):
            for i in range(num_records):
                registry.record(f"thread_{thread_id}", float(i))

        threads = [
            threading.Thread(target=record_timings, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(num_threads):
            timings = registry.get_timings(f"thread_{i}")
            assert len(timings) == num_records

    def test_concurrent_timer_context(self):
        """Test concurrent timer context managers."""
        registry = TimerRegistry()
        num_threads = 5
        results = []

        def use_timer():
            with registry.timer("concurrent"):
                time.sleep(0.01)
            results.append(True)

        threads = [
            threading.Thread(target=use_timer) for _ in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == num_threads
        timings = registry.get_timings("concurrent")
        assert len(timings) == num_threads


class TestGlobalTimerRegistry:
    """Test suite for global timer registry."""

    def test_get_global_registry(self):
        """Test get_global_timer_registry returns TimerRegistry."""
        registry = get_global_timer_registry()

        assert isinstance(registry, TimerRegistry)

    def test_global_registry_is_singleton(self):
        """Test that global registry is the same instance."""
        registry1 = get_global_timer_registry()
        registry2 = get_global_timer_registry()

        assert registry1 is registry2

    def test_global_registry_usage(self):
        """Test using global registry."""
        registry = get_global_timer_registry()
        registry.reset()  # Clear any previous timings

        with registry.timer("global_test"):
            time.sleep(0.01)

        timings = registry.get_timings("global_test")
        assert len(timings) >= 1


class TestTimerEdgeCases:
    """Test edge cases and error handling."""

    def test_timer_with_exception_in_context(self):
        """Test Timer context manager handles exceptions."""
        timer = Timer(verbose=False)

        with pytest.raises(ValueError):
            with timer:
                raise ValueError("Test error")

        # Timer should still have recorded elapsed time
        assert timer.elapsed is not None

    def test_registry_timer_with_exception(self):
        """Test TimerRegistry.timer handles exceptions."""
        registry = TimerRegistry()

        with pytest.raises(ValueError):
            with registry.timer("exception_test"):
                raise ValueError("Test error")

        # Timing should still be recorded
        timings = registry.get_timings("exception_test")
        assert len(timings) == 1

    def test_timer_decorator_with_exception(self):
        """Test Timer.decorator handles exceptions."""

        @Timer.decorator(verbose=False)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_very_short_timing(self):
        """Test measuring very short operations."""
        timer = Timer(verbose=False)

        timer.start()
        # Essentially no operation
        elapsed = timer.stop()

        assert elapsed >= 0
        assert elapsed < 0.01

    def test_registry_stats_single_timing(self):
        """Test stats with single timing."""
        registry = TimerRegistry()
        registry.record("single", 1.0)

        stats = registry.get_stats()

        assert stats["single"]["count"] == 1
        assert stats["single"]["mean"] == 1.0
        assert stats["single"]["std"] == 0.0
        assert stats["single"]["min"] == 1.0
        assert stats["single"]["max"] == 1.0
