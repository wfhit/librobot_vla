"""
Unit tests for the logging utilities module.

Tests Logger class, logging setup, and convenience functions.
"""

import logging
from pathlib import Path

import pytest

from librobot.utils.logging import (
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
    ColoredFormatter,
    Logger,
    get_logger,
    get_default_logger,
    set_default_logger,
    setup_logging,
    debug,
    info,
    warning,
    error,
    critical,
    exception,
)


class TestLogLevelConstants:
    """Test suite for log level constants."""

    def test_debug_level(self):
        """Test DEBUG level constant."""
        assert DEBUG == logging.DEBUG
        assert DEBUG == 10

    def test_info_level(self):
        """Test INFO level constant."""
        assert INFO == logging.INFO
        assert INFO == 20

    def test_warning_level(self):
        """Test WARNING level constant."""
        assert WARNING == logging.WARNING
        assert WARNING == 30

    def test_error_level(self):
        """Test ERROR level constant."""
        assert ERROR == logging.ERROR
        assert ERROR == 40

    def test_critical_level(self):
        """Test CRITICAL level constant."""
        assert CRITICAL == logging.CRITICAL
        assert CRITICAL == 50


class TestColoredFormatter:
    """Test suite for ColoredFormatter."""

    def test_formatter_has_colors(self):
        """Test that formatter has color definitions."""
        assert hasattr(ColoredFormatter, "COLORS")
        assert "DEBUG" in ColoredFormatter.COLORS
        assert "INFO" in ColoredFormatter.COLORS
        assert "WARNING" in ColoredFormatter.COLORS
        assert "ERROR" in ColoredFormatter.COLORS
        assert "CRITICAL" in ColoredFormatter.COLORS
        assert "RESET" in ColoredFormatter.COLORS

    def test_formatter_format(self):
        """Test that formatter formats records."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test",
            level=INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "Test message" in formatted


class TestLoggerClass:
    """Test suite for Logger class."""

    def test_logger_initialization(self):
        """Test Logger initialization."""
        logger = Logger("test_logger")

        assert logger.logger is not None
        assert logger.logger.name == "test_logger"

    def test_logger_with_level(self):
        """Test Logger with custom level."""
        logger = Logger("debug_logger", level=DEBUG)

        assert logger.logger.level == DEBUG

    def test_logger_handlers_cleared(self):
        """Test that Logger clears existing handlers."""
        # Create logger twice with same name
        logger1 = Logger("same_name")
        handler_count1 = len(logger1.logger.handlers)

        logger2 = Logger("same_name")
        handler_count2 = len(logger2.logger.handlers)

        # Should have same number of handlers (cleared and recreated)
        assert handler_count1 == handler_count2

    def test_logger_with_file(self, tmp_path):
        """Test Logger with file output."""
        log_file = tmp_path / "test.log"
        logger = Logger("file_logger", log_file=log_file)

        logger.info("Test message")

        # File should exist and contain message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_logger_creates_parent_dirs(self, tmp_path):
        """Test Logger creates parent directories for log file."""
        log_file = tmp_path / "nested" / "dir" / "test.log"
        logger = Logger("nested_logger", log_file=log_file)

        logger.info("Test")

        assert log_file.exists()

    def test_logger_use_color(self):
        """Test Logger with color option."""
        logger_color = Logger("color", use_color=True)
        logger_no_color = Logger("no_color", use_color=False)

        # Both should work
        assert logger_color.logger is not None
        assert logger_no_color.logger is not None

    def test_logger_debug(self, caplog):
        """Test Logger.debug method."""
        logger = Logger("debug_test", level=DEBUG)

        with caplog.at_level(DEBUG):
            logger.debug("Debug message")

        assert "Debug message" in caplog.text

    def test_logger_info(self, caplog):
        """Test Logger.info method."""
        logger = Logger("info_test", level=INFO)

        with caplog.at_level(INFO):
            logger.info("Info message")

        assert "Info message" in caplog.text

    def test_logger_warning(self, caplog):
        """Test Logger.warning method."""
        logger = Logger("warning_test", level=WARNING)

        with caplog.at_level(WARNING):
            logger.warning("Warning message")

        assert "Warning message" in caplog.text

    def test_logger_error(self, caplog):
        """Test Logger.error method."""
        logger = Logger("error_test", level=ERROR)

        with caplog.at_level(ERROR):
            logger.error("Error message")

        assert "Error message" in caplog.text

    def test_logger_critical(self, caplog):
        """Test Logger.critical method."""
        logger = Logger("critical_test", level=CRITICAL)

        with caplog.at_level(CRITICAL):
            logger.critical("Critical message")

        assert "Critical message" in caplog.text

    def test_logger_exception(self, caplog):
        """Test Logger.exception method."""
        logger = Logger("exception_test", level=ERROR)

        with caplog.at_level(ERROR):
            try:
                raise ValueError("Test error")
            except ValueError:
                logger.exception("Exception occurred")

        assert "Exception occurred" in caplog.text

    def test_logger_set_level(self, caplog):
        """Test Logger.set_level method."""
        logger = Logger("level_test", level=ERROR)

        # Initially ERROR level
        with caplog.at_level(DEBUG):
            logger.debug("Should not appear")

        initial_text = caplog.text

        # Change to DEBUG level
        logger.set_level(DEBUG)

        with caplog.at_level(DEBUG):
            logger.debug("Should appear")

        assert "Should not appear" not in initial_text
        assert "Should appear" in caplog.text


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("my_logger")

        assert isinstance(logger, Logger)

    def test_get_logger_with_name(self):
        """Test get_logger with custom name."""
        logger = get_logger(name="custom_name")

        assert logger.logger.name == "custom_name"

    def test_get_logger_default_name(self):
        """Test get_logger with default name."""
        logger = get_logger()

        assert logger.logger.name == "librobot"

    def test_get_logger_with_level(self):
        """Test get_logger with custom level."""
        logger = get_logger(level=DEBUG)

        assert logger.logger.level == DEBUG

    def test_get_logger_with_file(self, tmp_path):
        """Test get_logger with log file."""
        log_file = tmp_path / "get_logger.log"
        logger = get_logger(log_file=log_file)

        logger.info("Test")

        assert log_file.exists()

    @pytest.mark.parametrize("use_color", [True, False])
    def test_get_logger_use_color(self, use_color):
        """Test get_logger with color option."""
        logger = get_logger(use_color=use_color)

        assert logger is not None


class TestDefaultLogger:
    """Test suite for default logger functions."""

    def test_get_default_logger_creates_if_needed(self):
        """Test get_default_logger creates logger if not set."""
        # Reset by setting to None (internal)
        import librobot.utils.logging as logging_module

        logging_module._default_logger = None

        logger = get_default_logger()

        assert isinstance(logger, Logger)

    def test_set_default_logger(self):
        """Test set_default_logger."""
        custom_logger = Logger("custom_default")

        set_default_logger(custom_logger)

        assert get_default_logger() is custom_logger

    def test_setup_logging(self):
        """Test setup_logging function."""
        logger = setup_logging(level=DEBUG)

        assert isinstance(logger, Logger)
        assert get_default_logger() is logger

    def test_setup_logging_with_file(self, tmp_path):
        """Test setup_logging with log file."""
        log_file = tmp_path / "setup.log"
        logger = setup_logging(log_file=log_file)

        logger.info("Setup test")

        assert log_file.exists()


class TestConvenienceFunctions:
    """Test suite for convenience logging functions."""

    def setup_method(self):
        """Reset default logger before each test."""
        setup_logging(level=DEBUG)

    def test_debug_function(self, caplog):
        """Test debug convenience function."""
        with caplog.at_level(DEBUG):
            debug("Debug via function")

        assert "Debug via function" in caplog.text

    def test_info_function(self, caplog):
        """Test info convenience function."""
        with caplog.at_level(INFO):
            info("Info via function")

        assert "Info via function" in caplog.text

    def test_warning_function(self, caplog):
        """Test warning convenience function."""
        with caplog.at_level(WARNING):
            warning("Warning via function")

        assert "Warning via function" in caplog.text

    def test_error_function(self, caplog):
        """Test error convenience function."""
        with caplog.at_level(ERROR):
            error("Error via function")

        assert "Error via function" in caplog.text

    def test_critical_function(self, caplog):
        """Test critical convenience function."""
        with caplog.at_level(CRITICAL):
            critical("Critical via function")

        assert "Critical via function" in caplog.text

    def test_exception_function(self, caplog):
        """Test exception convenience function."""
        with caplog.at_level(ERROR):
            try:
                raise ValueError("Test")
            except ValueError:
                exception("Exception via function")

        assert "Exception via function" in caplog.text


class TestLoggerWithFormattedMessages:
    """Test Logger with formatted messages."""

    def test_logger_with_args(self, caplog):
        """Test Logger with format arguments."""
        logger = Logger("format_test", level=INFO)

        with caplog.at_level(INFO):
            logger.info("Value: %s", 42)

        assert "Value: 42" in caplog.text

    def test_logger_with_multiple_args(self, caplog):
        """Test Logger with multiple format arguments."""
        logger = Logger("multi_arg_test", level=INFO)

        with caplog.at_level(INFO):
            logger.info("Values: %s, %s, %s", 1, 2, 3)

        assert "Values: 1, 2, 3" in caplog.text

    def test_logger_with_kwargs(self, caplog):
        """Test Logger with extra kwargs."""
        logger = Logger("kwargs_test", level=INFO)

        with caplog.at_level(INFO):
            logger.info("Test message", extra={"custom": "value"})

        assert "Test message" in caplog.text
