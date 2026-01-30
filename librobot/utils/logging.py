"""Structured logging system with multiple levels and formatters."""

import sys
import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


# Define log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for terminal output.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']

        # Add color to levelname
        record.levelname = f"{log_color}{record.levelname}{reset_color}"

        return super().format(record)


class Logger:
    """
    Wrapper around Python logging with additional features.

    Examples:
        >>> logger = Logger("my_module")
        >>> logger.info("Processing data")
        >>> logger.warning("Low memory")
        >>> logger.error("Failed to load file", exc_info=True)
    """

    def __init__(
        self,
        name: str,
        level: int = INFO,
        log_file: Optional[Union[str, Path]] = None,
        use_color: bool = True,
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            level: Logging level
            log_file: Optional file path for logging
            use_color: If True, uses colored output for console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if use_color:
            console_format = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)

            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)

    def set_level(self, level: int) -> None:
        """
        Set logging level.

        Args:
            level: New logging level
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


# Global logger instance
_default_logger: Optional[Logger] = None


def get_logger(
    name: str = "librobot",
    level: int = INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_color: bool = True,
) -> Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        use_color: If True, uses colored output for console

    Returns:
        Logger: Logger instance
    """
    return Logger(name=name, level=level, log_file=log_file, use_color=use_color)


def get_default_logger() -> Logger:
    """
    Get the default global logger.

    Returns:
        Logger: Default logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    return _default_logger


def set_default_logger(logger: Logger) -> None:
    """
    Set the default global logger.

    Args:
        logger: Logger instance to set as default
    """
    global _default_logger
    _default_logger = logger


def setup_logging(
    level: int = INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_color: bool = True,
) -> Logger:
    """
    Setup global logging configuration.

    Args:
        level: Logging level
        log_file: Optional file path for logging
        use_color: If True, uses colored output for console

    Returns:
        Logger: Configured logger instance
    """
    logger = get_logger(level=level, log_file=log_file, use_color=use_color)
    set_default_logger(logger)
    return logger


# Convenience functions using default logger
def debug(msg: str, *args, **kwargs) -> None:
    """Log debug message using default logger."""
    get_default_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log info message using default logger."""
    get_default_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log warning message using default logger."""
    get_default_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log error message using default logger."""
    get_default_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log critical message using default logger."""
    get_default_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs) -> None:
    """Log exception with traceback using default logger."""
    get_default_logger().exception(msg, *args, **kwargs)
