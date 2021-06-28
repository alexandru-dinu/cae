"""
Inspired by https://github.com/SebiSebi/friendlylog
"""

import logging
import sys
from copy import copy
from typing import Union

from colored import attr, fg

DEBUG = "debug"
INFO = "info"
WARNING = "warning"
ERROR = "error"
CRITICAL = "critical"

LOG_LEVELS = {
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARNING: logging.WARNING,
    ERROR: logging.ERROR,
    CRITICAL: logging.CRITICAL,
}


class _Formatter(logging.Formatter):
    def __init__(self, colorize=False, *args, **kwargs):
        super(_Formatter, self).__init__(*args, **kwargs)
        self.colorize = colorize

    @staticmethod
    def _process(msg, loglevel, colorize):
        loglevel = str(loglevel).lower()
        if loglevel not in LOG_LEVELS:
            raise RuntimeError(
                f"{loglevel} should be one of {LOG_LEVELS}."
            )  # pragma: no cover

        msg = f"{str(loglevel).upper()}: {str(msg)}"

        if not colorize:
            return msg

        if loglevel == DEBUG:
            return "{}{}{}".format(fg(5), msg, attr(0))  # noqa: E501
        if loglevel == INFO:
            return "{}{}{}".format(fg(4), msg, attr(0))  # noqa: E501
        if loglevel == WARNING:
            return "{}{}{}{}{}".format(
                fg(214), attr(1), msg, attr(21), attr(0)
            )  # noqa: E501
        if loglevel == ERROR:
            return "{}{}{}{}{}".format(
                fg(202), attr(1), msg, attr(21), attr(0)
            )  # noqa: E501
        if loglevel == CRITICAL:
            return "{}{}{}{}{}".format(
                fg(196), attr(1), msg, attr(21), attr(0)
            )  # noqa: E501

    def format(self, record):
        record = copy(record)
        loglevel = record.levelname
        record.msg = _Formatter._process(record.msg, loglevel, self.colorize)
        return super(_Formatter, self).format(record)


class Logger:
    def __init__(self, name="default", colorize=False, stream=sys.stdout, level=DEBUG):
        self.name = name

        # get the logger object; keep it hidden as there's no need to directly access it
        self.__logger = logging.getLogger(f"_logger-{name}")
        self.__logger.propagate = False
        self.setLevel(level.lower())

        # use the custom formatter
        self.__formatter = _Formatter(
            colorize=colorize,
            fmt="[%(process)d][%(asctime)s.%(msecs)03d @ %(funcName)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )

        # install default handler
        self.__stream_to_handler = {}
        self.clear_handlers()
        self.__main_handler = self.add_handler(stream)

        # install logging functions
        self.debug = self.__logger.debug
        self.info = self.__logger.info
        self.warning = self.__logger.warning
        self.error = self.__logger.error
        self.critical = self.__logger.critical

    def log_function(self):
        def wrapper(func):
            def func_wrapper(*args, **kwargs):
                self.__logger.info(
                    f"calling <{func.__name__}>\n\t  args: {args}\n\tkwargs: {kwargs}"
                )
                out = func(*args, **kwargs)
                self.__logger.info(f"exiting <{func.__name__}>")
                return out

            return func_wrapper

        return wrapper

    def setLevel(self, level: Union[str, int]) -> None:
        if isinstance(level, int):
            self.__logger.setLevel(level)
        else:
            if level.lower() not in LOG_LEVELS:
                raise ValueError(f"level should be one of {LOG_LEVELS}")
            self.__logger.setLevel(LOG_LEVELS[level.lower()])

    def add_handler(self, stream) -> logging.StreamHandler:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(self.__formatter)
        self.__logger.addHandler(handler)
        self.__stream_to_handler[stream] = handler
        return handler

    def remove_handler(self, stream) -> bool:
        if stream in self.__stream_to_handler:
            self.__logger.removeHandler(self.__stream_to_handler[stream])
            self.__stream_to_handler.pop(stream)
            return True
        return False

    def clear_handlers(self) -> None:
        self.__logger.handlers = []
        self.__stream_to_handler = {}

    def get_handlers(self) -> list:
        return self.__logger.handlers

    # Don't use these unless you know what you are doing

    @property
    def inner_logger(self):
        return self.__logger

    @property
    def inner_stream_handler(self):
        return self.__main_handler

    @property
    def inner_formatter(self):
        return self.__formatter
