r"""
Logging utilities that make it easier to simultaneously log to a file and standard output.

Notes
-----
#.  Split out of utils by David C. Stauffer in July 2019.
"""

#%% Imports
import datetime
import doctest
import logging
from pathlib import Path
from typing import Any, List, Union
import unittest

from dstauffman.paths import get_output_dir

#%% Globals
root_logger = logging.getLogger('')
logger = logging.getLogger(__name__)

#%% Functions - activate_logging
def activate_logging(
    log_level: int = logging.INFO,
    filename: Union[str, Path] = None,
    *,
    file_level: int = None,
    log_format: str = None,
    file_format: str = None
) -> None:
    r"""
    Set up logging based on a user specified settings file.

    Parameters
    ----------
    log_level : int
        Level of logging
    filename : pathlib.Path
        File to log to, if empty, use default output folder with today's date
    file_level : int, optional
        Level of logging for the file, if not specified, use the same as the screen logger
    log_format : str, optional
        Format for the screen log level
    file_format : str, optional
        Format for the file log level

    Notes
    -----
    #.  Written by David C. Stauffer in August 2017.

    Examples
    --------
    >>> from dstauffman import activate_logging, deactivate_logging, get_tests_dir, LogLevel
    >>> import logging
    >>> filename = get_tests_dir() / 'testlog.txt'
    >>> activate_logging(log_level=LogLevel.L5, filename=filename)
    >>> logging.log(LogLevel.L5, 'Test message') # doctest: +SKIP
    >>> deactivate_logging()

    Remove the log file
    >>> filename.unlink()

    """
    # defaults
    if log_format is None:
        log_format = 'Log:%(levelname)s: %(message)s'
    if file_format is None:
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if file_level is None:
        file_level = log_level
    use_file = not isinstance(filename, str) or filename != 'none'
    # deactivate any current loggers
    deactivate_logging()

    # update the log level
    root_logger.setLevel(log_level)

    # optionally get the default filename
    if use_file:
        if isinstance(filename, str):
            filename = Path(filename) if filename != '' else None
        if filename is None:
            filename = get_output_dir().joinpath('log_file_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.txt')

        # create the log file handler
        fh = logging.FileHandler(filename)
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(fh)

    # create the log stream handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(ch)


#%% Functions - deactivate_logging
def deactivate_logging() -> None:
    r"""
    Tear down logging.

    Notes
    -----
    #.  Written by David C. Stauffer in August 2017.

    Examples
    --------
    >>> from dstauffman import deactivate_logging
    >>> deactivate_logging()

    """
    # hard-coded values
    max_handlers = 50
    # initialize a counter to avoid infinite while loop
    i = 0
    # loop through and remove all the handlers
    while root_logger.handlers and i < max_handlers:
        handler = root_logger.handlers.pop()
        handler.flush()
        handler.close()
        root_logger.removeHandler(handler)
        # increment the counter
        i += 1
    # check for bad situations
    if i == max_handlers or bool(root_logger.handlers):
        raise ValueError('Something bad happended when trying to close the logger.')  # pragma: no cover


#%% Functions - flush_logging
def flush_logging() -> None:
    r"""
    Flush the loggers.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2021.

    Examples
    --------
    >>> from dstauffman import flush_logging
    >>> flush_logging()

    """
    # loop through and flush all the handlers
    for handler in root_logger.handlers:
        handler.flush()


#%% Functions - log_multiline
def log_multiline(logger: logging.Logger, log_level: int, message: Any, *args: Any) -> None:
    r"""
    Passes messages through to the logger with options for multiline messages.

    Parameters
    ----------
    logger : class logging.Logger
        Logger
    log_level : int
        Log level
    message : str or list[str] or numeric
        Value to log
    args : list of additional arguments to log
        Additional options to log

    Examples
    --------
    >>> from dstauffman import activate_logging, deactivate_logging, log_multiline, LogLevel
    >>> import logging
    >>> logger = logging.getLogger('Test')
    >>> log_level = LogLevel.L5
    >>> activate_logging(log_level)
    >>> log_multiline(logger, log_level, 'Multi-line\nmessage') # doctest: +SKIP

    >>> deactivate_logging()

    """

    def _get_message_list(message: Any) -> List[str]:
        if isinstance(message, list):
            # if message is already a list, then make sure everything is already a string
            if all(isinstance(x, str) for x in message):
                return message
            return [str(message)]
        if isinstance(message, str):
            # if message is a string, then split it on every new line
            return message.split('\n')
        # otherwise, convert message to a string, and then split on every new line
        return str(message).split('\n')

    # if there are additional arguments, then append them with the same rules
    all_msg = _get_message_list(message)
    if args:
        for x in args:
            all_msg.extend(_get_message_list(x))
    # log all the messages
    for msg in all_msg:
        logger.log(log_level, msg)


#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_logs', exit=False)
    doctest.testmod(verbose=False)
