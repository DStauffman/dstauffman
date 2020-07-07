r"""
Logging utilities that make it easier to simultaneously log to a file and standard output.

Notes
-----
#.  Split out of utils by David C. Stauffer in July 2019.
"""

#%% Imports
import doctest
import logging
import os
import unittest
from datetime import datetime

from dstauffman.paths import get_output_dir

#%% Globals
root_logger = logging.getLogger('')
logger      = logging.getLogger(__name__)

#%% Functions - activate_logging
def activate_logging(log_level=logging.INFO, filename=''):
    r"""
    Set up logging based on a user specified settings file.

    Parameters
    ----------
    log_level : int
        Level of logging
    filename : str
        File to log to, if empty, use default output folder with today's date

    Notes
    -----
    #.  Written by David C. Stauffer in August 2017.

    Examples
    --------
    >>> from dstauffman import activate_logging, deactivate_logging, get_tests_dir
    >>> import logging
    >>> import os
    >>> filename = os.path.join(get_tests_dir(), 'testlog.txt')
    >>> activate_logging(log_level=logging.DEBUG, filename=filename)
    >>> logging.debug('Test message') # doctest: +SKIP
    >>> deactivate_logging()

    Remove the log file
    >>> os.remove(filename)

    """
    # update the log level
    root_logger.setLevel(log_level)

    # optionally get the default filename
    if not filename:
        filename = os.path.join(get_output_dir(), 'log_file_' + datetime.now().strftime('%Y-%m-%d') + '.txt')

    # create the log file handler
    fh = logging.FileHandler(filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(fh)

    # create the log stream handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter('Log: %(message)s'))
    root_logger.addHandler(ch)

#%% Functions - deactivate_logging
def deactivate_logging():
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
        raise ValueError('Something bad happended when trying to close the logger.') # pragma: no cover

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_logs', exit=False)
    doctest.testmod(verbose=False)
