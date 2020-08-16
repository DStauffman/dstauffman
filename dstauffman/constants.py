r"""
Define constants for use in the rest of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import logging
import os
import unittest
import warnings

#%% Numpy settings
try:
    import numpy as np
except ModuleNotFoundError:
    warnings.warn('numpy was not imported, so a lot of capabilities will be limited.')
else:
    # Set NumPy error state for module
    np.seterr(invalid='ignore', divide='ignore')
    # Set NumPy printing options
    np.set_printoptions(threshold=1000)

#%% Logging configuration
from dstauffman.enums import LogLevel

#%% Register custom logging levels
logging.addLevelName(LogLevel.L0, 'L0')
logging.addLevelName(LogLevel.L1, 'L1')
logging.addLevelName(LogLevel.L2, 'L2')
logging.addLevelName(LogLevel.L3, 'L3')
logging.addLevelName(LogLevel.L4, 'L4')
logging.addLevelName(LogLevel.L5, 'L5')
logging.addLevelName(LogLevel.L6, 'L6')
logging.addLevelName(LogLevel.L7, 'L7')
logging.addLevelName(LogLevel.L8, 'L8')
logging.addLevelName(LogLevel.L9, 'L9')
logging.addLevelName(LogLevel.L10, 'L10')
logging.addLevelName(LogLevel.L11, 'L11')
logging.addLevelName(LogLevel.L12, 'L12')
logging.addLevelName(LogLevel.L20, 'L20')

#%% Configure default logging if not already set
logging.basicConfig(level=logging.WARNING)

#%% Constants
# A specified integer token value to use when you need one
INT_TOKEN = -1

# Whether we are currently on Windows or not
IS_WINDOWS = os.name == 'nt'

#%% Functions
# None

#%% Script
if __name__ == '__main__':
    # print all the constants (anything in ALL_CAPS)
    fields = [k for k in sorted(dir()) if k.isupper()]
    max_len = max(map(len, fields))
    for field in fields:
        pad = ' ' * (max_len - len(field))
        value = locals()[field]
        print('{}{} = {}'.format(field, pad, value))

    # run unittests
    unittest.main(module='dstauffman.tests.test_constants', exit=False)
