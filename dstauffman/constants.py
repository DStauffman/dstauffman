# -*- coding: utf-8 -*-
r"""
Define constants for use in the rest of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.

"""

#%% Imports
import os
import unittest

#%% Constants
# Default colormap to use on certain plots
DEFAULT_COLORMAP = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'

# A specified integer token value to use when you need one
INT_TOKEN = -1

# Whether we are currently on Windows or not
IS_WINDOWS = os.name == 'nt'

# Number of months in a year
MONTHS_PER_YEAR = 12

# Number of elements that should be in a quaternion
QUAT_SIZE = 4

# Whether to include a classification on any generated plots
DEFAULT_CLASSIFICATION = ''

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
