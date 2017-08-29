# -*- coding: utf-8 -*-
r"""
Define constants for use in the rest of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.

"""

#%% Imports
import unittest

#%% Constants
MONTHS_PER_YEAR  = 12
INT_TOKEN        = -1
DEFAULT_COLORMAP = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'
QUAT_SIZE        = 4

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
    unittest.main(module='tests.test_constants', exit=False)
