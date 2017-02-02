# -*- coding: utf-8 -*-
r"""
Constants module file for the "dstauffman" library.  It defines constants.

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

#%% Unit Test
def _main():
    r"""Unit test function."""
    print('MONTHS_PER_YEAR      = {}'.format(MONTHS_PER_YEAR))
    print('INT_TOKEN            = {}'.format(INT_TOKEN))
    print('DEFAULT_COLORMAP     = {}'.format(DEFAULT_COLORMAP))
    print('QUAT_SIZE            = {}'.format(QUAT_SIZE))

if __name__ == '__main__':
    unittest.main(module='tests.test_constants', exit=False)
    _main()
