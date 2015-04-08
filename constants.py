# -*- coding: utf-8 -*-
r"""
Constants module file for the "dstauffman" library.  It defines constants.

Notes
-----
#. Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import unittest

#%% Constants
MONTHS_PER_YEAR = 12
INT_TOKEN       = -1

#%% Functions
# None

#%% Unit Test
def _main():
    r"""Unit test function."""
    print('MONTHS_PER_YEAR    = {}'.format(MONTHS_PER_YEAR))
    print('INT_TOKEN          = {}'.format(INT_TOKEN))

if __name__ == '__main__':
    unittest.main(module='tests.test_constants', exit=False)
    _main()
