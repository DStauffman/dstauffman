r"""
Test file for the `optimized` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import unittest

import numpy as np

import dstauffman as dcs

#%% np_any
class Test_np_any(unittest.TestCase):
    r"""
    Tests the np_any function with these cases:
        All false
        Some true
    """
    def test_false(self):
        x = np.zeros(1000, dtype=bool)
        self.assertFalse(dcs.np_any(x))

    def test_true(self):
        x = np.zeros(1000, dtype=bool)
        x[333] = True
        self.assertTrue(dcs.np_any(x))

#%% np_all
class Test_np_all(unittest.TestCase):
    r"""
    Tests the np_all function with these cases:
        All true
        Some false
    """
    def test_true(self):
        x = np.ones(1000, dtype=bool)
        self.assertTrue(dcs.np_all(x))

    def test_false(self):
        x = np.ones(1000, dtype=bool)
        x[333] = False
        self.assertFalse(dcs.np_all(x))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
