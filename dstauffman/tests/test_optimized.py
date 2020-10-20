r"""
Test file for the `optimized` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import unittest

from numba.typed import List
import numpy as np

import dstauffman as dcs

#%% np_any
class Test_np_any(unittest.TestCase):
    r"""
    Tests the np_any function with the following cases:
        All false
        Some true
    """
    def test_false(self) -> None:
        x = np.zeros(1000, dtype=bool)
        self.assertFalse(dcs.np_any(x))

    def test_true(self) -> None:
        x = np.zeros(1000, dtype=bool)
        x[333] = True
        self.assertTrue(dcs.np_any(x))

#%% np_all
class Test_np_all(unittest.TestCase):
    r"""
    Tests the np_all function with the following cases:
        All true
        Some false
    """
    def test_true(self) -> None:
        x = np.ones(1000, dtype=bool)
        self.assertTrue(dcs.np_all(x))

    def test_false(self) -> None:
        x = np.ones(1000, dtype=bool)
        x[333] = False
        self.assertFalse(dcs.np_all(x))

#%% issorted_opt
class Test_issorted_opt(unittest.TestCase):
    r"""
    Tests the issorted_opt function with the following cases:
        Sorted
        Not sorted
        Reverse sorted (x2)
        Lists
    """
    def test_sorted(self) -> None:
        x = np.array([1, 3, 3, 5, 7])
        self.assertTrue(dcs.issorted_opt(x))

    def test_not_sorted(self) -> None:
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(dcs.issorted_opt(x))

    def test_reverse_sorted(self) -> None:
        x = np.array([4, np.pi, 1., -1.])
        self.assertFalse(dcs.issorted_opt(x))
        self.assertTrue(dcs.issorted(x, descend=True))

    def test_lists(self) -> None:
        x = List([-np.inf, 0, 1, np.pi, 5, np.inf])
        self.assertTrue(dcs.issorted_opt(x))
        self.assertFalse(dcs.issorted(x, descend=True))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
