r"""
Test file for the `optimized` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np
    pi = np.pi
    inf = np.inf
else:
    from math import inf, pi
try:
    from numba.typed import List
except ModuleNotFoundError:
    List = lambda x: x

#%% np_any
class Test_np_any(unittest.TestCase):
    r"""
    Tests the np_any function with the following cases:
        All false
        Some true
    """
    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_false(self) -> None:
        x = np.zeros(1000, dtype=bool)
        self.assertFalse(dcs.np_any(x))

    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_true(self) -> None:
        x = np.zeros(1000, dtype=bool)
        x[333] = True
        self.assertTrue(dcs.np_any(x))

    def test_lists(self):
        x = List([False for i in range(1000)])
        self.assertFalse(dcs.np_any(x))
        x[333] = True
        self.assertTrue(dcs.np_any(x))

#%% np_all
class Test_np_all(unittest.TestCase):
    r"""
    Tests the np_all function with the following cases:
        All true
        Some false
    """
    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_true(self) -> None:
        x = np.ones(1000, dtype=bool)
        self.assertTrue(dcs.np_all(x))

    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_false(self) -> None:
        x = np.ones(1000, dtype=bool)
        x[333] = False
        self.assertFalse(dcs.np_all(x))

    def test_lists(self):
        x = List([True for i in range(1000)])
        self.assertTrue(dcs.np_all(x))
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
    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_sorted(self) -> None:
        x = np.array([1, 3, 3, 5, 7])
        self.assertTrue(dcs.issorted_opt(x))

    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_not_sorted(self) -> None:
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(dcs.issorted_opt(x))

    @unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
    def test_reverse_sorted(self) -> None:
        x = np.array([4, np.pi, 1., -1.])
        self.assertFalse(dcs.issorted_opt(x))
        self.assertTrue(dcs.issorted(x, descend=True))

    def test_lists(self) -> None:
        x = List([-inf, 0, 1, pi, 5, inf])
        self.assertTrue(dcs.issorted_opt(x))
        if dcs.HAVE_NUMPY:
            self.assertFalse(dcs.issorted(x, descend=True))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
