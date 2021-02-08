r"""
Test file for the `optimized` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import unittest

import dstauffman.numba as nub

try:
    import numba
    _HAVE_NUMBA = True
except ModuleNotFoundError:
    List = lambda x: x
    _HAVE_NUMBA = False

#%% types
@unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
class Test_types(unittest.TestCase):
    r"""
    Tests the following types:
        float64, int32
    """
    def test_types(self):
        self.assertIs(nub.float64, numba.float64)
        self.assertIs(nub.int32, numba.int32)

    def test_decorators(self):
        self.assertIs(nub.jit, numba.jit)
        self.assertIs(nub.njit, numba.njit)
        self.assertIs(nub.vectorize, numba.vectorize)

    def test_lists(self):
        from numba.typed import List
        self.assertIs(nub.List, List)

    def test_experimental(self):
        from numba.experimental import jitclass
        self.assertIs(nub.jitclass, jitclass)

    def test_callables(self):
        self.assertTrue(callable(nub.ncjit))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
