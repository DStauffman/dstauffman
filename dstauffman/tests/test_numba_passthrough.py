r"""
Test file for the `optimized` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import unittest

import dstauffman.numba as dcsnb

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
        self.assertIs(dcsnb.float64, numba.float64)
        self.assertIs(dcsnb.int32, numba.int32)

    def test_decorators(self):
        self.assertIs(dcsnb.jit, numba.jit)
        self.assertIs(dcsnb.njit, numba.njit)
        self.assertIs(dcsnb.vectorize, numba.vectorize)

    def test_lists(self):
        from numba.typed import List
        self.assertIs(dcsnb.List, List)

    def test_experimental(self):
        from numba.experimental import jitclass
        self.assertIs(dcsnb.jitclass, jitclass)

    def test_callables(self):
        self.assertTrue(callable(dcsnb.ncjit))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
