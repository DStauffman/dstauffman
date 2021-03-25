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
    assert numba
    _HAVE_NUMBA = True
except ModuleNotFoundError:
    _HAVE_NUMBA = False

#%% types
@unittest.skipIf(not _HAVE_NUMBA, 'Skipping due to missing numba dependency.')
class Test_types(unittest.TestCase):
    r"""
    Tests the passthrough options with the following cases:
        Callables
        Constants
    """
    def test_callables(self) -> None:
        self.assertTrue(callable(nub.ncjit))

    def test_constants(self) -> None:
        self.assertTrue(isinstance(nub.TARGET, str))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
