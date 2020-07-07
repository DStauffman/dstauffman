r"""
Test file for the `classes` module module of the "dstauffman.spacecraft" library.  It is intented
to contain test cases to demonstrate functionaliy and correct outcomes for all the functions within
the module.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import unittest

import dstauffman.spacecraft as space

#%% KfInnov
class Test_KfInnov(unittest.TestCase):
    r"""
    Tests the KfInnov class with the following cases:
        TBD
    """
    def test_nominal(self):
        innov = space.KfInnov()
        self.assertTrue(isinstance(innov, space.KfInnov)) # TODO: test better

#%% KfOut
class Test_KfOut(unittest.TestCase):
    r"""
    Tests the KfOut class with the following cases:
        TBD
    """
    def test_nominal(self):
        kf = space.Kf()
        self.assertTrue(isinstance(kf, space.Kf)) # TODO: test better

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
