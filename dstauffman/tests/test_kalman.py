# -*- coding: utf-8 -*-
r"""
Test file for the `plotting` module module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import unittest

import dstauffman as dcs

#%% KfInnov
class Test_KfInnov(unittest.TestCase):
    r"""
    Tests the KfInnov class with the following cases:
        TBD
    """
    def test_nominal(self):
        innov = dcs.KfInnov()
        self.assertTrue(isinstance(innov, dcs.KfInnov)) # TODO: test better

#%% KfOut
class Test_KfOut(unittest.TestCase):
    r"""
    Tests the KfOut class with the following cases:
        TBD
    """
    def test_nominal(self):
        kf = dcs.Kf()
        self.assertTrue(isinstance(kf, dcs.Kf)) # TODO: test better

#%% plot_attitude

#%% plot_position

#%% plot_innovation

#%% plot_covariance

#%% plot_states

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
