# -*- coding: utf-8 -*-
r"""
Test file for the `enums` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in July 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import numpy as np
import unittest
import dstauffman as dcs

#%% dist_enum_and_mons
class Test_dist_enum_and_mons(unittest.TestCase):
    r"""
    Tests the dist_enum_and_mons function with the following cases:
        Nominal usage
        All in one bin
    """
    def setUp(self):
        self.num = 100000
        self.distribution = 1./100*np.array([10, 20, 30, 40])
        self.max_months = np.array([1, 10, 50, 5])
        self.max_months = np.array([1, 1, 1, 1])
        self.prng = np.random.RandomState()
        self.per_lim = 0.01

    def test_calling(self):
        (state, mons) = dcs.dist_enum_and_mons(self.num, self.distribution, self.max_months, self.prng)
        breakout = np.histogram(state, bins = [0.5, 1.5, 2.5, 3.5, 4.5])[0]
        breakout_per = breakout / self.num
        for ix in range(len(self.distribution)):
            self.assertTrue(np.abs(breakout_per[ix] - self.distribution[ix]) <= self.per_lim)
        self.assertTrue(np.all(mons <= self.max_months[state-1]) and np.all(mons >= 1))

    def test_all_in_one_bin(self):
        for i in range(4):
            temp = np.zeros(4)
            temp[i] = 1
            (tb_state, _) = dcs.dist_enum_and_mons(self.num, temp, self.max_months, self.prng)
            self.assertTrue(np.all(tb_state == i+1))

    def test_alpha_and_beta(self):
        pass #TODO: write this

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
