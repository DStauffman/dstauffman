# -*- coding: utf-8 -*-
r"""
Test file for the `matlab` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import os
import unittest

import numpy as np

import dstauffman as dcs

#%% load_matlab
class Test_load_matlab(unittest.TestCase):
    r"""
    Tests the load_matlab function with the following cases:
        Nominal
    """
    def setUp(self):
        self.filename = os.path.join(dcs.get_tests_dir(), 'test_numbers.mat')
        self.row_nums = np.array([[1., 2.2, 3.]]).T
        self.col_nums = np.array([[1.], [2.], [3.], [4.], [5.]]).T
        # TODO: figure out why the data is transposed and potentially reverse it?

    def test_nominal(self):
        out = dcs.load_matlab(self.filename)
        self.assertEqual(set(out.keys()), {'col_nums', 'row_nums'})
        np.testing.assert_array_equal(out['row_nums'], self.row_nums)
        np.testing.assert_array_equal(out['col_nums'], self.col_nums)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
