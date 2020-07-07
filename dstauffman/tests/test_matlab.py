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
        self.filename  = os.path.join(dcs.get_tests_dir(), 'test_numbers.mat')
        self.filename2 = os.path.join(dcs.get_tests_dir(), 'test_struct.mat')
        self.row_nums  = np.array([[1., 2.2, 3.]])
        self.col_nums  = np.array([[1.], [2.], [3.], [4.], [5.]])
        self.mat_nums  = np.array([[1, 2, 3], [4, 5, 6]])

    def test_nominal(self):
        out = dcs.load_matlab(self.filename, squeeze=False)
        self.assertEqual(set(out.keys()), {'col_nums', 'row_nums', 'mat_nums'})
        np.testing.assert_array_equal(out['row_nums'], self.row_nums)
        np.testing.assert_array_equal(out['col_nums'], self.col_nums)
        np.testing.assert_array_equal(out['mat_nums'], self.mat_nums)

    def test_struct(self):
        out = dcs.load_matlab(self.filename2, squeeze=True)
        self.assertEqual(set(out.keys()), {'x'})
        np.testing.assert_array_equal(out['x']['r'], np.squeeze(self.row_nums))
        np.testing.assert_array_equal(out['x']['c'], np.squeeze(self.col_nums))
        np.testing.assert_array_equal(out['x']['m'], self.mat_nums)

    def test_load_varlist(self):
        out = dcs.load_matlab(self.filename2, varlist=['y'])
        self.assertEqual(out.keys(), set())

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
