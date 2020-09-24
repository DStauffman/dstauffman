r"""
Test file for the `matlab` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import os
from typing import ClassVar
import unittest

import numpy as np

import dstauffman as dcs

#%% Classes
class _Gender(dcs.IntEnumPlus):
    r"""Enumeration to match the MATLAB one from the test cases."""
    null: ClassVar[int]   = 0
    female: ClassVar[int] = 1
    male: ClassVar[int]   = 2

#%% load_matlab
class Test_load_matlab(unittest.TestCase):
    r"""
    Tests the load_matlab function with the following cases:
        Nominal
    """
    def setUp(self):
        self.filename1 = os.path.join(dcs.get_tests_dir(), 'test_numbers.mat')
        self.filename2 = os.path.join(dcs.get_tests_dir(), 'test_struct.mat')
        self.filename3 = os.path.join(dcs.get_tests_dir(), 'test_enums.mat')
        self.filename4 = os.path.join(dcs.get_tests_dir(), 'test_nested.mat')
        self.row_nums  = np.array([[1., 2.2, 3.]])
        self.col_nums  = np.array([[1.], [2.], [3.], [4.], [5.]])
        self.mat_nums  = np.array([[1, 2, 3], [4, 5, 6]])
        self.exp_enum  = np.array([_Gender.male, _Gender.female, _Gender.female], dtype=int)
        self.offsets   = {'r': 10, 'c': 20, 'm': 30}
        self.enums     = {'Gender': [getattr(_Gender, x) for x in sorted(_Gender.list_of_names())]}

    def test_nominal(self):
        out = dcs.load_matlab(self.filename1, squeeze=False)
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

    @unittest.skip('Enum test case not working.')
    def test_enum(self):
        out = dcs.load_matlab(self.filename3, enums=self.enums)
        self.assertEqual(set(out.keys()), {'enum'})
        np.testing.assert_array_equal(out['enum'], self.exp_enum)

    def test_unknown_enum(self):
        with self.assertRaises(ValueError):
            dcs.load_matlab(self.filename3, enums={'Nope': [1, 2]})

    def test_nested(self):
        out = dcs.load_matlab(self.filename4, enums=self.enums)
        self.assertEqual(set(out.keys()), {'col_nums', 'row_nums', 'mat_nums', 'x', 'enum', 'data'})
        np.testing.assert_array_equal(out['row_nums'], np.squeeze(self.row_nums))
        np.testing.assert_array_equal(out['col_nums'], np.squeeze(self.col_nums))
        np.testing.assert_array_equal(out['mat_nums'], self.mat_nums)
        np.testing.assert_array_equal(out['x']['r'], np.squeeze(self.row_nums))
        np.testing.assert_array_equal(out['x']['c'], np.squeeze(self.col_nums))
        np.testing.assert_array_equal(out['x']['m'], self.mat_nums)
        np.testing.assert_array_equal(out['data']['x']['r'], np.squeeze(self.row_nums))
        np.testing.assert_array_equal(out['data']['x']['c'], np.squeeze(self.col_nums))
        np.testing.assert_array_equal(out['data']['x']['m'], self.mat_nums)
        np.testing.assert_array_equal(out['data']['y']['r'], np.squeeze(self.row_nums) + self.offsets['r'])
        np.testing.assert_array_equal(out['data']['y']['c'], np.squeeze(self.col_nums) + self.offsets['c'])
        np.testing.assert_array_equal(out['data']['y']['m'], self.mat_nums + self.offsets['m'])
        #np.testing.assert_array_equal(out['enum'], self.enum) # TODO: fix this one along with the other case

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
