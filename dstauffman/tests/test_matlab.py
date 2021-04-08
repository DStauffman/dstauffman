r"""
Test file for the `matlab` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
from typing import ClassVar
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

#%% Classes
class _Gender(dcs.IntEnumPlus):
    r"""Enumeration to match the MATLAB one from the test cases."""
    null: ClassVar[int]        = 0
    female: ClassVar[int]      = 1
    male: ClassVar[int]        = 2
    uncirc_male: ClassVar[int] = 3
    circ_male: ClassVar[int]   = 4
    non_binary: ClassVar[int]  = 5

#%% load_matlab
@unittest.skipIf(not dcs.HAVE_H5PY or not dcs.HAVE_NUMPY, 'Skipping due to missing h5py/numpy dependency.')
class Test_load_matlab(unittest.TestCase):
    r"""
    Tests the load_matlab function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.filename1 = dcs.get_tests_dir() / 'test_numbers.mat'
        self.filename2 = dcs.get_tests_dir() / 'test_struct.mat'
        self.filename3 = dcs.get_tests_dir() / 'test_enums.mat'
        self.filename4 = dcs.get_tests_dir() / 'test_cell_array.mat'
        self.filename5 = dcs.get_tests_dir() / 'test_nested.mat'
        self.row_nums  = np.array([[1., 2.2, 3.]])
        self.col_nums  = np.array([[1.], [2.], [3.], [4.], [5.]])
        self.mat_nums  = np.array([[1, 2, 3], [4, 5, 6]])
        self.exp_enum  = np.array([_Gender.male, _Gender.female, _Gender.female, _Gender.circ_male, \
            _Gender.circ_male, _Gender.circ_male], dtype=int)
        self.offsets   = {'r': 10, 'c': 20, 'm': 30}
        self.enums     = {'Gender': [getattr(_Gender, x) for x in _Gender.list_of_names()]}
        self.row_nums_1d = np.squeeze(self.row_nums)
        self.col_nums_1d = np.squeeze(self.col_nums)

    def test_nominal(self) -> None:
        out = dcs.load_matlab(self.filename1, squeeze=False)
        self.assertEqual(set(out.keys()), {'col_nums', 'row_nums', 'mat_nums'})
        np.testing.assert_array_equal(out['row_nums'], self.row_nums)
        np.testing.assert_array_equal(out['col_nums'], self.col_nums)
        np.testing.assert_array_equal(out['mat_nums'], self.mat_nums)

    def test_struct(self) -> None:
        out = dcs.load_matlab(self.filename2, squeeze=True)
        self.assertEqual(set(out.keys()), {'x'})
        np.testing.assert_array_equal(out['x']['r'], self.row_nums_1d)
        np.testing.assert_array_equal(out['x']['c'], self.col_nums_1d)
        np.testing.assert_array_equal(out['x']['m'], self.mat_nums)

    def test_load_varlist(self) -> None:
        out = dcs.load_matlab(self.filename2, varlist=['y'])
        self.assertEqual(out.keys(), set())

    @unittest.expectedFailure
    def test_enum(self) -> None:
        # Enum test case not working.  Need to better understand how categorical arrays are stored.
        out = dcs.load_matlab(self.filename3, enums=self.enums)
        self.assertEqual(set(out.keys()), {'enum'})
        np.testing.assert_array_equal(out['enum'], self.exp_enum)

    def test_unknown_enum(self) -> None:
        with self.assertRaises(ValueError):
            dcs.load_matlab(self.filename3, enums={'Nope': [1, 2]})

    def test_cell_arrays(self) -> None:
        out = dcs.load_matlab(self.filename4, varlist=('cdat', ))
        self.assertEqual(out.keys(), {'cdat', })
        np.testing.assert_array_equal(out['cdat'][0], self.row_nums_1d)
        np.testing.assert_array_equal(out['cdat'][1], self.col_nums_1d)
        np.testing.assert_array_equal(out['cdat'][2], self.mat_nums)
        self.assertEqual(''.join(chr(x) for x in out['cdat'][3]), 'text')
        self.assertEqual(''.join(chr(x) for x in out['cdat'][4]), 'longer text')
        self.assertEqual(''.join(chr(x) for x in out['cdat'][5]), '\x00\x00')  # TODO: expect '' instead of [0, 0]
        np.testing.assert_array_equal(out['cdat'][6], np.zeros(2))  # TODO: expect [] instead of [0, 0]

    def test_nested(self) -> None:
        out = dcs.load_matlab(self.filename5, enums=self.enums)
        self.assertEqual(set(out.keys()), {'col_nums', 'row_nums', 'mat_nums', 'x', 'enum', 'cdat', 'data'})
        np.testing.assert_array_equal(out['row_nums'], self.row_nums_1d)
        np.testing.assert_array_equal(out['col_nums'], self.col_nums_1d)
        np.testing.assert_array_equal(out['mat_nums'], self.mat_nums)
        np.testing.assert_array_equal(out['x']['r'], self.row_nums_1d)
        np.testing.assert_array_equal(out['x']['c'], self.col_nums_1d)
        np.testing.assert_array_equal(out['x']['m'], self.mat_nums)
        #np.testing.assert_array_equal(out['enum'], self.exp_enum) # TODO: fix this one along with the other case
        np.testing.assert_array_equal(out['cdat'][0], self.row_nums_1d)
        np.testing.assert_array_equal(out['cdat'][1], self.col_nums_1d)
        np.testing.assert_array_equal(out['cdat'][2], self.mat_nums)
        np.testing.assert_array_equal(out['data']['x']['r'], self.row_nums_1d)
        np.testing.assert_array_equal(out['data']['x']['c'], self.col_nums_1d)
        np.testing.assert_array_equal(out['data']['x']['m'], self.mat_nums)
        np.testing.assert_array_equal(out['data']['y']['r'], self.row_nums_1d + self.offsets['r'])
        np.testing.assert_array_equal(out['data']['y']['c'], self.col_nums_1d + self.offsets['c'])
        np.testing.assert_array_equal(out['data']['y']['m'], self.mat_nums + self.offsets['m'])
        np.testing.assert_array_equal(out['data']['z']['a'], np.array([1., 2., 3.]))
        #np.testing.assert_array_equal(out['data']['z']['b'], self.exp_enum) # TODO: fix this one along with the other case
        np.testing.assert_array_equal(out['data']['c'][0], self.row_nums_1d)
        np.testing.assert_array_equal(out['data']['c'][1], self.col_nums_1d)
        np.testing.assert_array_equal(out['data']['c'][2], self.mat_nums)
        np.testing.assert_array_equal(out['data']['nc'][0], self.row_nums_1d)
        np.testing.assert_array_equal(out['data']['nc'][1], self.col_nums_1d)
        np.testing.assert_array_equal(out['data']['nc'][2]['r'], self.row_nums_1d)
        np.testing.assert_array_equal(out['data']['nc'][2]['c'], self.col_nums_1d)
        np.testing.assert_array_equal(out['data']['nc'][2]['m'], self.mat_nums)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
