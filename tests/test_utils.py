# -*- coding: utf-8 -*-
r"""
Test file for the `utils` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#. Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import sys
import unittest
import dstauffman as dcs

#%% rms
class Test_rms(unittest.TestCase):
    r"""
    Tests the rms function with these cases:
        rms on just a scalar input
        normal rms on vector input
        rms on vector with axis specified
        rms on vector with bad axis specified
        rms on matrix without keeping dimensions, no axis
        rms on matrix without keeping dimensions, axis 0
        rms on matrix without keeping dimensions, axis 1
        rms on matrix with keeping dimensions
        rms on complex numbers
        rms on complex numbers that would return a real if done incorrectly
    """
    def setUp(self):
        self.inputs1   = np.array([0, 1, 0., -1])
        self.outputs1  = np.sqrt(2)/2
        self.inputs2   = [[0, 1, 0., -1], [1., 1, 1, 1]]
        self.outputs2a = np.sqrt(3)/2
        self.outputs2b = np.array([np.sqrt(2)/2, 1, np.sqrt(2)/2, 1])
        self.outputs2c = np.array([np.sqrt(2)/2, 1])
        self.outputs2d = np.matrix([[np.sqrt(2)/2], [1]])

    def test_scalar_input(self):
        out = dcs.rms(-1.5)
        self.assertEqual(out, 1.5)

    def test_rms_series(self):
        out = dcs.rms(self.inputs1)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1a(self):
        out = dcs.rms(self.inputs1, axis=0)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1b(self):
        with self.assertRaises(IndexError):
            dcs.rms(self.inputs1, axis=1)

    def test_axis_drop2a(self):
        out = dcs.rms(self.inputs2)
        self.assertAlmostEqual(out, self.outputs2a)

    def test_axis_drop2b(self):
        out = dcs.rms(self.inputs2, axis=0, keepdims=False)
        for (ix, val) in enumerate(out):
            self.assertAlmostEqual(val, self.outputs2b[ix])

    def test_axis_drop2c(self):
        out = dcs.rms(self.inputs2, axis=1, keepdims=False)
        for (ix, val) in enumerate(out):
            self.assertAlmostEqual(val, self.outputs2c[ix])

    def test_axis_keep(self):
        out = dcs.rms(self.inputs2, axis=1, keepdims=True)
        for i in range(0, len(out)):
            for j in range(0, len(out[i])):
                self.assertAlmostEqual(out[i, j], self.outputs2d[i, j])

    def test_complex_rms(self):
        out = dcs.rms(1.5j)
        self.assertEqual(out, np.complex(1.5, 0))

    def test_complex_conj(self):
        out = dcs.rms(np.array([1+1j, 1-1j]))
        self.assertAlmostEqual(out, np.sqrt(2))

#%% setup_dir
class Test_setup_dir(unittest.TestCase):
    r"""
    Tests the setup_dir function with these cases:
        create a new folder
        delete the contents of an existing folder
        fail to create a folder due to permissions
        fail to delete the contents of an existing folder due to permissions
        fail to create a folder due to a bad name
        delete the contents of an existing folder recursively
    """
    def test_create_folder(self):
        pass

    def test_clean_up_folder(self):
        pass

    def test_fail_to_create_folder(self):
        pass

    def test_fail_to_clean_folder(self):
        pass

    def test_bad_name_file_ext(self):
        pass

    def test_clean_up_recursively(self):
        pass

#%% compare_two_classes
# TODO:

#%% compare_two_dicts
class Test_compare_two_dicts(unittest.TestCase):
    r"""
    Tests the compare_two_dicts function with these cases:
        compares the same dicts
        compares different dicts
        compares same with names passed in
        compares with/without suppressed output
    """
    def setUp(self):
        self.d1 = {'a': 1, 'b': 2, 'c': 3}
        self.d2 = {'a': 1, 'b': 5, 'd': 6}
        self.names = ['Dict 1', 'Dict 2']

    def test_good_comparison(self):
        with dcs.capture_output() as (out, err):
            is_same = dcs.compare_two_dicts(self.d1, self.d1)
        output = out.getvalue().strip()
        self.assertEqual(output, '"d1" and "d2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self):
        with dcs.capture_output() as (out, err):
            is_same = dcs.compare_two_dicts(self.d1, self.d2)
        output = out.getvalue().strip()
        self.assertEqual(output, 'b is different.\nc is only in d1.\nd is only in d2.\n"d1" and "d2" are not the same.')
        self.assertFalse(is_same)

    def test_names(self):
        with dcs.capture_output() as (out, err):
            is_same = dcs.compare_two_dicts(self.d2, self.d2, names=self.names)
        output = out.getvalue().strip()
        self.assertEqual(output, '"Dict 1" and "Dict 2" are the same.')
        self.assertTrue(is_same)

    def test_suppression(self):
        with dcs.capture_output() as (out, err):
            is_same = dcs.compare_two_dicts(self.d1, self.d2, suppress_output=True, names=self.names)
        output = out.getvalue().strip()
        self.assertEqual(output, '')
        self.assertFalse(is_same)

    def test_printing(self):
        with dcs.capture_output() as (out, err):
            is_same = dcs.compare_two_dicts(self.d1, self.d1)
        output = out.getvalue().strip()
        self.assertEqual(output, '"d1" and "d2" are the same.')
        self.assertTrue(is_same)

#%% round_time

#%% read_text_file
class Test_read_text_file(unittest.TestCase):
    r"""
    Tests the reat_text_file function with these cases:
        read a file that exists
        read a file that does not exist (raise error)
    """
    @classmethod
    def setUpClass(self):
        self.folder   = dcs.get_tests_dir()
        self.contents = 'Hello, World!\n'
        self.filepath = os.path.join(self.folder, 'temp_file.txt')
        self.badpath  = r'AA:\non_existent_path\bad_file.txt'
        with open(self.filepath, 'wt') as file:
            file.write(self.contents)

    def test_reading(self):
        text = dcs.read_text_file(self.filepath)
        self.assertEqual(text, self.contents)

    def test_bad_reading(self):
        with dcs.capture_output() as (out, err):
            with self.assertRaises(OSError):
                dcs.read_text_file(self.badpath)
        output = out.getvalue().strip()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for reading.')

    @classmethod
    def tearDownClase(self):
        os.remove(self.filepath)

#%% write_text_file
class Test_write_text_file(unittest.TestCase):
    r"""
    Tests the write_text_file function with these cases:
        write a file
        write a bad file location (raise error)
    """
    @classmethod
    def setUpClass(self):
        self.folder   = dcs.get_tests_dir()
        self.contents = 'Hello, World!\n'
        self.filepath = os.path.join(self.folder, 'temp_file.txt')
        self.badpath  = r'AA:\non_existent_path\bad_file.txt'

    def test_writing(self):
        dcs.write_text_file(self.filepath, self.contents)
        with open(self.filepath, 'rt') as file:
            text = file.read()
        self.assertEqual(text, self.contents)

    def test_bad_writing(self):
        with dcs.capture_output() as (out, err):
            with self.assertRaises(OSError):
                dcs.write_text_file(self.badpath, self.contents)
        output = out.getvalue().strip()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for writing.')

    @classmethod
    def tearDownClass(self):
        os.remove(self.filepath)

#%% disp

#%% convert_annual_to_monthly_probability
class Test_convert_annual_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the convert_annual_to_monthly_probability function with these cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
    """
    def setUp(self):
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1, 12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self):
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        for i in range(0, len(self.monthly)):
            self.assertAlmostEqual(monthly[i], self.monthly[i])

    def test_scalar(self):
        monthly = dcs.convert_annual_to_monthly_probability(0)
        self.assertIn(monthly, self.monthly)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, 1.5]))

#%% get_root_dir
pass

#%% get_tests_dir

#%% get_data_dir

#%% capture_output
class Test_capture_output(unittest.TestCase):
    r"""Tests the capture_output function with these cases:
        capture standard output
        capture standard error
    """
    def test_std_out(self):
        with dcs.capture_output() as (out, err):
            print('Hello, World!')
        output = out.getvalue().strip()
        error  = err.getvalue().strip()
        self.assertEqual(output, 'Hello, World!')
        self.assertEqual(error, '')

    def test_std_err(self):
        with dcs.capture_output() as (out, err):
            print('Error Raised.', file=sys.stderr)
        output = out.getvalue().strip()
        error  = err.getvalue().strip()
        self.assertEqual(output, '')
        self.assertEqual(error, 'Error Raised.')


#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
