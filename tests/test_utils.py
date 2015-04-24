# -*- coding: utf-8 -*-
r"""
Test file for the `utils` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import copy
from datetime import datetime
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
        create a new nested folder
        delete the contents of an existing folder
        fail to create a folder due to permissions
        fail to delete the contents of an existing folder due to permissions
        fail to create a folder due to a bad name
        delete the contents of an existing folder recursively
    """
    def setUp(self):
        self.folder   = os.path.join(dcs.get_tests_dir(), 'temp_dir')
        self.subdir   = os.path.join(dcs.get_tests_dir(), 'temp_dir', 'temp_dir2')
        self.filename = os.path.join(self.folder, 'temp_file.txt')
        self.subfile  = os.path.join(self.subdir, 'temp_file.txt')
        self.text     = 'Hello, World!\n'

    def test_create_folder(self):
        with dcs.capture_output() as (out, _):
            dcs.setup_dir(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Created directory: "{}"'.format(self.folder))

    def test_nested_folder(self):
        with dcs.capture_output() as (out, _):
            dcs.setup_dir(self.subdir)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Created directory: "{}"'.format(self.subdir))

    def test_clean_up_folder(self):
        with dcs.capture_output():
            dcs.setup_dir(self.folder)
            dcs.write_text_file(self.filename, self.text)
        with dcs.capture_output() as (out, _):
            dcs.setup_dir(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Files/Sub-folders were removed from: "{}"'.format(self.folder))

    def test_clean_up_partial(self):
        with dcs.capture_output():
            dcs.setup_dir(self.folder)
            dcs.write_text_file(self.filename, '')
            dcs.setup_dir(self.subdir)
            dcs.write_text_file(self.subfile, '')
        with dcs.capture_output() as (out, _):
            dcs.setup_dir(self.folder, rec=False)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Files/Sub-folders were removed from: "{}"'.format(self.folder))

    def test_fail_to_create_folder(self):
        pass #TODO: write this test

    def test_fail_to_clean_folder(self):
        pass #TODO: write this test

    def test_bad_name_file_ext(self):
        pass #TODO: write this test

    def test_clean_up_recursively(self):
        with dcs.capture_output():
            dcs.setup_dir(self.subdir)
            dcs.write_text_file(self.subfile, self.text)
        with dcs.capture_output() as (out, _):
            dcs.setup_dir(self.folder, rec=True)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Files/Sub-folders were removed from: "{}"\n'.format(self.subdir) + \
            'Files/Sub-folders were removed from: "{}"'.format(self.folder))

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        if os.path.isfile(self.subfile):
            os.remove(self.subfile)
        if os.path.isdir(self.subdir):
            os.rmdir(self.subdir)
        if os.path.isdir(self.folder):
            os.rmdir(self.folder)

#%% compare_two_classes
class Test_compare_two_classes(unittest.TestCase):
    r"""
    Tests the compare_two_classes function with these cases:
        compares the same classes
        compares different classes
        compares same with names passed in
        compares with suppressed output
        compare subclasses
    """
    def setUp(self):
        self.c1 = type('Class1', (object, ), {'a': 0, 'b' : '[1, 2, 3]', 'c': 'text'})
        self.c2 = type('Class2', (object, ), {'a': 0, 'b' : '[1, 2, 4]', 'd': 'text'})
        self.names = ['Class 1', 'Class 2']

    def test_is_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c1, self.c1)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_good_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c1, copy.copy(self.c1))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different.\nc is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

    def test_names(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c2, self.c2, names=self.names)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"Class 1" and "Class 2" are the same.')
        self.assertTrue(is_same)

    def test_suppression(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c1, self.c2, suppress_output=True, names=self.names)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')
        self.assertFalse(is_same)

    def test_subclasses(self):
        temp1 = copy.copy(self.c1)
        temp2 = copy.copy(self.c2)
        temp1.e = copy.copy(self.c1)
        temp2.e = copy.copy(self.c2)
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(temp1, temp2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different.\ne is different.\nc is only in c1.\n' + \
            'd is only in c2.\n"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

#%% compare_two_dicts
class Test_compare_two_dicts(unittest.TestCase):
    r"""
    Tests the compare_two_dicts function with these cases:
        compares the same dicts
        compares different dicts
        compares same with names passed in
        compares with suppressed output
    """
    def setUp(self):
        self.d1 = {'a': 1, 'b': 2, 'c': 3}
        self.d2 = {'a': 1, 'b': 5, 'd': 6}
        self.names = ['Dict 1', 'Dict 2']

    def test_good_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_dicts(self.d1, self.d1)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"d1" and "d2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_dicts(self.d1, self.d2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different.\nc is only in d1.\nd is only in d2.\n"d1" and "d2" are not the same.')
        self.assertFalse(is_same)

    def test_names(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_dicts(self.d2, self.d2, names=self.names)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"Dict 1" and "Dict 2" are the same.')
        self.assertTrue(is_same)

    def test_suppression(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_dicts(self.d1, self.d2, suppress_output=True, names=self.names)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')
        self.assertFalse(is_same)

#%% round_time
class Test_round_time(unittest.TestCase):
    r"""
    Tests the round_time function with these cases:
        normal use (round to one minute)
        extended use (round to a different specified time)
        get current time
    """
    def test_normal_use(self):
        rounded_time = dcs.round_time(datetime(2015, 3, 13, 8, 4, 10))
        self.assertEqual(rounded_time, datetime(2015, 3, 13, 8, 4, 0))

    def test_extended_use(self):
        rounded_time = dcs.round_time(datetime(2015, 3, 13, 8, 4, 10), round_to_sec=300)
        self.assertEqual(rounded_time, datetime(2015, 3, 13, 8, 5, 0))

    def test_current_time(self):
        dcs.round_time()
        self.assertTrue(True)

#%% read_text_file
class Test_read_text_file(unittest.TestCase):
    r"""
    Tests the reat_text_file function with these cases:
        read a file that exists
        read a file that does not exist (raise error)
    """
    @classmethod
    def setUpClass(cls):
        cls.folder   = dcs.get_tests_dir()
        cls.contents = 'Hello, World!\n'
        cls.filepath = os.path.join(cls.folder, 'temp_file.txt')
        cls.badpath  = r'AA:\non_existent_path\bad_file.txt'
        with open(cls.filepath, 'wt') as file:
            file.write(cls.contents)

    def test_reading(self):
        text = dcs.read_text_file(self.filepath)
        self.assertEqual(text, self.contents)

    def test_bad_reading(self):
        with dcs.capture_output() as (out, _):
            count = 0
            try:
                dcs.read_text_file(self.badpath)
            except OSError:
                count += 1
            except IOError:
                count += 1
            else:
                raise
            # must raise at least one of the two errors
            self.assertTrue(count > 0)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for reading.')

    @classmethod
    def tearDownClase(cls):
        os.remove(cls.filepath)

#%% write_text_file
class Test_write_text_file(unittest.TestCase):
    r"""
    Tests the write_text_file function with these cases:
        write a file
        write a bad file location (raise error)
    """
    @classmethod
    def setUpClass(cls):
        cls.folder   = dcs.get_tests_dir()
        cls.contents = 'Hello, World!\n'
        cls.filepath = os.path.join(cls.folder, 'temp_file.txt')
        cls.badpath  = r'AA:\non_existent_path\bad_file.txt'

    def test_writing(self):
        dcs.write_text_file(self.filepath, self.contents)
        with open(self.filepath, 'rt') as file:
            text = file.read()
        self.assertEqual(text, self.contents)

    def test_bad_writing(self):
        with dcs.capture_output() as (out, _):
            count = 0
            try:
                dcs.write_text_file(self.badpath, self.contents)
            except OSError:
                count += 1
            except IOError:
                count += 1
            else:
                raise
            # must raise at least one of the two errors
            self.assertTrue(count > 0)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for writing.')

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.filepath)

#%% disp
class Test_disp(unittest.TestCase):
    r"""
    Tests the disp function with these cases:
        TBD
    """
    def setUp(self):
        self.a      = type('a', (object, ), {'b': 0, 'c' : '[1, 2, 3]', 'd': 'text'})
        self.output = 'b : 0\nc : [1, 2, 3]\nd : text'

    def test_normal(self):
        with dcs.capture_output() as (out, _):
            text = dcs.disp(self.a)
        output = out.getvalue().strip()
        self.assertEqual(output, self.output)
        self.assertEqual(output, text)

    def test_padding(self):
        pass #TODO: write this

    def test_level(self):
        pass #TODO: write this

    def test_suppressed_output(self):
        pass #TODO: write this

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
class Test_get_root_dir(unittest.TestCase):
    r"""Tests the get_root_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_root_dir()
        self.assertTrue(folder) # TODO: don't know an independent way to test this

#%% get_tests_dir
class Test_get_tests_dir(unittest.TestCase):
    r"""Tests the get_tests_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_tests_dir()
        self.assertEqual(folder, os.path.join(dcs.get_root_dir(), 'tests'))

#%% get_data_dir
class Test_get_data_dir(unittest.TestCase):
    r"""Tests the get_data_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_data_dir()
        self.assertEqual(folder, os.path.join(dcs.get_root_dir(), 'data'))

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
        out.close()
        err.close()
        self.assertEqual(output, 'Hello, World!')
        self.assertEqual(error, '')

    def test_std_err(self):
        with dcs.capture_output() as (out, err):
            print('Error Raised.', file=sys.stderr)
        output = out.getvalue().strip()
        error  = err.getvalue().strip()
        out.close()
        err.close()
        self.assertEqual(output, '')
        self.assertEqual(error, 'Error Raised.')


#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
