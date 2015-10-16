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
import time
import unittest
import dstauffman as dcs

#%% _nan_equal
class Test__nan_equal(unittest.TestCase):
    r"""
    Tests the local _nan_equal function with these cases:
        TBD
    """
    def setUp(self):
        self.a = np.array([1, 2, np.nan])
        self.b = np.array([1, 2, np.nan])
        self.c = np.array([3, 2, np.nan])

    def test_equal(self):
        self.assertTrue(dcs.utils._nan_equal(self.a, self.b))

    def test_not_equal(self):
        self.assertFalse(dcs.utils._nan_equal(self.a, self.c))

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
        self.inputs3   = np.hstack((self.inputs1, np.nan))
        self.inputs4   = [[0, 0., np.nan], [1., np.nan, 1]]
        self.outputs4a = np.sqrt(2)/2
        self.outputs4b = np.array([np.sqrt(2)/2, 0, 1])
        self.outputs4c = np.array([0, 1])

    def test_scalar_input(self):
        out = dcs.rms(-1.5)
        self.assertEqual(out, 1.5)

    def test_empty(self):
        out = dcs.rms([])
        self.assertTrue(np.isnan(out))

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

    def test_with_nans(self):
        out = dcs.rms(self.inputs3, ignore_nans=False)
        self.assertTrue(np.isnan(out))

    def test_ignore_nans1(self):
        out = dcs.rms(self.inputs3, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs1)

    def test_ignore_nans2(self):
        out = dcs.rms(self.inputs4, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs4a)

    def test_ignore_nans3(self):
        out = dcs.rms(self.inputs4, ignore_nans=True, axis=0)
        np.testing.assert_array_almost_equal(out, self.outputs4b)

    def test_ignore_nans4(self):
        out = dcs.rms(self.inputs4, ignore_nans=True, axis=1)
        np.testing.assert_array_almost_equal(out, self.outputs4c)

    def test_all_nans(self):
        out = dcs.rms(np.array([np.nan, np.nan]), ignore_nans=True)
        self.assertTrue(np.isnan(out))

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
        def _clean(self):
            if os.path.isfile(self.filename):
                os.remove(self.filename)
            if os.path.isfile(self.subfile):
                os.remove(self.subfile)
            if os.path.isdir(self.subdir):
                os.rmdir(self.subdir)
            if os.path.isdir(self.folder):
                os.rmdir(self.folder)
        try:
            _clean(self)
        except PermissionError:
            # pause to let Windows catch up and close files
            time.sleep(1)
            # retry
            _clean(self)
        except OSError:
            # pause to let Windows catch up and close files
            time.sleep(1)
            # retry
            _clean(self)

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
        self.c3 = type('Class3', (object, ), {'a': 0, 'b' : '[1, 2, 3]', 'c': 'text', 'e': self.c1})
        self.c4 = type('Class4', (object, ), {'a': 0, 'b' : '[1, 2, 4]', 'd': 'text', 'e': self.c2})

    def test_is_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c1, self.c1)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_good_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c1, copy.deepcopy(self.c1))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different from c1 to c2.\nc is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')
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

    def test_subclasses_match(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c3, self.c3, ignore_callables=False)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_subclasses_recurse(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_same)
        self.assertEqual(output, 'b is different from c1 to c2.\nb is different from c1.e to c2.e.\n' + \
            'c is only in c1.e.\nd is only in c2.e.\n"c1.e" and "c2.e" are not the same.\n' + \
            'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')

    def test_subclasses_norecurse(self):
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False, compare_recursively=False)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different from c1 to c2.\n' + \
            'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

    def test_mismatched_subclasses(self):
        self.c4.e = 5
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_same)
        self.assertEqual(output, 'b is different from c1 to c2.\ne is different from c1 to c2.\n' + \
            'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')
        is_same = dcs.compare_two_classes(self.c4, self.c3, ignore_callables=False, suppress_output=True)
        self.assertFalse(is_same)

    def test_callables(self):
        def f(x): return x
        def g(x): return x
        self.c3.e = f
        self.c4.e = g
        self.c4.b = self.c3.b
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c4, self.c3, ignore_callables=False)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_same)
        self.assertEqual(output, 'e is different from c1 to c2.\nc is only in c2.\nd is only in c1.\n' + \
            '"c1" and "c2" are not the same.')

    def test_ignore_callables(self):
        def f(x): return x
        def g(x): return x
        self.c3.e = f
        self.c4.e = g
        self.c4.b = self.c3.b
        with dcs.capture_output() as (out, _):
            is_same = dcs.compare_two_classes(self.c4, self.c3, ignore_callables=True)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_same)
        self.assertEqual(output, 'c is only in c2.\nd is only in c1.\n"c1" and "c2" are not the same.')

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

#%% make_python_init
class Test_make_python_init(unittest.TestCase):
    r"""
    Tests the make_python_init function with these cases:
        TBD
    """
    def setUp(self):
        self.folder   = dcs.get_root_dir()
        self.text     = 'from .classes import'
        self.folder2  = dcs.get_tests_dir()
        self.filepath = os.path.join(self.folder2, 'temp_file.py')

    def test_nominal_use(self):
        text = dcs.make_python_init(self.folder)
        self.assertEqual(text[0:20], self.text)

    def test_duplicated_funcs(self):
        with open(self.filepath, 'wt') as file:
            file.write('def Test_Frozen():\n    pass\n')
        with dcs.capture_output() as (out, _):
            text = dcs.make_python_init(self.folder2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(text[0:34], 'from .temp_file import Test_Frozen')
        self.assertTrue(output.startswith('Uniqueness Problem'))

    def tearDown(self):
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)

#%% get_python_definitions
class Test_get_python_definitions(unittest.TestCase):
    r"""
    Tests the get_python_definitions function with these cases:
        TBD
    """
    def setUp(self):
        self.text = 'def a():\n    pass\ndef _b():\n    pass\n'
        self.funcs = ['a']

    def nominal_usage(self):
        funcs = dcs.get_python_definitions(self.text)
        self.assertEqual(funcs, self.funcs)

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
            try:
                dcs.read_text_file(self.badpath)
            except:
                self.assertTrue(sys.exc_info()[0] in [OSError, IOError])
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for reading.')

    @classmethod
    def tearDownClass(cls):
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
            try:
                dcs.write_text_file(self.badpath, self.contents)
            except:
                self.assertTrue(sys.exc_info()[0] in [OSError, IOError])
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for writing.')

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.filepath)

#%% convert_annual_to_monthly_probability
class Test_convert_annual_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the convert_annual_to_monthly_probability function with these cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
    """
    def setUp(self):
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1, 12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self):
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_almost_equal(monthly, self.monthly)

    def test_scalar(self):
        monthly = dcs.convert_annual_to_monthly_probability(0)
        self.assertIn(monthly, self.monthly)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, 1.5]))

    def test_circular(self):
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_almost_equal(monthly, self.monthly)
        annual = dcs.convert_monthly_to_annual_probability(monthly)
        np.testing.assert_almost_equal(annual, self.annuals)

#%% convert_monthly_to_annual_probability
class Test_convert_monthly_to_annual_probability(unittest.TestCase):
    r"""
    Tests the convert_annual_to_monthly_probability function with these cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
    """
    def setUp(self):
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1, 12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self):
        annual = dcs.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_almost_equal(annual, self.annuals)

    def test_scalar(self):
        annual = dcs.convert_monthly_to_annual_probability(0)
        self.assertIn(annual, self.annuals)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.convert_monthly_to_annual_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.convert_monthly_to_annual_probability(np.array([0., 0.5, 1.5]))

    def test_circular(self):
        annual = dcs.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_almost_equal(annual, self.annuals)
        monthly = dcs.convert_annual_to_monthly_probability(annual)
        np.testing.assert_almost_equal(monthly, self.monthly)

#%% get_root_dir
class Test_get_root_dir(unittest.TestCase):
    r"""
    Tests the get_root_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_root_dir()
        self.assertTrue(folder) # TODO: don't know an independent way to test this

#%% get_tests_dir
class Test_get_tests_dir(unittest.TestCase):
    r"""
    Tests the get_tests_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_tests_dir()
        self.assertEqual(folder, os.path.join(dcs.get_root_dir(), 'tests'))

#%% get_data_dir
class Test_get_data_dir(unittest.TestCase):
    r"""
    Tests the get_data_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_data_dir()
        self.assertEqual(folder, os.path.join(dcs.get_root_dir(), 'data'))

#%% get_images_dir
class Test_get_images_dir(unittest.TestCase):
    r"""
    Tests the get_images_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_images_dir()
        self.assertEqual(folder, os.path.join(dcs.get_root_dir(), 'images'))

#%% capture_output
class Test_capture_output(unittest.TestCase):
    r"""
    Tests the capture_output function with these cases:
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

#%% unit
class Test_unit(unittest.TestCase):
    r"""
    Tests the unit function with these cases:
        Nominal case
    """
    def setUp(self):
        self.data = np.array([[1, 0, -1],[0, 0, 0], [0, 0, 1]])
        hr2 = np.sqrt(2)/2
        self.norm_data = np.array([[1, 0, -hr2], [0, 0, 0], [0, 0, hr2]])

    def test_nominal(self):
        norm_data = dcs.unit(self.data, axis=0)
        np.testing.assert_almost_equal(norm_data, self.norm_data)

#%% combine_sets
class Test_combine_sets(unittest.TestCase):
    r"""
    Tests the combine_sets function with the following cases:
        Normal use
        No deviation
        Empty set 1
        Empty set 2
        All empty
        Exactly one point
        Negative values (should silently fail)
        Negative values, weird exception case (should raise error)
        Array cases (should raise error)
    """
    def setUp(self):
        self.n1 = 5
        self.u1 = 1
        self.s1 = 0.5
        self.n2 = 10
        self.u2 = 2
        self.s2 = 0.25
        self.n  = 15
        self.u  = 1.6666666666666667
        self.s  = 0.59135639081046598

    def test_nominal(self):
        (n, u, s) = dcs.combine_sets(self.n1, self.u1, self.s1, self.n2, self.u2, self.s2)
        self.assertEqual(n, self.n)
        self.assertAlmostEqual(u, self.u)
        self.assertAlmostEqual(s, self.s)

    def test_no_deviation(self):
        (n, u, s) = dcs.combine_sets(self.n1, self.u1, 0, self.n1, self.u1, 0)
        self.assertEqual(n, 2*self.n1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, 0)

    def test_empty1(self):
        (n, u, s) = dcs.combine_sets(0, 0, 0, self.n2, self.u2, self.s2)
        self.assertEqual(n, self.n2)
        self.assertAlmostEqual(u, self.u2)
        self.assertAlmostEqual(s, self.s2)

    def test_empty2(self):
        (n, u, s) = dcs.combine_sets(self.n1, self.u1, self.s1, 0, 0, 0)
        self.assertEqual(n, self.n1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, self.s1)

    def test_all_empty(self):
        (n, u, s) = dcs.combine_sets(0, 0, 0, 0, 0, 0)
        self.assertEqual(n, 0)
        self.assertEqual(u, 0)
        self.assertEqual(s, 0)

    def test_exactly_one_point1(self):
        (n, u, s) = dcs.combine_sets(1, self.u1, self.s1, 0, 0, 0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(u, self.u1)
        self.assertAlmostEqual(s, self.s1)

    def test_exactly_one_point2(self):
        (n, u, s) = dcs.combine_sets(0, 0, 0, 1, self.u2, self.s2)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(u, self.u2)
        self.assertAlmostEqual(s, self.s2)

    def test_negatives(self):
        try:
            dcs.combine_sets(-self.n1, -self.u1, -self.s1, -self.n2, -self.u2, -self.s2)
        except:
            self.assertTrue(sys.exc_info()[0] in [AssertionError, ValueError])

    def test_negative_weird(self):
        try:
            dcs.combine_sets(5, self.u1, self.s1, -4, self.u2, self.s2)
        except:
            self.assertTrue(sys.exc_info()[0] in [AssertionError, ValueError])

    def test_broadcasting(self):
        with self.assertRaises(ValueError):
            (n, u, s) = dcs.combine_sets(np.array([self.n1, self.n1]), self.u1, self.s1, self.n2, self.u2, self.s2)

#%% reload_package
# Note this this function can't be unittested correctly.  It deletes too many dependent references.
#class Test_reload_package(unittest.TestCase):
#    r"""
#    Tests the reload_package function with the following cases:
#        Normal reload
#        Messages suppressed
#    """
#    def test_nominal(self):
#        with dcs.capture_output() as (out, _):
#            dcs.reload_package(dcs)
#        output = out.getvalue().strip()
#        out.close()
#        self.assertTrue(output.startswith('loading dstauffman.'))
#
#    def test_suppressed(self):
#        dcs.reload_package(dcs, disp_reloads=False)

#%% delete_pyc
class Test_delete_pyc(unittest.TestCase):
    r"""
    Tests the delete_pyc function with the following cases:
        Recursive
        Not recursive
    """
    def setUp(self):
        self.fold1 = dcs.get_tests_dir()
        self.file1 = os.path.join(self.fold1, 'temp_file.pyc')
        self.fold2 = os.path.join(self.fold1, 'temp_sub_dir')
        self.file2 = os.path.join(self.fold2, 'temp_file2.pyc')
        dcs.write_text_file(self.file1, 'Text.')
        os.makedirs(self.fold2)
        dcs.write_text_file(self.file2, 'More text.')

    def test_recursive(self):
        self.assertTrue(os.path.isfile(self.file1))
        self.assertTrue(os.path.isdir(self.fold2))
        self.assertTrue(os.path.isfile(self.file2))
        with dcs.capture_output() as (out, _):
            dcs.delete_pyc(self.fold1)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertFalse(os.path.isfile(self.file1))
        self.assertFalse(os.path.isfile(self.file2))
        for this_line in lines:
            self.assertTrue(this_line.startswith('Removing "'))
            self.assertTrue(this_line.endswith('temp_file.pyc"') or this_line.endswith('temp_file2.pyc"'))

    def test_not_recursive(self):
        self.assertTrue(os.path.isfile(self.file1))
        self.assertTrue(os.path.isdir(self.fold2))
        self.assertTrue(os.path.isfile(self.file2))
        with dcs.capture_output() as (out, _):
            dcs.delete_pyc(self.fold1, recursive=False)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertFalse(os.path.isfile(self.file1))
        self.assertTrue(os.path.isfile(self.file2))
        for this_line in lines:
            self.assertTrue(this_line.startswith('Removing "'))
            self.assertTrue(this_line.endswith('temp_file.pyc"'))

    def test_no_logging(self):
        self.assertTrue(os.path.isfile(self.file1))
        self.assertTrue(os.path.isdir(self.fold2))
        self.assertTrue(os.path.isfile(self.file2))
        with dcs.capture_output() as (out, _):
            dcs.delete_pyc(self.fold1, print_progress=False)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(os.path.isfile(self.file1))
        self.assertFalse(os.path.isfile(self.file2))
        self.assertEqual(output, '')

    def tearDown(self):
        if os.path.isfile(self.file1):
            os.remove(self.file1)
        if os.path.isfile(self.file2):
            os.remove(self.file2)
        if os.path.isdir(self.fold2):
            os.removedirs(self.fold2)

#%% rename_module
class Test_rename_module(unittest.TestCase):
    r"""
    Tests the rename_module function with the following cases:
        TBD
    """
    def setUp(self):
        self.folder = os.path.split(dcs.get_root_dir())[0]
        self.old_name = 'dstauffman'
        self.new_name = 'dcs_tools'
        self.print_status = True

    @unittest.skip # TODO: only print without executing??
    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.rename_module(self.folder, self.new_name, self.old_name, self.print_status)
        output = out.getvalue().strip()
        out.close()
        print(output)
        self.assertEqual(output, 'Text to match...')

#%% modd
class Test_modd(unittest.TestCase):
    r"""
    Tests the modd function with the following cases:
        Nominal
        Scalar
        List (x2)
        Modify in-place
    """
    def setUp(self):
        self.x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        self.y = np.array([ 4,  1,  2,  3, 4, 1, 2, 3, 4])
        self.mod = 4

    def test_nominal(self):
        y = dcs.modd(self.x, self.mod)
        np.testing.assert_array_equal(y, self.y)

    def test_scalar(self):
        y = dcs.modd(4, 4)
        self.assertEqual(y, 4)

    def test_list1(self):
        y = dcs.modd([2, 4], 4)
        np.testing.assert_array_equal(y, np.array([2, 4]))

    def test_list2(self):
        y = dcs.modd(4, [3, 4])
        np.testing.assert_array_equal(y, np.array([1, 4]))

    def test_modify_inplace(self):
        out = np.zeros(self.x.shape, dtype=int)
        dcs.modd(self.x, self.mod, out)
        np.testing.assert_array_equal(out, self.y)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
