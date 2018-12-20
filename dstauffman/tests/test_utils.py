# -*- coding: utf-8 -*-
r"""
Test file for the `utils` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import copy
import inspect
import logging
import os
import platform
import sys
import time
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

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

#%% rss
class Test_rss(unittest.TestCase):
    r"""
    Tests the rss function with these cases:
        rss on just a scalar input
        normal rss on vector input
        rss on vector with axis specified
        rss on vector with bad axis specified
        rss on matrix without keeping dimensions, no axis
        rss on matrix without keeping dimensions, axis 0
        rss on matrix without keeping dimensions, axis 1
        rss on matrix with keeping dimensions
        rss on complex numbers
        rss on complex numbers that would return a real if done incorrectly
    """
    def setUp(self):
        self.inputs1   = np.array([0, 1, 0, -1])
        self.outputs1  = 2
        self.inputs2   = [[0, 1, 0, -1], [1, 1, 1, 1]]
        self.outputs2a = 6
        self.outputs2b = np.array([1, 2, 1, 2])
        self.outputs2c = np.array([2, 4])
        self.outputs2d = np.matrix([[2], [4]])
        self.inputs3   = np.hstack((self.inputs1, np.nan))
        self.inputs4   = [[0, 0, np.nan], [1, np.nan, 1]]
        self.outputs4a = 2
        self.outputs4b = np.array([1, 0, 1])
        self.outputs4c = np.array([0, 2])

    def test_scalar_input(self):
        out = dcs.rss(-1.5)
        self.assertEqual(out, 1.5**2)

    def test_empty(self):
        out = dcs.rss([])
        self.assertTrue(np.isnan(out))

    def test_rss_series(self):
        out = dcs.rss(self.inputs1)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1a(self):
        out = dcs.rss(self.inputs1, axis=0)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1b(self):
        with self.assertRaises(ValueError):
            dcs.rss(self.inputs1, axis=1)

    def test_axis_drop2a(self):
        out = dcs.rss(self.inputs2)
        self.assertAlmostEqual(out, self.outputs2a)

    def test_axis_drop2b(self):
        out = dcs.rss(self.inputs2, axis=0, keepdims=False)
        for (ix, val) in enumerate(out):
            self.assertAlmostEqual(val, self.outputs2b[ix])

    def test_axis_drop2c(self):
        out = dcs.rss(self.inputs2, axis=1, keepdims=False)
        for (ix, val) in enumerate(out):
            self.assertAlmostEqual(val, self.outputs2c[ix])

    def test_axis_keep(self):
        out = dcs.rss(self.inputs2, axis=1, keepdims=True)
        for i in range(0, len(out)):
            for j in range(0, len(out[i])):
                self.assertAlmostEqual(out[i, j], self.outputs2d[i, j])

    def test_complex_rss(self):
        out = dcs.rss(1.5j)
        self.assertEqual(out, 1.5**2)

    def test_complex_conj(self):
        out = dcs.rss(np.array([1+1j, 1-1j]))
        self.assertAlmostEqual(out, 4)

    def test_with_nans(self):
        out = dcs.rss(self.inputs3, ignore_nans=False)
        self.assertTrue(np.isnan(out))

    def test_ignore_nans1(self):
        out = dcs.rss(self.inputs3, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs1)

    def test_ignore_nans2(self):
        out = dcs.rss(self.inputs4, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs4a)

    def test_ignore_nans3(self):
        out = dcs.rss(self.inputs4, ignore_nans=True, axis=0)
        np.testing.assert_array_almost_equal(out, self.outputs4b)

    def test_ignore_nans4(self):
        out = dcs.rss(self.inputs4, ignore_nans=True, axis=1)
        np.testing.assert_array_almost_equal(out, self.outputs4c)

    def test_all_nans(self):
        out = dcs.rss(np.array([np.nan, np.nan]), ignore_nans=True)
        self.assertTrue(np.isnan(out))

#%% setup_dir
@patch('dstauffman.utils.logger')
class Test_setup_dir(unittest.TestCase):
    r"""
    Tests the setup_dir function with these cases:
        null case
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

    def test_empty_string(self, mock_logger):
        dcs.setup_dir('')
        mock_logger.info.assert_not_called()

    def test_create_folder(self, mock_logger):
        dcs.setup_dir(self.folder)
        mock_logger.info.assert_called_once()
        mock_logger.info.assert_called_with('Created directory: "{}"'.format(self.folder))

    def test_nested_folder(self, mock_logger):
        dcs.setup_dir(self.subdir)
        mock_logger.info.assert_called_once()
        mock_logger.info.assert_called_with('Created directory: "{}"'.format(self.subdir))

    def test_clean_up_folder(self, mock_logger):
        dcs.setup_dir(self.folder)
        dcs.write_text_file(self.filename, self.text)
        with patch('dstauffman.utils.logger') as mock_logger2:
            dcs.setup_dir(self.folder)
            mock_logger2.info.assert_called_once()
            mock_logger2.info.assert_called_with('Files/Sub-folders were removed from: "{}"'.format(self.folder))
        mock_logger.info.assert_called_once()

    def test_clean_up_partial(self, mock_logger):
        dcs.setup_dir(self.folder)
        dcs.write_text_file(self.filename, '')
        dcs.setup_dir(self.subdir)
        dcs.write_text_file(self.subfile, '')
        with patch('dstauffman.utils.logger') as mock_logger2:
            dcs.setup_dir(self.folder, rec=False)
            mock_logger2.info.assert_called_once()
            mock_logger2.info.assert_called_with('Files/Sub-folders were removed from: "{}"'.format(self.folder))
        self.assertEqual(mock_logger.info.call_count, 2)

    def test_fail_to_create_folder(self, mock_logger):
        pass #TODO: write this test

    def test_fail_to_clean_folder(self, mock_logger):
        pass #TODO: write this test

    def test_bad_name_file_ext(self, mock_logger):
        pass #TODO: write this test

    def test_clean_up_recursively(self, mock_logger):
        dcs.setup_dir(self.subdir)
        dcs.write_text_file(self.subfile, self.text)
        with patch('dstauffman.utils.logger') as mock_logger2:
            dcs.setup_dir(self.folder, rec=True)
            self.assertEqual(mock_logger2.info.call_count, 2)
            mock_logger2.info.assert_any_call('Files/Sub-folders were removed from: "{}"'.format(self.subdir))
            mock_logger2.info.assert_any_call('Files/Sub-folders were removed from: "{}"'.format(self.subdir))

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
        except {PermissionError, OSError}:
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
        self.c1 = type('Class1', (object, ), {'a': 0, 'b': '[1, 2, 3]', 'c': 'text', 'e': {'key1': 1}})
        self.c2 = type('Class2', (object, ), {'a': 0, 'b': '[1, 2, 4]', 'd': 'text', 'e': {'key1': 1}})
        self.names = ['Class 1', 'Class 2']
        self.c3 = type('Class3', (object, ), {'a': 0, 'b': '[1, 2, 3]', 'c': 'text', 'e': self.c1})
        self.c4 = type('Class4', (object, ), {'a': 0, 'b': '[1, 2, 4]', 'd': 'text', 'e': self.c2})

    def test_is_comparison(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, self.c1)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_good_comparison(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, copy.deepcopy(self.c1))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different from c1 to c2.\nc is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

    def test_names(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c2, self.c2, names=self.names)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"Class 1" and "Class 2" are the same.')
        self.assertTrue(is_same)

    def test_suppression(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, self.c2, suppress_output=True, names=self.names)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')
        self.assertFalse(is_same)

    def test_subclasses_match(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c3, self.c3, ignore_callables=False)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_subclasses_recurse(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_same)
        self.assertEqual(output, 'b is different from c1 to c2.\nb is different from c1.e to c2.e.\n' + \
            'c is only in c1.e.\nd is only in c2.e.\n"c1.e" and "c2.e" are not the same.\n' + \
            'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')

    def test_subclasses_norecurse(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False, compare_recursively=False)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different from c1 to c2.\n' + \
            'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

    def test_subdict_comparison(self):
        delattr(self.c1, 'b')
        delattr(self.c1, 'c')
        delattr(self.c2, 'b')
        delattr(self.c2, 'd')
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)
        self.c1.e['key1'] += 1
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

    def test_custom_dicts(self):
        delattr(self.c1, 'b')
        delattr(self.c1, 'c')
        delattr(self.c2, 'b')
        delattr(self.c2, 'd')
        self.c1.e = dcs.FixedDict()
        self.c1.e['key1'] = 1
        self.c1.e.freeze()
        self.c2.e = dcs.FixedDict()
        self.c2.e['key1'] = 1
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1.e" and "c2.e" are the same.\n"c1" and "c2" are the same.')
        self.assertTrue(is_same)
        self.c1.e['key1'] += 1
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'key1 is different.\n"c1.e" and "c2.e" are not the same.\n"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

    def test_mismatched_subclasses(self):
        self.c4.e = 5
        with dcs.capture_output() as out:
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
        with dcs.capture_output() as out:
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
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(self.c4, self.c3, ignore_callables=True)
        output = out.getvalue().strip()
        out.close()
        self.assertFalse(is_same)
        self.assertEqual(output, 'c is only in c2.\nd is only in c1.\n"c1" and "c2" are not the same.')

    def test_two_different_lists(self):
        c1 = [1]
        c2 = [1]
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_classes(c1, c2, ignore_callables=True)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

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
        self.d1 = {'a': 1, 'b': 2, 'c': 3, 'e': {'key1':1}}
        self.d2 = {'a': 1, 'b': 5, 'd': 6, 'e': {'key1':1}}
        self.names = ['Dict 1', 'Dict 2']

    def test_good_comparison(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_dicts(self.d1, self.d1)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"d1" and "d2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_dicts(self.d1, self.d2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'b is different.\nc is only in d1.\nd is only in d2.\n"d1" and "d2" are not the same.')
        self.assertFalse(is_same)

    def test_names(self):
        with dcs.capture_output() as out:
            is_same = dcs.compare_two_dicts(self.d2, self.d2, names=self.names)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '"Dict 1" and "Dict 2" are the same.')
        self.assertTrue(is_same)

    def test_suppression(self):
        with dcs.capture_output() as out:
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
        self.text     = 'from .bpe import'
        self.text2    = 'from .bpe          import'
        self.folder2  = dcs.get_tests_dir()
        self.filepath = os.path.join(self.folder2, 'temp_file.py')
        self.filename = os.path.join(self.folder2, '__init__2.py')

    def test_nominal_use(self):
        text = dcs.make_python_init(self.folder)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text2)], self.text2)

    def test_duplicated_funcs(self):
        with open(self.filepath, 'wt') as file:
            file.write('def Test_Frozen():\n    pass\n')
        with dcs.capture_output() as out:
            text = dcs.make_python_init(self.folder2)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(text[0:42], 'from .temp_file         import Test_Frozen')
        self.assertTrue(output.startswith('Uniqueness Problem'))

    def test_no_lineup(self):
        text = dcs.make_python_init(self.folder, lineup=False)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text)], self.text)

    def test_big_wrap(self):
        text = dcs.make_python_init(self.folder, wrap=1000)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text2)], self.text2)

    def test_small_wrap(self):
        with self.assertRaises(ValueError) as context:
            dcs.make_python_init(self.folder, wrap=30)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "26:30" was too small.')

    def test_really_small_wrap(self):
        with self.assertRaises(ValueError) as context:
            dcs.make_python_init(self.folder, wrap=10)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "26:10" was too small.')

    def test_saving(self):
        text = dcs.make_python_init(self.folder, filename=self.filename)
        lines = text.split('\n')
        self.assertEqual(lines[1][0:len(self.text2)], self.text2)
        self.assertTrue(os.path.isfile(self.filename))

    def tearDown(self):
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)
        if os.path.isfile(self.filename):
            os.remove(self.filename)

#%% get_python_definitions
class Test_get_python_definitions(unittest.TestCase):
    r"""
    Tests the get_python_definitions function with these cases:
        Functions
        Classes
        No arguments
        Lots of arguments
    """
    def test_functions(self):
        funcs = dcs.get_python_definitions('def a():\n    pass\ndef _b():\n    pass\n')
        self.assertEqual(funcs, ['a'])

    def test_classes(self):
        funcs = dcs.get_python_definitions('def a():\n    pass\nclass b():\n    pass\nclass _c():\n    pass\n')
        self.assertEqual(funcs, ['a', 'b'])

    def test_no_inputs(self):
        funcs = dcs.get_python_definitions('def _a:\n    pass\ndef b:\n    pass\n')
        self.assertEqual(funcs, ['b'])

    def test_with_inputs(self):
        funcs = dcs.get_python_definitions('def a(a, b=2):\n    pass\nclass bbb(c, d):\n    pass\nclass _c(e):\n    pass\n')
        self.assertEqual(funcs, ['a', 'bbb'])

    def test_nothing(self):
        funcs = dcs.get_python_definitions('')
        self.assertEqual(len(funcs), 0)

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
        with dcs.capture_output() as out:
            try:
                dcs.read_text_file(self.badpath)
            except:
                self.assertTrue(sys.exc_info()[0] in [OSError, IOError, FileNotFoundError])
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
        if platform.system() != 'Windows':
            return
        with dcs.capture_output() as out:
            try:
                dcs.write_text_file(self.badpath, self.contents)
            except:
                self.assertTrue(sys.exc_info()[0] in [OSError, IOError, FileNotFoundError])
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for writing.')

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.filepath)

#%% get_root_dir
class Test_get_root_dir(unittest.TestCase):
    r"""
    Tests the get_root_dir function with these cases:
        call the function
    """
    def test_function(self):
        filepath      = inspect.getfile(dcs.get_root_dir)
        expected_root = os.path.split(filepath)[0]
        folder = dcs.get_root_dir()
        self.assertEqual(folder, expected_root)
        self.assertTrue(os.path.isdir(folder))

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

#%% get_output_dir
class Test_get_output_dir(unittest.TestCase):
    r"""
    Tests the get_output_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_output_dir()
        self.assertEqual(folder, os.path.join(dcs.get_root_dir(), 'results'))

#%% capture_output
class Test_capture_output(unittest.TestCase):
    r"""
    Tests the capture_output function with these cases:
        capture standard output
        capture standard error
    """
    def test_std_out(self):
        with dcs.capture_output() as out:
            print('Hello, World!')
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Hello, World!')

    def test_std_err(self):
        with dcs.capture_output('err') as err:
            print('Error Raised.', file=sys.stderr)
        error  = err.getvalue().strip()
        err.close()
        self.assertEqual(error, 'Error Raised.')

    def test_all(self):
        with dcs.capture_output('all') as (out, err):
            print('Hello, World!')
            print('Error Raised.', file=sys.stderr)
        output = out.getvalue().strip()
        error  = err.getvalue().strip()
        out.close()
        err.close()
        self.assertEqual(output, 'Hello, World!')
        self.assertEqual(error, 'Error Raised.')

    def test_bad_value(self):
        with self.assertRaises(RuntimeError):
            with dcs.capture_output('bad') as (out, err):
                print('Lost values')

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
        np.testing.assert_array_almost_equal(norm_data, self.norm_data)

    def test_bad_axis(self):
        with self.assertRaises(ValueError) as context:
            dcs.unit(self.data, axis=2)
        self.assertEqual(str(context.exception), 'axis 2 is out of bounds for array of dimension 2')

    def test_single_vector(self):
        for i in range(3):
            norm_data = dcs.unit(self.data[:, i])
            np.testing.assert_array_almost_equal(norm_data, self.norm_data[:, i])

    def test_single_vector_axis0(self):
        for i in range(3):
            norm_data = dcs.unit(self.data[:, i], axis=0)
            np.testing.assert_array_almost_equal(norm_data, self.norm_data[:, i])

    def test_single_vector_bad_axis(self):
        with self.assertRaises(ValueError) as context:
            dcs.unit(self.data[:, 0], axis=1)
        self.assertEqual(str(context.exception), 'axis 1 is out of bounds for array of dimension 1')

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
#        with dcs.capture_output() as out:
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
        with dcs.capture_output() as out:
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
        with dcs.capture_output() as out:
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
        with dcs.capture_output() as out:
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
        Nominal
    """
    def setUp(self):
        self.folder   = dcs.get_tests_dir()
        self.old_name = 'test1'
        self.new_name = 'test2'
        self.print_status = True
        self.old_dir    = os.path.join(self.folder, self.old_name)
        self.new_dir    = os.path.join(self.folder, self.new_name)
        self.git_dir    = os.path.join(self.old_dir, '.git')
        self.files      = ['__init__.py', '__init__.pyc', '__init__.misc']
        # make some files
        if not os.path.isdir(self.old_dir):
            os.mkdir(self.old_dir)
        if not os.path.isdir(self.git_dir):
            os.mkdir(self.git_dir)
        dcs.write_text_file(os.path.join(self.old_dir, '__init__.py'),'# Init file for "temp1".\n')
        dcs.write_text_file(os.path.join(self.old_dir, '__init__.pyc'),'')
        dcs.write_text_file(os.path.join(self.old_dir, '__init__.misc'),'# Misc file for "temp1".\n')

    def test_nominal(self):
        with dcs.capture_output() as out:
            dcs.rename_module(self.folder, self.old_name, self.new_name, self.print_status)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        # check for dir creation, files copied, files skipped, files edited
        # expect at least one of each
        self.assertTrue(os.path.isdir(self.new_dir))
        for this_line in lines:
            if this_line.startswith('Copying : '):
                break
        else:
            self.assertTrue(False,'No files were copied.')
        for this_line in lines:
            if this_line.startswith('Editing : '):
                break
        else:
            self.assertTrue(False,'No files were edited.')
        for this_line in lines:
            if this_line.startswith('Skipping: '):
                break
        else:
            self.assertTrue(False,'No files were skipped.')
        self.assertTrue(os.path.isfile(os.path.join(self.new_dir, '__init__.py')))
        self.assertFalse(os.path.isfile(os.path.join(self.new_dir, '__init__.pyc')))
        self.assertTrue(os.path.isfile(os.path.join(self.new_dir, '__init__.misc')))

    def test_no_printing(self):
        with dcs.capture_output() as out:
            dcs.rename_module(self.folder, self.old_name, self.new_name, print_status=False)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '')
        self.assertTrue(os.path.isfile(os.path.join(self.new_dir, '__init__.py')))
        self.assertFalse(os.path.isfile(os.path.join(self.new_dir, '__init__.pyc')))
        self.assertTrue(os.path.isfile(os.path.join(self.new_dir, '__init__.misc')))

    def tearDown(self):
        for this_file in self.files:
            if os.path.isfile(os.path.join(self.old_dir, this_file)):
                os.remove(os.path.join(self.old_dir, this_file))
            if os.path.isfile(os.path.join(self.new_dir, this_file)):
                os.remove(os.path.join(self.new_dir, this_file))
        if os.path.isdir(self.git_dir):
            os.rmdir(self.git_dir)
        if os.path.isdir(self.old_dir):
            os.rmdir(self.old_dir)
        if os.path.isdir(self.new_dir):
            os.rmdir(self.new_dir)

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

#%% find_tabs
class Test_find_tabs(unittest.TestCase):
    r"""
    Tests the find_tabs function with the following cases:
        Nominal
        Different Extensions
        List All
        Trailing Spaces
        Exclusions x2
        Bad New Lines
    """

    @classmethod
    def setUpClass(cls):
        cls.folder = dcs.get_tests_dir()
        cls.linesep = os.linesep.replace('\n', '\\n').replace('\r', '\\r')
        file1 = os.path.join(cls.folder, 'temp_code_01.py')
        file2 = os.path.join(cls.folder, 'temp_code_02.py')
        file3 = os.path.join(cls.folder, 'temp_code_03.m')
        cont1 = 'Line 1\n\nAnother line\n    Line with leading spaces\n'
        cont2 = '\n\n    Start line\nNo Bad tab lines\n    Start and end line    \nAnother line\n\n'
        cont3 = '\n\n    Start line\n\tBad tab line\n    Start and end line    \nAnother line\n\n'
        cls.files = [file1, file2, file3]
        dcs.write_text_file(file1, cont1)
        dcs.write_text_file(file2, cont2)
        dcs.write_text_file(file3, cont3)
        cls.bad1 = "    Line 004: '\\tBad tab line" + cls.linesep + "'"
        cls.bad2 = "    Line 005: '    Start and end line    " + cls.linesep + "'"

    def test_nominal(self):
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, extensions='m', list_all=False, trailing=False)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad1)
        self.assertEqual(lines[2], '')
        self.assertEqual(len(lines), 3)

    def test_different_extensions(self):
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, extensions=('txt',))
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], '')
        self.assertEqual(len(lines), 1)

    def test_list_all(self):
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, list_all=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(self.bad1 in lines)
        self.assertFalse(self.bad2 in lines)

    def test_trailing_spaces(self):
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, trailing=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad2)
        self.assertTrue(lines[2].startswith('Evaluating: "'))
        self.assertEqual(lines[3], self.bad1)
        self.assertEqual(lines[4], self.bad2)
        self.assertEqual(lines[5], '')
        self.assertEqual(len(lines), 6)

    def test_trailing_and_list_all(self):
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, list_all=True, trailing=True)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertTrue(self.bad1 in lines)
        self.assertTrue(self.bad2 in lines)
        self.assertTrue(len(lines) > 7)

    def test_exclusions_skip(self):
        exclusions = (self.folder)
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, exclusions=exclusions)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertEqual(lines, [''])

    def test_exclusions_invalid(self):
        exclusions = (r'C:\non_existant_path', )
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, exclusions=exclusions)
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('Evaluating: "'))
        self.assertEqual(lines[1], self.bad1)
        self.assertEqual(lines[2], '')
        self.assertEqual(len(lines),  3)

    def test_bad_newlines(self):
        with dcs.capture_output() as out:
            dcs.find_tabs(self.folder, extensions='m', check_eol='0')
        lines = out.getvalue().split('\n')
        out.close()
        self.assertTrue(lines[0].startswith('File: "'))
        self.assertTrue(lines[0].endswith('" has bad line endings of "{}".'.format(self.linesep)))
        self.assertTrue(lines[1].startswith('Evaluating: "'))
        self.assertEqual(lines[2], self.bad1)
        self.assertEqual(lines[3], '')
        self.assertEqual(len(lines),  4)

    @classmethod
    def tearDownClass(cls):
        for this_file in cls.files:
            if os.path.isfile(this_file):
                os.remove(this_file)

#%% np_digitize
class Test_np_digitize(unittest.TestCase):
    r"""
    Tests the np_digitize function with the following cases:
        Nominal
        Bad minimum
        Bad maximum
        Empty input
        Optional right flag
        Bad NaNs
    """
    def setUp(self):
        self.x    = np.array([1.1, 2.2, 3.3, 3.3, 5.5, 10])
        self.bins = np.array([-1, 2, 3.1, 4, 4.4, 6, 20])
        self.out  = np.array([0, 1, 2, 2, 4, 5], dtype=int)

    def test_nominal(self):
        out = dcs.np_digitize(self.x, self.bins)
        np.testing.assert_array_equal(out, self.out)

    def test_bad_min(self):
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([-5, 5]), self.bins)

    def test_bad_max(self):
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([5, 25]), self.bins)

    def test_empty(self):
        out = dcs.np_digitize(np.array([]), self.bins)
        self.assertEqual(out.size, 0)

    def test_right(self):
        out = dcs.np_digitize(self.x, self.bins, right=True)
        np.testing.assert_array_equal(out, self.out)

    def test_bad_right(self):
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([5, 25]), self.bins, right=True)

    def test_for_nans(self):
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([1, 10, np.nan]), self.bins)

#%% full_print
class Test_full_print(unittest.TestCase):
    r"""Tests the full_print function with the following cases:
        Nominal
        Small (x2)
    """
    @staticmethod
    def _norm_output(lines):
        out = []
        for line in lines:
            # normalize whitespace
            temp = ' '.join(line.strip().split())
            # get rid of spaces near brackets
            temp = temp.replace('[ ', '[')
            temp = temp.replace(' ]', ']')
            out.append(temp)
        return out

    def setUp(self):
        self.x = np.zeros((10, 5))
        self.x[3, :] = 1.23
        self.x_print = ['[[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]',
            '...', '[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]]']
        self.x_full  = ['[[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]','[0. 0. 0. 0. 0.]',
            '[1.23 1.23 1.23 1.23 1.23]', '[0. 0. 0. 0. 0.]','[0. 0. 0. 0. 0.]',
            '[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]', '[0. 0. 0. 0. 0.]]']
        # explicitly set default threshold to 10 (since consoles sometimes use 1000 instead)
        self.orig = np.get_printoptions()
        np.set_printoptions(threshold=10)

    def test_nominal(self):
        with dcs.capture_output() as out:
            print(self.x)
        lines = out.getvalue().strip().split('\n')
        out.close()
        # normalize whitespace
        lines = self._norm_output(lines)
        self.assertEqual(lines, self.x_print)
        with dcs.capture_output() as out:
            with dcs.full_print():
                print(self.x)
        lines = out.getvalue().strip().split('\n')
        # normalize whitespace
        lines = self._norm_output(lines)
        out.close()
        self.assertEqual(lines, self.x_full)

    def test_small1(self):
        with dcs.capture_output() as out:
            with dcs.full_print():
                print(np.array(0))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '0')

    def test_small2(self):
        with dcs.capture_output() as out:
            with dcs.full_print():
                print(np.array([1.35, 1.58]))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '[1.35 1.58]')

    def tearDown(self):
        # restore the print_options
        np.set_printoptions(**self.orig)

#%% pprint_dict
class Test_pprint_dict(unittest.TestCase):
    r"""
    Tests the pprint_dict function with the following cases:
        Nominal
        No name
        Different indentation
        No alignment
    """
    def setUp(self):
        self.name   = 'Example'
        self.dct    = {'a': 1, 'bb': 2, 'ccc': 3}

    def test_nominal(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')

    def test_no_name(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'a   = 1')
        self.assertEqual(lines[1], ' bb  = 2')
        self.assertEqual(lines[2], ' ccc = 3')

    def test_indent(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name, indent=4)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], '    a   = 1')
        self.assertEqual(lines[2], '    bb  = 2')
        self.assertEqual(lines[3], '    ccc = 3')

    def test_no_align(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name, align=False)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a = 1')
        self.assertEqual(lines[2], ' bb = 2')
        self.assertEqual(lines[3], ' ccc = 3')

    def test_printed(self):
        with dcs.capture_output() as out:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=True)
        output = out.getvalue().strip()
        lines = output.split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')
        self.assertEqual(text, output)

    def test_not_printed(self):
        with dcs.capture_output() as out:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=False)
        output = out.getvalue().strip()
        self.assertEqual(output, '')
        lines = text.split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')

#%% line_wrap
class Test_line_wrap(unittest.TestCase):
    r"""
    Tests the line_wrap function with the following cases:
        TBD
    """
    def setUp(self):
        self.text     = ('lots of repeated words ' * 4).strip()
        self.wrap     = 40
        self.min_wrap = 0
        self.indent   = 4
        self.out      = ['lots of repeated words lots of \\', '    repeated words lots of repeated \\', \
            '    words lots of repeated words']

    def test_str(self):
        out = dcs.line_wrap(self.text, self.wrap, self.min_wrap, self.indent)
        self.assertEqual(out, '\n'.join(self.out))

    def test_list(self):
        out = dcs.line_wrap([self.text], self.wrap, self.min_wrap, self.indent)
        self.assertEqual(out, self.out)

    def test_list2(self):
        out = dcs.line_wrap(3*['aaaaaaaaaa bbbbbbbbbb cccccccccc'], wrap=25, min_wrap=15, indent=2)
        self.assertEqual(out, 3*['aaaaaaaaaa bbbbbbbbbb \\', '  cccccccccc'])

    def test_min_wrap(self):
        out = dcs.line_wrap('aaaaaaaaaaaaaaaaaaaa bbbbbbbbbb', 25, 18, 0)
        self.assertEqual(out, 'aaaaaaaaaaaaaaaaaaaa \\\nbbbbbbbbbb')

    def test_min_wrap2(self):
        with self.assertRaises(ValueError) as context:
            dcs.line_wrap('aaaaaaaaaaaaaaaaaaaa bbbbbbbbbb', 25, 22, 0)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "22:25" was too small.')

#%% combine_per_year
class Test_combine_per_year(unittest.TestCase):
    r"""
    Tests the combine_per_year function with the following cases:
        1D
        2D
        data is None
        all nans
        non-integer 12 month 1D
        non-integer 12 month 2D
        Different specified function
        Bad function (not specified)
        Bad function (not callable)
    """
    def setUp(self):
        self.time  = np.arange(120)
        self.data  = self.time // 12
        self.data2 = np.arange(10)
        self.data3 = np.column_stack((self.data, self.data))
        self.data4 = np.column_stack((self.data2, self.data2))
        self.data5 = np.full(120, np.nan, dtype=float)
        self.func1 = np.nanmean
        self.func2 = np.nansum

    def test_1D(self):
        data2 = dcs.combine_per_year(self.data, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data2)

    def test_2D(self):
        data2 = dcs.combine_per_year(self.data3, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data4)

    def test_data_is_none(self):
        data2 = dcs.combine_per_year(None, func=self.func1)
        self.assertEqual(data2, None)

    def test_data_is_all_nan(self):
        data2 = dcs.combine_per_year(self.data5, func=self.func1)
        self.assertTrue(len(data2) == 10)
        self.assertTrue(np.all(np.isnan(data2)))

    def test_non12_months1d(self):
        data = np.arange(125) // 12
        data2 = dcs.combine_per_year(data, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data2)

    def test_non12_months2d(self):
        data = np.arange(125) // 12
        data3 = np.column_stack((data, data))
        data2 = dcs.combine_per_year(data3, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data4)

    def test_other_funcs(self):
        data2a = dcs.combine_per_year(self.data, func=self.func1)
        data2b = dcs.combine_per_year(self.data, func=self.func2)
        np.testing.assert_array_almost_equal(12 * data2a, data2b)

    def test_bad_func1(self):
        with self.assertRaises(AssertionError):
            dcs.combine_per_year(self.data)

    def test_bad_func2(self):
        with self.assertRaises(AssertionError):
            dcs.combine_per_year(self.data, func=1.5)

#%% activate_logging and deactivate_logging
class Test_act_deact_logging(unittest.TestCase):
    r"""
    Tests the activate_logging and deactivate_logging functions with the following cases:
        Nominal
        Default filename
    """
    def setUp(self):
        self.level    = logging.DEBUG
        self.filename = os.path.join(dcs.get_tests_dir(), 'testlog.txt')

    def test_nominal(self):
        self.assertFalse(os.path.isfile(self.filename))
        dcs.activate_logging(self.level, self.filename)
        self.assertTrue(os.path.isfile(self.filename))
        self.assertTrue(dcs.utils.root_logger.hasHandlers())
        with self.assertLogs(level='DEBUG') as cm:
            logger = logging.getLogger('Test')
            logger.debug('Test message')
        lines = cm.output
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], 'DEBUG:Test:Test message')
        dcs.deactivate_logging()
        self.assertFalse(dcs.utils.root_logger.handlers)

    def test_default_filename(self):
        default_filename = os.path.join(dcs.get_output_dir(), 'log_file_' + datetime.now().strftime('%Y-%m-%d') + '.txt')
        was_there = os.path.isfile(default_filename)
        dcs.activate_logging(self.level)
        self.assertTrue(dcs.utils.root_logger.hasHandlers())
        time.sleep(0.01)
        dcs.deactivate_logging()
        self.assertFalse(dcs.utils.root_logger.handlers)
        if not was_there:
            os.remove(default_filename)

    def tearDown(self):
        dcs.deactivate_logging()
        if os.path.isfile(self.filename):
            os.remove(self.filename)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
