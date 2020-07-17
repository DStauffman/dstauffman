r"""
Test file for the `utils` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import copy
import os
import platform
import sys
import unittest

import numpy as np
from scipy.interpolate import interp1d

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

#%% is_np_int
class Test_is_np_int(unittest.TestCase):
    r"""
    Tests the is_np_int function with the following cases:
        int
        float
        large int
        ndarray of int
        ndarray of float
        ndarray of large int
        ndarray of unsigned int
    """
    def test_int(self):
        self.assertTrue(dcs.is_np_int(10))

    def test_float(self):
        self.assertFalse(dcs.is_np_int(10.))

    def test_large_int(self):
        self.assertTrue(dcs.is_np_int(2**62))

    def test_np_int(self):
        self.assertTrue(dcs.is_np_int(np.array([1, 2, 3])))

    def test_np_float(self):
        self.assertFalse(dcs.is_np_int(np.array([2., np.pi])))

    def test_np_large_int(self):
        self.assertTrue(dcs.is_np_int(np.array(2**62)))

    def test_np_uint(self):
        self.assertTrue(dcs.is_np_int(np.array([1, 2, 3], dtype=np.uint32)))

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

#%% histcounts
class Test_histcounts(unittest.TestCase):
    r"""Tests the histcounts function with the following cases:
        TBD
    """
    def setUp(self):
        self.x        = np.array([0.2, 6.4, 3.0, 1.6, 0.5])
        self.bins     = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        self.expected = np.array([2, 1, 1, 1])

    def test_nominal(self):
        hist = dcs.histcounts(self.x, self.bins)
        np.testing.assert_array_equal(hist, self.expected)

    def test_right(self):
        x = np.array([1, 1, 2, 2, 2])
        bins = np.array([0, 1, 2, 3])
        hist = dcs.histcounts(x, bins, right=False)
        np.testing.assert_array_equal(hist, np.array([0, 2, 3]))
        hist2 = dcs.histcounts(x, bins, right=True)
        np.testing.assert_array_equal(hist2, np.array([2, 3, 0]))

    def test_out_of_bounds(self):
        with self.assertRaises(ValueError):
            dcs.histcounts(self.x, np.array([100, 1000]))

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

#%% execute
pass # TODO: write this

#%% execute_wrapper
pass # TODO: write this

#%% get_env_var
class Test_get_env_var(unittest.TestCase):
    r"""
    Tests the get_env_var function with the following cases:
        Valid key
        Unknown key
        Default key
        Not allowed
    """
    def test_valid(self):
        home = dcs.get_env_var('HOME')
        self.assertTrue(bool(home))

    def test_bad_key(self):
        with self.assertRaises(KeyError):
            dcs.get_env_var('Nonexisting_environment_key_name')

    def test_default_key(self):
        key = dcs.get_env_var('Nonexisting_environment_key_name', default='test')
        self.assertEqual(key, 'test')

    @unittest.SkipTest
    def test_not_allowed(self):
        # TODO: fix this test case
        dcs._ALLOWED_ENVS = {'user', 'username'}
        with self.assertRaises(KeyError):
            dcs.get_env_var('HOME')
        dcs._ALLOWED_ENVS = None

#%% Functions - is_datetime
pass # TODO: write this

#%% intersect
class Test_intersect(unittest.TestCase):
    r"""
    Tests the intersect function with the following cases:
        Nominal
        Floats
        Assume unique
    """
    def test_nominal(self):
        a = np.array([1, 2, 4, 4, 6], dtype=int)
        b = np.array([0, 8, 2, 2, 5, 8, 6, 8, 8], dtype=int)
        (c, ia, ib) = dcs.intersect(a, b, return_indices=True)
        np.testing.assert_array_equal(c, np.array([2, 6], dtype=int))
        np.testing.assert_array_equal(ia, np.array([1, 4], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 6], dtype=int))

    def test_floats(self):
        a = np.array([1, 2.5, 4, 6], dtype=float)
        b = np.array([0, 8, 2.5, 4, 6], dtype=float)
        (c, ia, ib) = dcs.intersect(a, b, return_indices=True)
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6], dtype=float))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))

    def test_unique(self):
        a = np.array([1, 2.5, 4, 6], dtype=float)
        b = np.array([0, 8, 2.5, 4, 6], dtype=float)
        (c, ia, ib) = dcs.intersect(a, b, assume_unique=True, return_indices=True)
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6], dtype=float))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))
        (c, ia, ib) = dcs.intersect(a, b, tolerance=1e-7, assume_unique=True, return_indices=True)
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6], dtype=float))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))

    def test_no_indices(self):
        a = np.array([1, 2, 4, 4, 6], dtype=int)
        b = np.array([0, 8, 2, 2, 5, 8, 6, 8, 8], dtype=int)
        c = dcs.intersect(a, b)
        np.testing.assert_array_equal(c, np.array([2, 6], dtype=int))

    def test_tolerance(self):
        a = np.array([1., 2., 3.1, 3.9, 4.0, 6.0])
        b = np.array([2., 3., 4., 5.])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0.12, return_indices=True)
        np.testing.assert_array_equal(c, np.array([2., 3.1, 3.9, 4.0], dtype=float))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3, 4], dtype=int))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2], dtype=int))

    def test_tolerance_no_ix(self):
        a = np.array([1., 3., 5., 7., 9.])
        b = np.array([1.01, 2.02, 3.03, 4.04, 5.05, 6.06, 7.07, 8.08, 9.09])
        c = dcs.intersect(a, b, tolerance=0.055, return_indices=False)
        np.testing.assert_array_equal(c, np.array([1., 3., 5.], dtype=float))
        c2 = dcs.intersect(b, a, tolerance=0.055, return_indices=False)
        np.testing.assert_array_equal(c2, np.array([1.01, 3.03, 5.05], dtype=float))

    def test_scalars(self):
        a = 5
        b = 4.9
        c = dcs.intersect(a, b, tolerance=0.5)
        self.assertEqual(c, 5)

    def test_int(self):
        a = np.array([0, 4, 10, 20, 30, -40, 30])
        b = np.array([1, 5, 7, 31, -10, -40])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0, return_indices=True)
        np.testing.assert_array_equal(c, np.array([-40]))
        np.testing.assert_array_equal(ia, np.array([5]))
        np.testing.assert_array_equal(ib, np.array([5]))

    def test_int_even_tol(self):
        a = np.array([0, 4, 10, 20, 30, -40, 30])
        b = np.array([1, 5, 7, 31, -10, -40])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=2, return_indices=True)
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 30]))
        np.testing.assert_array_equal(ia, np.array([0, 1, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 3, 5]))

    def test_int_odd_tol(self):
        a = np.array([0, 4, 10, 20, 30, -40, 30])
        b = np.array([1, 5, 7, 31, -10, -40])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=3, return_indices=True)
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 10, 30]))
        np.testing.assert_array_equal(ia, np.array([0, 1, 2, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2, 3, 5]))

    def test_int64(self):
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0, return_indices=True)
        np.testing.assert_array_equal(c, np.array([-40], dtype=np.int64) + t_offset)
        np.testing.assert_array_equal(ia, np.array([5]))
        np.testing.assert_array_equal(ib, np.array([5]))

    def test_int64_even_tol(self):
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=2, return_indices=True)
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 30], dtype=np.int64) + t_offset)
        np.testing.assert_array_equal(ia, np.array([0, 1, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 3, 5]))

    def test_int64_odd_tol(self):
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=3, return_indices=True)
        np.testing.assert_array_equal(ia, np.array([0, 1, 2, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2, 3, 5]))
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 10, 30], dtype=np.int64) + t_offset)

    def test_npint64_tol(self):
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=np.array(3), return_indices=True)
        np.testing.assert_array_equal(ia, np.array([0, 1, 2, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2, 3, 5]))
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 10, 30], dtype=np.int64) + t_offset)

    def test_empty(self):
        a = np.array([])
        b = np.array([1, 2, 3, 4])
        c = dcs.intersect(a, b, tolerance=0.1)
        self.assertEqual(len(c), 0)

    def test_datetimes(self):
        date_zero = np.datetime64('2020-06-01 00:00:00', 'ms')
        dt = np.arange(0, 11000, 1000).astype('timedelta64[ms]')
        a = date_zero + dt
        dt[3] += 5
        dt[5] -= 30
        b = date_zero + dt
        # no tolerance
        exp = np.array([0, 1, 2, 4, 6, 7, 8, 9, 10])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0, return_indices=True)
        np.testing.assert_array_equal(c, a[exp])
        np.testing.assert_array_equal(ia, exp)
        np.testing.assert_array_equal(ib, exp)
        # with tolerance
        exp = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=np.timedelta64(10, 'ms'), return_indices=True)
        np.testing.assert_array_equal(c, a[exp])
        np.testing.assert_array_equal(ia, exp)
        np.testing.assert_array_equal(ib, exp)

#%% issorted
class Test_issorted(unittest.TestCase):
    r"""
    Tests the issorted function with the following cases:
        Sorted
        Not sorted
        Reverse sorted (x2)
        Lists
    """
    def test_sorted(self):
        x = np.array([1, 3, 3, 5, 7])
        self.assertTrue(dcs.issorted(x))

    def test_not_sorted(self):
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(dcs.issorted(x))

    def test_reverse_sorted(self):
        x = np.array([4, np.pi, 1., -1.])
        self.assertFalse(dcs.issorted(x))
        self.assertTrue(dcs.issorted(x, descend=True))

    def test_lists(self):
        x = [-np.inf, 0, 1, np.pi, 5, np.inf]
        self.assertTrue(dcs.issorted(x))
        self.assertFalse(dcs.issorted(x, descend=True))

#%% zero_order_hold
class Test_zero_order_hold(unittest.TestCase):
    r"""
    Tests the zero_order_hold function with the following cases:
        Subsample high rate
        Supersample low rate
        xp Not sorted
        x not sorted
        Left extrapolation
        Lists instead of arrays

    Notes
    -----
    #.  Uses scipy.interpolate.interp1d as the gold standard (but it's slower)
    """
    def test_subsample(self):
        xp = np.linspace(0., 100*np.pi, 500000)
        yp = np.sin(2 * np.pi * xp)
        x  = np.arange(0., 350., 0.1)
        func = interp1d(xp, yp, kind='zero', fill_value='extrapolate', assume_sorted=True)
        y_exp = func(x)
        y = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)
        y = dcs.zero_order_hold(x, xp, yp, assume_sorted=True)
        np.testing.assert_array_equal(y, y_exp)

    def test_supersample(self):
        xp = np.array([0., 5000., 10000., 86400.])
        yp = np.array([0, 1, -2, 0])
        x  = np.arange(0., 86400.,)
        func = interp1d(xp, yp, kind='zero', fill_value='extrapolate', assume_sorted=True)
        y_exp = func(x)
        y = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)
        y = dcs.zero_order_hold(x, xp, yp, assume_sorted=True)
        np.testing.assert_array_equal(y, y_exp)

    def test_xp_not_sorted(self):
        xp    = np.array([0, 10, 5, 15])
        yp    = np.array([0, 1, -2, 3])
        x     = np.array([10, 2, 14,  6,  8, 10, 4, 14, 0, 16])
        y_exp = np.array([ 1, 0,  1, -2, -2,  1, 0,  1, 0,  3])
        y     = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)

    def test_x_not_sorted(self):
        xp    = np.array([0, 5, 10, 15])
        yp    = np.array([0, -2, 1, 3])
        x     = np.array([10, 2, 14,  6,  8, 10, 4, 14, 0, 16])
        y_exp = np.array([ 1, 0,  1, -2, -2,  1, 0,  1, 0,  3])
        y     = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)

    def test_left_end(self):
        xp    = np.array([0, 5, 10, 15, 4])
        yp    = np.array([0, 1, -2, 3, 0])
        x     = np.array([-4, -2, 0, 2, 4, 6])
        y_exp = np.array([-5, -5, 0, 0, 0, 1])
        y     = dcs.zero_order_hold(x, xp, yp, left=-5)
        np.testing.assert_array_equal(y, y_exp)

    def test_lists(self):
        xp    = [0, 5, 10, 15]
        yp    = [0, 1, 2, 3]
        x     = [-4, -2, 0, 2, 4, 6, 20]
        y_exp = [-1, -1, 0, 0, 0, 1, 3]
        y     = dcs.zero_order_hold(x, xp, yp, left=-1)
        np.testing.assert_array_equal(y, y_exp)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
