r"""
Test file for the `utils_log` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.  Split to separate file in June 2020.
"""

#%% Imports
import os
import time
import unittest
from unittest.mock import patch

import numpy as np

import dstauffman as dcs

#%% setup_dir
@patch('dstauffman.utils_log.logger')
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
        mock_logger.log.assert_not_called()

    def test_create_folder(self, mock_logger):
        dcs.setup_dir(self.folder)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(dcs.LogLevel.L1, 'Created directory: "{}"'.format(self.folder))

    def test_nested_folder(self, mock_logger):
        dcs.setup_dir(self.subdir)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(dcs.LogLevel.L1, 'Created directory: "{}"'.format(self.subdir))

    def test_clean_up_folder(self, mock_logger):
        dcs.setup_dir(self.folder)
        dcs.write_text_file(self.filename, self.text)
        with patch('dstauffman.utils_log.logger') as mock_logger2:
            dcs.setup_dir(self.folder)
            mock_logger2.log.assert_called_once()
            mock_logger2.log.assert_called_with(dcs.LogLevel.L1, 'Files/Sub-folders were removed from: "{}"'.format(self.folder))
        mock_logger.log.assert_called_once()

    def test_clean_up_partial(self, mock_logger):
        dcs.setup_dir(self.folder)
        dcs.write_text_file(self.filename, '')
        dcs.setup_dir(self.subdir)
        dcs.write_text_file(self.subfile, '')
        with patch('dstauffman.utils_log.logger') as mock_logger2:
            dcs.setup_dir(self.folder, rec=False)
            mock_logger2.log.assert_called_once()
            mock_logger2.log.assert_called_with(dcs.LogLevel.L1, 'Files/Sub-folders were removed from: "{}"'.format(self.folder))
        self.assertEqual(mock_logger.log.call_count, 2)

    def test_fail_to_create_folder(self, mock_logger):
        pass #TODO: write this test

    def test_fail_to_clean_folder(self, mock_logger):
        pass #TODO: write this test

    def test_bad_name_file_ext(self, mock_logger):
        pass #TODO: write this test

    def test_clean_up_recursively(self, mock_logger):
        dcs.setup_dir(self.subdir)
        dcs.write_text_file(self.subfile, self.text)
        with patch('dstauffman.utils_log.logger') as mock_logger2:
            dcs.setup_dir(self.folder, rec=True)
            self.assertEqual(mock_logger2.log.call_count, 2)
            mock_logger2.log.assert_any_call(dcs.LogLevel.L1, 'Files/Sub-folders were removed from: "{}"'.format(self.subdir))
            mock_logger2.log.assert_any_call(dcs.LogLevel.L1, 'Files/Sub-folders were removed from: "{}"'.format(self.subdir))

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

#%% fix_rollover
@patch('dstauffman.utils_log.logger')
class Test_fix_rollover(unittest.TestCase):
    r"""
    Tests the fix_rollover function with the following cases:
        Nominal
        Matrix dim 1
        Matrix dim 2
        Log level 1
        Optional inputs
    """
    def setUp(self):
        self.data  = np.array([1, 2, 3, 4, 5, 6, 0, 1,  3,  6,  0,  6,  5, 2])
        self.data2 = np.array([])
        self.exp   = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 13, 12, 9])
        self.roll  = 7
        self.axis  = None

    def test_nominal(self, mock_logger):
        unrolled = dcs.fix_rollover(self.data, self.roll)
        np.testing.assert_array_equal(unrolled, self.exp)
        mock_logger.log.assert_any_call(dcs.LogLevel.L6, 'corrected 1 bottom to top rollover(s)')
        mock_logger.log.assert_called_with(dcs.LogLevel.L6, 'corrected 2 top to bottom rollover(s)')
        self.assertEqual(mock_logger.log.call_count, 2)

    def test_matrix_dim1(self, mock_logger):
        self.axis = 0
        data      = np.vstack((self.data, self.data))
        exp       = np.vstack((self.data, self.data))
        unrolled  = dcs.fix_rollover(data, self.roll, axis=self.axis)
        np.testing.assert_array_equal(unrolled, exp)
        mock_logger.log.assert_not_called()

    def test_matrix_dim2(self, mock_logger):
        self.axis  = 1
        self.data2 = np.vstack((self.data, self.data))
        exp        = np.vstack((self.exp, self.exp))
        unrolled   = dcs.fix_rollover(self.data2, self.roll, axis=self.axis)
        np.testing.assert_array_equal(unrolled, exp)
        mock_logger.log.assert_any_call(dcs.LogLevel.L6, 'corrected 1 bottom to top rollover(s)')
        mock_logger.log.assert_called_with(dcs.LogLevel.L6, 'corrected 2 top to bottom rollover(s)')
        self.assertEqual(mock_logger.log.call_count, 4)

    def test_non_integer_roll(self, mock_logger):
        exp      = np.arange(0., 10.1, 0.1)
        roll     = 3.35
        data     = roll * ((exp / roll) % 1)
        unrolled = dcs.fix_rollover(data, roll)
        np.testing.assert_array_almost_equal(unrolled, exp, decimal=12)
        mock_logger.log.assert_called_once_with(dcs.LogLevel.L6, 'corrected 2 top to bottom rollover(s)')

    def test_signed_rollover(self, mock_logger):
        exp  = np.arange(21)
        data = np.array([0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, -4])
        roll = 8
        unrolled = dcs.fix_rollover(data, roll)
        np.testing.assert_array_equal(unrolled, exp)
        mock_logger.log.assert_called_with(dcs.LogLevel.L6, 'corrected 3 top to bottom rollover(s)')

    def test_recursive(self, mock_logger):
        pass # TODO: figure out a test case where this actually happens.  I think the code was there for a reason?

    def test_empty(self, mock_logger):
        data = dcs.fix_rollover(np.array([]), self.roll)
        self.assertEqual(data.ndim, 1)
        self.assertEqual(data.size, 0)

    def test_bad_ndims(self, mock_logger):
        with self.assertRaises(ValueError) as context:
            dcs.fix_rollover(np.zeros((2, 5), dtype=float), self.roll)
        self.assertEqual(str(context.exception), 'Input argument "data" must be a vector.')

    def test_bad_axis(self, mock_logger):
        with self.assertRaises(AssertionError):
            dcs.fix_rollover(np.zeros((2, 3, 4), dtype=float), self.roll, axis=2)
        with self.assertRaises(ValueError) as context:
            dcs.fix_rollover(np.zeros((2, 5), dtype=float), self.roll, axis=2)
        self.assertEqual(str(context.exception), 'Unexpected axis: "2".')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
