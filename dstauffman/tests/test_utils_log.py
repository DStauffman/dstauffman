r"""
Test file for the `utils_log` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.  Split to separate file in June 2020.
"""

# %% Imports
from __future__ import annotations

import contextlib
import time
from typing import Optional, TYPE_CHECKING
import unittest
from unittest.mock import Mock, patch

from slog import LogLevel

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _N = NDArray[np.float64]


# %% setup_dir
@patch("dstauffman.utils_log.logger")
class Test_setup_dir(unittest.TestCase):
    r"""
    Tests the setup_dir function with the following cases:
        null case
        create a new folder
        create a new nested folder
        delete the contents of an existing folder
        fail to create a folder due to permissions
        fail to delete the contents of an existing folder due to permissions
        fail to create a folder due to a bad name
        delete the contents of an existing folder recursively
    """

    def setUp(self) -> None:
        self.folder = dcs.get_tests_dir() / "temp_dir"
        self.subdir = dcs.get_tests_dir().joinpath("temp_dir", "temp_dir2")
        self.filename = self.folder / "temp_file.txt"
        self.subfile = self.subdir / "temp_file.txt"
        self.text = "Hello, World!\n"

    def test_empty_string(self, mock_logger: Mock) -> None:
        dcs.setup_dir("")
        mock_logger.log.assert_not_called()

    def test_create_folder(self, mock_logger: Mock) -> None:
        dcs.setup_dir(self.folder)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(LogLevel.L1, 'Created directory: "%s"', self.folder)

    def test_nested_folder(self, mock_logger: Mock) -> None:
        dcs.setup_dir(self.subdir)
        mock_logger.log.assert_called_once()
        mock_logger.log.assert_called_with(LogLevel.L1, 'Created directory: "%s"', self.subdir)

    def test_clean_up_folder(self, mock_logger: Mock) -> None:
        dcs.setup_dir(self.folder)
        dcs.write_text_file(self.filename, self.text)
        with patch("dstauffman.utils_log.logger") as mock_logger2:
            dcs.setup_dir(self.folder)
            mock_logger2.log.assert_called_once()
            mock_logger2.log.assert_called_with(LogLevel.L1, 'Files/Sub-folders were removed from: "%s"', self.folder)
        mock_logger.log.assert_called_once()

    def test_clean_up_partial(self, mock_logger: Mock) -> None:
        dcs.setup_dir(self.folder)
        dcs.write_text_file(self.filename, "")
        dcs.setup_dir(self.subdir)
        dcs.write_text_file(self.subfile, "")
        with patch("dstauffman.utils_log.logger") as mock_logger2:
            dcs.setup_dir(self.folder, recursive=False)
            mock_logger2.log.assert_called_once()
            mock_logger2.log.assert_called_with(LogLevel.L1, 'Files/Sub-folders were removed from: "%s"', self.folder)
        self.assertEqual(mock_logger.log.call_count, 2)

    def test_fail_to_create_folder(self, mock_logger: Mock) -> None:
        pass  # TODO: write this test

    def test_fail_to_clean_folder(self, mock_logger: Mock) -> None:
        pass  # TODO: write this test

    def test_bad_name_file_ext(self, mock_logger: Mock) -> None:
        pass  # TODO: write this test

    def test_clean_up_recursively(self, mock_logger: Mock) -> None:
        dcs.setup_dir(self.subdir)
        dcs.write_text_file(self.subfile, self.text)
        with patch("dstauffman.utils_log.logger") as mock_logger2:
            dcs.setup_dir(self.folder, recursive=True)
            self.assertEqual(mock_logger2.log.call_count, 2)
            mock_logger2.log.assert_any_call(LogLevel.L1, 'Files/Sub-folders were removed from: "%s"', self.subdir)
            mock_logger2.log.assert_any_call(LogLevel.L1, 'Files/Sub-folders were removed from: "%s"', self.subdir)

    def tearDown(self) -> None:
        def _clean(self: Test_setup_dir) -> None:
            self.filename.unlink(missing_ok=True)
            self.subfile.unlink(missing_ok=True)
            with contextlib.suppress(FileNotFoundError):
                self.subdir.rmdir()
            with contextlib.suppress(FileNotFoundError):
                self.folder.rmdir()

        try:
            _clean(self)
        except {PermissionError, OSError}:  # type: ignore[misc]  # pragma: no cover
            # pause to let Windows catch up and close files
            time.sleep(1)
            # retry
            _clean(self)


# %% fix_rollover
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
@patch("dstauffman.utils_log.logger")
class Test_fix_rollover(unittest.TestCase):
    r"""
    Tests the fix_rollover function with the following cases:
        Nominal
        Matrix dim 1
        Matrix dim 2
        Log level 1
        Optional inputs
    """

    def setUp(self) -> None:
        # fmt: off
        self.data  = np.array([1, 2, 3, 4, 5, 6, 0, 1,  3,  6,  0,  6,  5, 2])
        self.data2 = np.array([], dtype=int)
        self.exp   = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 13, 12, 9])
        self.roll  = 7
        self.axis: Optional[int] = None
        # fmt: on

    def test_nominal(self, mock_logger: Mock) -> None:
        unrolled = dcs.fix_rollover(self.data, self.roll)
        np.testing.assert_array_equal(unrolled, self.exp)
        mock_logger.log.assert_any_call(LogLevel.L6, "corrected %s top to bottom rollover(s)", 2)
        mock_logger.log.assert_called_with(LogLevel.L6, "corrected %s bottom to top rollover(s)", 1)
        self.assertEqual(mock_logger.log.call_count, 2)

    def test_matrix_dim1(self, mock_logger: Mock) -> None:
        self.axis = 0
        data = np.vstack((self.data, self.data))
        exp = np.vstack((self.data, self.data))
        unrolled = dcs.fix_rollover(data, self.roll, axis=self.axis)
        np.testing.assert_array_equal(unrolled, exp)
        mock_logger.log.assert_not_called()

    def test_matrix_dim2(self, mock_logger: Mock) -> None:
        self.axis = 1
        self.data2 = np.vstack((self.data, self.data))
        exp = np.vstack((self.exp, self.exp))
        unrolled = dcs.fix_rollover(self.data2, self.roll, axis=self.axis)
        np.testing.assert_array_equal(unrolled, exp)
        mock_logger.log.assert_any_call(LogLevel.L6, "corrected %s top to bottom rollover(s)", 2)
        mock_logger.log.assert_called_with(LogLevel.L6, "corrected %s bottom to top rollover(s)", 1)
        self.assertEqual(mock_logger.log.call_count, 4)

    def test_non_integer_roll(self, mock_logger: Mock) -> None:
        exp = np.arange(0.0, 10.1, 0.1)
        roll = 3.35
        data: _N = roll * ((exp / roll) % 1.0)
        unrolled = dcs.fix_rollover(data, roll)
        np.testing.assert_array_almost_equal(unrolled, exp, decimal=12)
        mock_logger.log.assert_called_once_with(LogLevel.L6, "corrected %s top to bottom rollover(s)", 2)

    def test_signed_rollover(self, mock_logger: Mock) -> None:
        exp = np.arange(21)
        data = np.array([0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, -4])
        roll = 8
        unrolled = dcs.fix_rollover(data, roll)
        np.testing.assert_array_equal(unrolled, exp)
        mock_logger.log.assert_called_with(LogLevel.L6, "corrected %s top to bottom rollover(s)", 3)

    def test_recursive(self, mock_logger: Mock) -> None:
        pass  # TODO: figure out a test case where this actually happens.  I think the code was there for a reason?

    def test_empty(self, mock_logger: Mock) -> None:
        data = dcs.fix_rollover(np.array([]), self.roll)
        self.assertEqual(data.ndim, 1)
        self.assertEqual(data.size, 0)

    def test_bad_ndims(self, mock_logger: Mock) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.fix_rollover(np.zeros((2, 5)), self.roll)
        self.assertEqual(str(context.exception), 'Input argument "data" must be a vector.')

    def test_bad_axis(self, mock_logger: Mock) -> None:
        with self.assertRaises(AssertionError):
            dcs.fix_rollover(np.zeros((2, 3, 4)), self.roll, axis=2)
        with self.assertRaises(ValueError) as context:
            dcs.fix_rollover(np.zeros((2, 5)), self.roll, axis=2)
        self.assertEqual(str(context.exception), 'Unexpected axis: "2".')

    def test_with_nans(self, mock_logger: Mock) -> None:
        data = self.data.astype(float, copy=True)
        exp = self.exp.astype(float, copy=True)
        data[2] = np.nan
        exp[2] = np.nan
        unrolled = dcs.fix_rollover(data, self.roll)
        np.testing.assert_array_equal(unrolled, exp)
        mock_logger.log.assert_any_call(LogLevel.L6, "corrected %s top to bottom rollover(s)", 2)
        mock_logger.log.assert_called_with(LogLevel.L6, "corrected %s bottom to top rollover(s)", 1)
        self.assertEqual(mock_logger.log.call_count, 2)

    def test_high_rate_double_roll(self, mock_logger: Mock) -> None:
        time = np.arange(40.0)
        sign = 0.999 - np.mod(time / 10, 2)
        vel = np.where(sign > 0, np.mod(time, 10.0), np.mod(9.0 - time, 10.0) + 1.0)
        pos = np.cumsum(vel)
        roll = 8.25
        rolled_pos = pos - roll * np.floor(pos / roll)
        unrolled_pos = dcs.fix_rollover(rolled_pos, roll, check_accel=True, sigma=2.5)  # type: ignore[call-overload]
        np.testing.assert_array_equal(unrolled_pos, pos)
        mock_logger.log.assert_any_call(LogLevel.L6, "corrected %s top to bottom rollover(s)", 5)
        mock_logger.log.assert_any_call(LogLevel.L6, "corrected %s bottom to top rollover(s)", 3)
        mock_logger.log.assert_any_call(LogLevel.L6, "corrected %s rollovers due to acceleration checks", 4)
        self.assertGreater(mock_logger.log.call_count, 3)


# %% remove_outliers
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
@patch("dstauffman.utils_log.logger")
class Test_remove_outliers(unittest.TestCase):
    r"""
    Tests the remove_outliers function with the following cases:
        Nominal
        2D
        Inplace
        Stats
    """

    def setUp(self) -> None:
        self.x = 0.6 * np.random.rand(1000)
        self.x[5] = 1e5
        self.x[15] = 1e24
        self.x[100] = np.nan
        self.x[200] = np.nan

    def test_nominal(self, mock_logger: Mock) -> None:
        y = dcs.remove_outliers(self.x)
        self.assertEqual(self.x[0], y[0])
        self.assertEqual(self.x[5], 1e5)
        self.assertTrue(np.isnan(y[5]))
        self.assertTrue(np.isnan(y[15]))
        mock_logger.log.assert_any_call(LogLevel.L6, "Number of NaNs = %s", 2)
        mock_logger.log.assert_any_call(LogLevel.L6, "Number of outliers = %s", 2)

    def test_sigma(self, mock_logger: Mock) -> None:
        x = np.random.rand(10000)
        x[50] = 4.0
        y = dcs.remove_outliers(x, sigma=3.0)
        self.assertTrue(np.isnan(y[50]))
        y = dcs.remove_outliers(x, sigma=10.0)
        self.assertFalse(np.any(np.isnan(y)))

    def test_2d_axis(self, mock_logger: Mock) -> None:
        x = 1e-3 * np.random.rand(3, 5000)
        x[0, 10] = 1e5
        x[1, 20] = 1e3
        x[0, 30] = 1.4e-3
        for axis in [0, 1, None]:
            y = dcs.remove_outliers(x, axis=axis, sigma=3.0)  # type: ignore[call-overload]
            self.assertEqual(y.shape, (3, 5000))
            if axis in {1, None}:
                self.assertTrue(np.isnan(y[0, 10]))
                self.assertTrue(np.isnan(y[1, 20]))
            else:
                self.assertFalse(np.isnan(y[0, 10]))
                self.assertFalse(np.isnan(y[1, 20]))
            self.assertFalse(np.isnan(y[0, 30]))

    def test_inplace(self, mock_logger: Mock) -> None:
        y = dcs.remove_outliers(self.x, inplace=True)  # type: ignore[call-overload]
        self.assertIs(y, self.x)

    def test_stats(self, mock_logger: Mock) -> None:
        (y, num_replaced, rms_initial, rms_removed) = dcs.remove_outliers(self.x, return_stats=True)  # type: ignore[call-overload]
        self.assertEqual(self.x[0], y[0])
        self.assertEqual(self.x[5], 1e5)
        self.assertTrue(np.isnan(y[5]))
        self.assertTrue(np.isnan(y[15]))
        self.assertEqual(num_replaced, 2)
        self.assertGreater(rms_initial, rms_removed)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
