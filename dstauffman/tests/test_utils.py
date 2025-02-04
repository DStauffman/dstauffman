r"""
Test file for the `utils` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

# %% Imports
from __future__ import annotations

import copy
import os
import pathlib
import platform
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

from slog import capture_output

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

    nan = np.nan
else:
    from math import nan
if dcs.HAVE_SCIPY:
    from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _I = NDArray[np.int_]
    _N = NDArray[np.floating]


# %% _nan_equal
class Test__nan_equal(unittest.TestCase):
    r"""
    Tests the _nan_equal function with the following cases:
        Equal
        Not equal
        non-numpy good cases
        non-numpy bad cases
    """

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_equal(self) -> None:
        a = np.array([1, 2, np.nan])
        b = np.array([1, 2, np.nan])
        self.assertTrue(dcs.utils._nan_equal(a, b))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_not_equal(self) -> None:
        a = np.array([1, 2, np.nan])
        b = np.array([3, 2, np.nan])
        self.assertFalse(dcs.utils._nan_equal(a, b))

    def test_goods(self) -> None:
        self.assertTrue(dcs.utils._nan_equal(1, 1))
        self.assertTrue(dcs.utils._nan_equal(1, 1.0))
        self.assertTrue(dcs.utils._nan_equal(1.0, 1.0))
        self.assertTrue(dcs.utils._nan_equal([1.0, 2, nan], [1, 2, nan]))
        if dcs.HAVE_NUMPY:
            self.assertTrue(dcs.utils._nan_equal((1.0, 2, nan), [1, 2, nan]))
        self.assertTrue(dcs.utils._nan_equal({1.0, 2, nan}, {1, 2, nan}))
        self.assertTrue(dcs.utils._nan_equal("text", "text"))
        self.assertTrue(dcs.utils._nan_equal(None, None))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_goods_tol(self) -> None:
        self.assertTrue(dcs.utils._nan_equal(1, 1, tolerance=1e-6))
        self.assertTrue(dcs.utils._nan_equal(1, 1.0, tolerance=1e-6))
        self.assertTrue(dcs.utils._nan_equal(1.0, 1.0, tolerance=1e-6))
        self.assertTrue(dcs.utils._nan_equal([1.0, 2, nan], [1, 2, nan], tolerance=1e-6))
        if dcs.HAVE_NUMPY:  # pragma: no branch
            self.assertTrue(dcs.utils._nan_equal((1.0, 2, nan), [1, 2, nan], tolerance=1e-6))
        self.assertTrue(dcs.utils._nan_equal({1.0, 2, nan}, {1, 2, nan}, tolerance=1e-6))
        self.assertTrue(dcs.utils._nan_equal("text", "text", tolerance=1e-6))
        self.assertTrue(dcs.utils._nan_equal(None, None, tolerance=1e-6))

    def test_bads(self) -> None:
        self.assertFalse(dcs.utils._nan_equal(1, 1.01))
        self.assertFalse(dcs.utils._nan_equal(1, 2))
        self.assertFalse(dcs.utils._nan_equal(1.1, 1.2))
        self.assertFalse(dcs.utils._nan_equal([1, 2, 3], [3, 2, 1]))
        self.assertFalse(dcs.utils._nan_equal([1, 2, 3, 4], [1, 2, 3]))
        self.assertFalse(dcs.utils._nan_equal("text", "good"))
        self.assertFalse(dcs.utils._nan_equal("text", "longer"))
        self.assertFalse(dcs.utils._nan_equal(0, None))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_tolerance(self) -> None:
        a = np.array([1, 2.0000, np.nan])
        b = np.array([1, 2.0001, np.nan])
        self.assertFalse(dcs.utils._nan_equal(a, b))
        self.assertTrue(dcs.utils._nan_equal(a, b, tolerance=0.01))
        self.assertFalse(dcs.utils._nan_equal(a, b, tolerance=1e-12))

    def test_bad_tolerance(self) -> None:
        with patch("dstauffman.utils.HAVE_NUMPY", False):
            with self.assertRaises(ValueError):
                dcs.utils._nan_equal(0.01, 0.01, 1e-6)


# %% find_in_range
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_find_in_range(unittest.TestCase):
    r"""
    Tests the find_in_range function with the following cases:
        normal use
        age with NaNs included
    """

    def test_normal_use(self) -> None:
        value = np.array([-1, -2, 0, np.nan, 15, 34.2, np.nan, 85])
        exp   = np.array([ 1,  1, 1,      0,  1,    1,      0,  1], dtype=bool)  # fmt: skip
        valid = dcs.find_in_range(value)
        np.testing.assert_array_equal(valid, exp)

    def test_min_max(self) -> None:
        value = np.array([-1, -2, 0, np.nan, 15, 34.2, np.nan, 85])
        exp   = np.array([ 0,  0, 0,      0,  1,    1,      0,  0], dtype=bool)  # fmt: skip
        valid = dcs.find_in_range(value, min_=10, max_=40)
        np.testing.assert_array_equal(valid, exp)

    def test_mask(self) -> None:
        value = np.array([-1, -2, 0, np.nan, 15, 34.2, np.nan, 85])
        exp   = np.array([ 1,  0, 0,      0,  1,    0,      0,  0], dtype=bool)  # fmt: skip
        mask  = np.array([ 1,  0, 0,      1,  1,    0,      0,  0], dtype=bool)  # fmt: skip
        valid = dcs.find_in_range(value, mask=mask)
        np.testing.assert_array_equal(valid, exp)

    def test_bad_min(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.find_in_range(np.array(0), min_=np.nan)

    def test_bad_max(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.find_in_range([0, 1], max_=-np.inf)

    def test_inclusive(self) -> None:
        # fmt: off
        value = np.array([1, 2, 3, 4, 5])
        exp1  = np.array([0, 0, 1, 0, 0], dtype=bool)
        exp2  = np.array([0, 1, 1, 1, 0], dtype=bool)
        exp_l = np.array([0, 1, 1, 0, 0], dtype=bool)
        exp_r = np.array([0, 0, 1, 1, 0], dtype=bool)
        # fmt: on
        valid = dcs.find_in_range(value, min_=2, max_=4)
        np.testing.assert_array_equal(valid, exp1)
        valid = dcs.find_in_range(value, min_=2, max_=4, inclusive=True)
        np.testing.assert_array_equal(valid, exp2)
        valid = dcs.find_in_range(value, min_=2, max_=4, left=True)
        np.testing.assert_array_equal(valid, exp_l)
        valid = dcs.find_in_range(value, min_=2, max_=4, right=True)
        np.testing.assert_array_equal(valid, exp_r)
        valid = dcs.find_in_range(value, min_=2, max_=4, inclusive=True, left=True, right=False)
        np.testing.assert_array_equal(valid, exp2)
        valid = dcs.find_in_range(value, min_=2, max_=4, inclusive=True, left=False, right=True)
        np.testing.assert_array_equal(valid, exp2)

    def test_precision(self) -> None:
        value = np.array([1, 1.999, 3.1, 4.005, 4.012, 5])
        exp1  = np.array([0,     0,   1,     0,     0, 0], dtype=bool)  # fmt: skip
        exp2  = np.array([0,     1,   1,     1,     0, 0], dtype=bool)  # fmt: skip
        valid = dcs.find_in_range(value, min_=2.0, max_=4.0)
        np.testing.assert_array_equal(valid, exp1)
        valid = dcs.find_in_range(value, min_=2.0, max_=4.0, precision=0.01)
        np.testing.assert_array_equal(valid, exp2)

    def test_2d_array(self) -> None:
        value = np.array([[-1, -2, 0, np.nan], [15, 34.2, np.nan, 85]])
        exp   = np.array([[ 1,  1, 1,      0], [ 1,    1,      0,  1]], dtype=bool)  # fmt: skip
        valid = dcs.find_in_range(value)
        np.testing.assert_array_equal(valid, exp)

    def test_dates(self) -> None:
        value = np.datetime64("2020-09-01 00:00:00", "ns") + 10**9 * np.arange(0, 5 * 60, 30, dtype=np.int64)
        value[2] = np.datetime64("nat", "ns")
        value[7] = np.datetime64("nat", "ns")
        tmin = np.datetime64("2020-09-01 00:02:00", "ns")
        tmax = np.datetime64("2020-09-01 00:04:00", "ns")
        exp = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0], dtype=bool)
        valid = dcs.find_in_range(value, min_=tmin, max_=tmax, inclusive=True)
        np.testing.assert_array_equal(valid, exp)


# %% rms
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_rms(unittest.TestCase):
    r"""
    Tests the rms function with the following cases:
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

    def setUp(self) -> None:
        # fmt: off
        self.inputs1   = np.array([0, 1, 0.0, -1])
        self.outputs1  = np.sqrt(2)/2
        self.inputs2   = [[0, 1, 0.0, -1], [1.0, 1, 1, 1]]
        self.outputs2a = np.sqrt(3)/2
        self.outputs2b = np.array([np.sqrt(2)/2, 1, np.sqrt(2)/2, 1])
        self.outputs2c = np.array([np.sqrt(2)/2, 1])
        self.outputs2d = np.array([[np.sqrt(2)/2], [1]])
        self.inputs3   = np.hstack((self.inputs1, np.nan))
        self.inputs4   = [[0, 0.0, np.nan], [1.0, np.nan, 1]]
        self.outputs4a = np.sqrt(2)/2
        self.outputs4b = np.array([np.sqrt(2)/2, 0, 1])
        self.outputs4c = np.array([0, 1])
        # fmt: on

    def test_scalar_input(self) -> None:
        out = dcs.rms(-1.5)
        self.assertEqual(out, 1.5)

    def test_empty(self) -> None:
        out = dcs.rms([])
        self.assertTrue(np.isnan(out))

    def test_rms_series(self) -> None:
        out = dcs.rms(self.inputs1)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1a(self) -> None:
        out = dcs.rms(self.inputs1, axis=0)
        assert isinstance(out, float)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1b(self) -> None:
        with self.assertRaises(IndexError):
            dcs.rms(self.inputs1, axis=1)

    def test_axis_drop2a(self) -> None:
        out = dcs.rms(self.inputs2)
        self.assertAlmostEqual(out, self.outputs2a)

    def test_axis_drop2b(self) -> None:
        out = dcs.rms(self.inputs2, axis=0, keepdims=False)
        np.testing.assert_array_almost_equal(out, self.outputs2b)

    def test_axis_drop2c(self) -> None:
        out = dcs.rms(self.inputs2, axis=1, keepdims=False)
        np.testing.assert_array_almost_equal(out, self.outputs2c)

    def test_axis_keep(self) -> None:
        out = dcs.rms(self.inputs2, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(out, self.outputs2d)

    def test_complex_rms(self) -> None:
        out = dcs.rms(1.5j)
        self.assertEqual(out, complex(1.5, 0))

    def test_complex_conj(self) -> None:
        out = dcs.rms(np.array([1 + 1j, 1 - 1j]))
        assert isinstance(out, complex)
        self.assertAlmostEqual(out, np.sqrt(2))

    def test_with_nans(self) -> None:
        out = dcs.rms(self.inputs3, ignore_nans=False)
        self.assertTrue(np.isnan(out))

    def test_ignore_nans1(self) -> None:
        out = dcs.rms(self.inputs3, ignore_nans=True)
        assert isinstance(out, float)
        self.assertAlmostEqual(out, self.outputs1)

    def test_ignore_nans2(self) -> None:
        out = dcs.rms(self.inputs4, ignore_nans=True)
        assert isinstance(out, float)
        self.assertAlmostEqual(out, self.outputs4a)

    def test_ignore_nans3(self) -> None:
        out = dcs.rms(self.inputs4, ignore_nans=True, axis=0)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_almost_equal(out, self.outputs4b)

    def test_ignore_nans4(self) -> None:
        out = dcs.rms(self.inputs4, ignore_nans=True, axis=1)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_almost_equal(out, self.outputs4c)

    def test_all_nans(self) -> None:
        x = np.full((4, 3), np.nan)
        out: float | _N = dcs.rms(x, ignore_nans=True)
        self.assertTrue(np.isnan(out))
        out = dcs.rms(x, axis=0, ignore_nans=True)
        assert isinstance(out, np.ndarray)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (3,))
        out = dcs.rms(x, axis=1, ignore_nans=True)
        assert isinstance(out, np.ndarray)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (4,))
        out = dcs.rms(x, axis=0, ignore_nans=True, keepdims=True)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (1, 3))
        out = dcs.rms(x, axis=1, ignore_nans=True, keepdims=True)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (4, 1))


# %% rss
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_rss(unittest.TestCase):
    r"""
    Tests the rss function with the following cases:
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

    def setUp(self) -> None:
        # fmt: off
        self.inputs1   = np.array([0, 1, 0, -1])
        self.outputs1  = np.sqrt(2.0)
        self.inputs2   = [[0, 1, 0, -1], [1, 1, 1, 1]]
        self.outputs2a = np.sqrt(6.0)
        self.outputs2b = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2.0)])
        self.outputs2c = np.array([np.sqrt(2.0), 2.0])
        self.outputs2d = np.array([[np.sqrt(2.0)], [2.0]])
        self.inputs3   = np.hstack((self.inputs1, np.nan))
        self.inputs4   = [[0, 0, np.nan], [1, np.nan, 1]]
        self.outputs4a = np.sqrt(2.0)
        self.outputs4b = np.array([1.0, 0.0, 1.0])
        self.outputs4c = np.array([0.0, np.sqrt(2.0)])
        # fmt: on

    def test_scalar_input(self) -> None:
        out = dcs.rss(-1.5)
        self.assertEqual(out, 1.5)

    def test_empty(self) -> None:
        out = dcs.rss([])
        self.assertTrue(np.isnan(out))

    def test_rss_series(self) -> None:
        out = dcs.rss(self.inputs1)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1a(self) -> None:
        out = dcs.rss(self.inputs1, axis=0)
        self.assertAlmostEqual(out, self.outputs1)

    def test_axis_drop1b(self) -> None:
        with self.assertRaises(ValueError):
            dcs.rss(self.inputs1, axis=1)

    def test_axis_drop2a(self) -> None:
        out = dcs.rss(self.inputs2)
        self.assertAlmostEqual(out, self.outputs2a)

    def test_axis_drop2b(self) -> None:
        out = dcs.rss(self.inputs2, axis=0, keepdims=False)
        np.testing.assert_array_almost_equal(out, self.outputs2b)

    def test_axis_drop2c(self) -> None:
        out = dcs.rss(self.inputs2, axis=1, keepdims=False)
        np.testing.assert_array_almost_equal(out, self.outputs2c)

    def test_axis_keep(self) -> None:
        out = dcs.rss(self.inputs2, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(out, self.outputs2d)

    def test_complex_rss(self) -> None:
        out = dcs.rss(1.5j)
        self.assertEqual(out, 1.5)

    def test_complex_conj(self) -> None:
        out = dcs.rss(np.array([1 + 1j, 1 - 1j]))
        self.assertAlmostEqual(out, 2.0)

    def test_with_nans(self) -> None:
        out = dcs.rss(self.inputs3, ignore_nans=False)
        self.assertTrue(np.isnan(out))

    def test_ignore_nans1(self) -> None:
        out = dcs.rss(self.inputs3, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs1)

    def test_ignore_nans2(self) -> None:
        out = dcs.rss(self.inputs4, ignore_nans=True)
        self.assertAlmostEqual(out, self.outputs4a)

    def test_ignore_nans3(self) -> None:
        out = dcs.rss(self.inputs4, ignore_nans=True, axis=0)
        np.testing.assert_array_almost_equal(out, self.outputs4b)

    def test_ignore_nans4(self) -> None:
        out = dcs.rss(self.inputs4, ignore_nans=True, axis=1)
        np.testing.assert_array_almost_equal(out, self.outputs4c)

    def test_all_nans(self) -> None:
        x = np.full((4, 3), np.nan)
        out: float | _N = dcs.rss(x, ignore_nans=True)
        self.assertTrue(np.isnan(out))
        out = dcs.rss(x, axis=0, ignore_nans=True)
        assert isinstance(out, np.ndarray)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (3,))
        out = dcs.rss(x, axis=1, ignore_nans=True)
        assert isinstance(out, np.ndarray)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (4,))
        out = dcs.rss(x, axis=0, ignore_nans=True, keepdims=True)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (1, 3))
        out = dcs.rss(x, axis=1, ignore_nans=True, keepdims=True)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(out.shape, (4, 1))


# %% compare_two_classes
class Test_compare_two_classes(unittest.TestCase):
    r"""
    Tests the compare_two_classes function with the following cases:
        compares the same classes
        compares different classes
        compares same with names passed in
        compares with suppressed output
        compare subclasses
    """

    def setUp(self) -> None:
        self.c1 = type("Class1", (object,), {"a": 0, "b": "[1, 2, 3]", "c": "text", "e": {"key1": 1}})
        self.c2 = type("Class2", (object,), {"a": 0, "b": "[1, 2, 4]", "d": "text", "e": {"key1": 1}})
        self.names = ["Class 1", "Class 2"]
        self.c3 = type("Class3", (object,), {"a": 0, "b": "[1, 2, 3]", "c": "text", "e": self.c1})
        self.c4 = type("Class4", (object,), {"a": 0, "b": "[1, 2, 4]", "d": "text", "e": self.c2})

    def test_is_comparison(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, self.c1)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_good_comparison(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, copy.deepcopy(self.c1))
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(
            output, "b is different from c1 to c2.\nc is only in c1.\n" + 'd is only in c2.\n"c1" and "c2" are not the same.'
        )
        self.assertFalse(is_same)

    def test_names(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c2, self.c2, names=self.names)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"Class 1" and "Class 2" are the same.')
        self.assertTrue(is_same)

    def test_suppression(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, self.c2, suppress_output=True, names=self.names)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, "")
        self.assertFalse(is_same)

    def test_subclasses_match(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c3, self.c3, ignore_callables=False)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_subclasses_recurse(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False)
        output = ctx.get_output()
        ctx.close()
        self.assertFalse(is_same)
        self.assertEqual(
            output,
            "b is different from c1 to c2.\nb is different from c1.e to c2.e.\n"
            + 'c is only in c1.e.\nd is only in c2.e.\n"c1.e" and "c2.e" are not the same.\n'
            + 'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.',
        )

    def test_subclasses_norecurse(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False, compare_recursively=False)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(
            output, "b is different from c1 to c2.\n" + 'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.'
        )
        self.assertFalse(is_same)

    def test_subdict_comparison(self) -> None:
        delattr(self.c1, "b")
        delattr(self.c1, "c")
        delattr(self.c2, "b")
        delattr(self.c2, "d")
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)
        self.c1.e["key1"] += 1  # type: ignore[attr-defined]
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"c1" and "c2" are not the same.')
        self.assertFalse(is_same)

    def test_custom_dicts(self) -> None:
        delattr(self.c1, "b")
        delattr(self.c1, "c")
        delattr(self.c2, "b")
        delattr(self.c2, "d")
        self.c1.e = dcs.FixedDict()  # type: ignore[attr-defined]
        self.c1.e["key1"] = 1  # type: ignore[attr-defined]
        self.c1.e.freeze()  # type: ignore[attr-defined]
        self.c2.e = dcs.FixedDict()  # type: ignore[attr-defined]
        self.c2.e["key1"] = 1  # type: ignore[attr-defined]
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"c1.e" and "c2.e" are the same.\n"c1" and "c2" are the same.')
        self.assertTrue(is_same)
        self.c1.e["key1"] += 1  # type: ignore[attr-defined]
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c1, self.c2)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(
            output, 'key1 is different.\n"c1.e" and "c2.e" are not the same.\n' + '"c1" and "c2" are not the same.'
        )
        self.assertFalse(is_same)

    def test_mismatched_subclasses(self) -> None:
        self.c4.e = 5  # type: ignore[attr-defined]
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c3, self.c4, ignore_callables=False)
        output = ctx.get_output()
        ctx.close()
        self.assertFalse(is_same)
        self.assertEqual(
            output,
            "b is different from c1 to c2.\ne is different from c1 to c2.\n"
            + 'c is only in c1.\nd is only in c2.\n"c1" and "c2" are not the same.',
        )
        is_same = dcs.compare_two_classes(self.c4, self.c3, ignore_callables=False, suppress_output=True)
        self.assertFalse(is_same)

    def test_callables(self) -> None:
        def f(x: int) -> int:
            return x  # pragma: no cover

        def g(x: int) -> int:
            return x  # pragma: no cover

        self.c3.e = f  # type: ignore[attr-defined]
        self.c4.e = g  # type: ignore[attr-defined]
        self.c4.b = self.c3.b  # type: ignore[attr-defined]
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c4, self.c3, ignore_callables=False)
        output = ctx.get_output()
        ctx.close()
        self.assertFalse(is_same)
        self.assertEqual(
            output, "e is different from c1 to c2.\nc is only in c2.\nd is only in c1.\n" + '"c1" and "c2" are not the same.'
        )

    def test_ignore_callables(self) -> None:
        def f(x: float) -> float:
            return x  # pragma: no cover

        def g(x: float) -> float:
            return x  # pragma: no cover

        self.c3.e = f  # type: ignore[attr-defined]
        self.c4.e = g  # type: ignore[attr-defined]
        self.c4.b = self.c3.b  # type: ignore[attr-defined]
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(self.c4, self.c3, ignore_callables=True)
        output = ctx.get_output()
        ctx.close()
        self.assertFalse(is_same)
        self.assertEqual(output, 'c is only in c2.\nd is only in c1.\n"c1" and "c2" are not the same.')

    def test_two_different_lists(self) -> None:
        c1 = [1]
        c2 = [1]
        with capture_output() as ctx:
            is_same = dcs.compare_two_classes(c1, c2, ignore_callables=True)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"c1" and "c2" are the same.')
        self.assertTrue(is_same)

    def test_subset(self) -> None:
        delattr(self.c1, "b")
        delattr(self.c1, "c")
        self.c2.e["key2"] = 2  # type: ignore[attr-defined]
        with capture_output() as ctx:
            is_same1 = dcs.compare_two_classes(self.c1, self.c2, ignore_callables=True, is_subset=False, suppress_output=True)
            is_same2 = dcs.compare_two_classes(self.c1, self.c2, ignore_callables=True, is_subset=True, suppress_output=False)
            is_same3 = dcs.compare_two_classes(self.c2, self.c1, ignore_callables=True, is_subset=True, suppress_output=True)
        output = ctx.get_output()
        ctx.close()
        self.assertFalse(is_same1)
        self.assertTrue(is_same2)
        self.assertFalse(is_same3)
        self.assertEqual(output, '"c1" and "c2" are the same (subset).')

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_tolerance(self) -> None:
        self.c1.a = 0.00000001  # type: ignore[attr-defined]
        self.c2.e["key1"] = 1 + 1e-8  # type: ignore[attr-defined]
        is_same = dcs.compare_two_classes(self.c1, self.c2, tolerance=1e-4, suppress_output=True)
        self.assertFalse(is_same)
        delattr(self.c1, "b")
        delattr(self.c1, "c")
        delattr(self.c2, "b")
        delattr(self.c2, "d")
        is_same = dcs.compare_two_classes(self.c1, self.c2, tolerance=1e-4, suppress_output=True)
        self.assertTrue(is_same)


# %% compare_two_dicts
class Test_compare_two_dicts(unittest.TestCase):
    r"""
    Tests the compare_two_dicts function with the following cases:
        compares the same dicts
        compares different dicts
        compares same with names passed in
        compares with suppressed output
    """

    def setUp(self) -> None:
        self.d1 = {"a": 1, "b": 2, "c": 3, "e": {"key1": 1}}
        self.d2 = {"a": 1, "b": 5, "d": 6, "e": {"key1": 1}}
        self.names = ["Dict 1", "Dict 2"]

    def test_good_comparison(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_dicts(self.d1, self.d1)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"d1" and "d2" are the same.')
        self.assertTrue(is_same)

    def test_bad_comparison(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_dicts(self.d1, self.d2)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(
            output,
            "b is different.\n\"d1['e']\" and \"d2['e']\" are the same.\n"
            + 'c is only in d1.\nd is only in d2.\n"d1" and "d2" are not the same.',
        )
        self.assertFalse(is_same)

    def test_names(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_dicts(self.d2, self.d2, names=self.names)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, '"Dict 1" and "Dict 2" are the same.')
        self.assertTrue(is_same)

    def test_suppression(self) -> None:
        with capture_output() as ctx:
            is_same = dcs.compare_two_dicts(self.d1, self.d2, suppress_output=True, names=self.names)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, "")
        self.assertFalse(is_same)

    def test_is_subset(self) -> None:
        d1 = {"a": 1, "b": [1, 2], "e": {"key1": 1}}
        d2 = {"a": 1, "b": [1, 2], "c": "extra", "e": {"key1": 1, "key2": 2}}
        with capture_output() as ctx:
            is_same1 = dcs.compare_two_dicts(d1, d2, suppress_output=True)
            is_same2 = dcs.compare_two_dicts(d1, d2, suppress_output=False, is_subset=True)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertFalse(is_same1)
        self.assertTrue(is_same2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "\"d1['e']\" and \"d2['e']\" are the same (subset).")
        self.assertEqual(lines[1], '"d1" and "d2" are the same (subset).')

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_tolerance(self) -> None:
        self.d1["a"] = 1.00000000001
        self.d1["d"] = self.d2["d"]
        self.d2["c"] = self.d1["c"]
        self.d2["e"]["key1"] = 0.999999999998  # type: ignore[index]
        is_same = dcs.compare_two_dicts(self.d1, self.d2, tolerance=0.0001, suppress_output=True)
        self.assertTrue(is_same)


# %% read_text_file
class Test_read_text_file(unittest.TestCase):
    r"""
    Tests the read_text_file function with the following cases:
        read a file that exists
        read a file that does not exist (raise error)
    """

    folder: pathlib.Path
    contents: str
    filepath: pathlib.Path
    badpath: pathlib.Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = dcs.get_tests_dir()
        cls.contents = "Hello, World!\n"
        cls.filepath = cls.folder / "temp_file.txt"
        cls.badpath = pathlib.Path(r"AA:\non_existent_path\bad_file.txt")
        with open(cls.filepath, "wt") as file:
            file.write(cls.contents)

    def test_reading(self) -> None:
        text = dcs.read_text_file(self.filepath)
        self.assertEqual(text, self.contents)

    def test_string(self) -> None:
        text = dcs.read_text_file(str(self.filepath))
        self.assertEqual(text, self.contents)

    def test_bad_reading(self) -> None:
        with capture_output() as ctx:
            with self.assertRaises((OSError, IOError, FileNotFoundError)):
                dcs.read_text_file(self.badpath)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for reading.')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.filepath.unlink(missing_ok=True)


# %% write_text_file
class Test_write_text_file(unittest.TestCase):
    r"""
    Tests the write_text_file function with the following cases:
        write a file
        write a bad file location (raise error)
    """

    folder: pathlib.Path
    contents: str
    filepath: pathlib.Path
    badpath: pathlib.Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = dcs.get_tests_dir()
        cls.contents = "Hello, World!\n"
        cls.filepath = cls.folder / "temp_file.txt"
        cls.badpath = pathlib.Path(r"AA:\non_existent_path\bad_file.txt")

    def test_writing(self) -> None:
        dcs.write_text_file(self.filepath, self.contents)
        with open(self.filepath, "rt") as file:
            text = file.read()
        self.assertEqual(text, self.contents)

    def test_str(self) -> None:
        dcs.write_text_file(str(self.filepath), self.contents)
        with open(str(self.filepath), "rt") as file:
            text = file.read()
        self.assertEqual(text, self.contents)

    def test_bad_writing(self) -> None:
        if platform.system() != "Windows":
            return  # pragma: noc windows
        with capture_output() as ctx:
            with self.assertRaises((OSError, IOError, FileNotFoundError)):
                dcs.write_text_file(self.badpath, self.contents)
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, r'Unable to open file "AA:\non_existent_path\bad_file.txt" for writing.')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.filepath.unlink(missing_ok=True)


# %% magnitude
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_magnitude(unittest.TestCase):
    r"""
    Tests the magnitude function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.data = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.mag = np.array([1.0, 0.0, np.sqrt(2)])

    def test_nominal(self) -> None:
        mag = dcs.magnitude(self.data, axis=0)
        np.testing.assert_array_almost_equal(mag, self.mag)

    def test_bad_axis(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.magnitude(self.data, axis=2)
        self.assertEqual(str(context.exception), "axis 2 is out of bounds for array of dimension 2")

    def test_single_vector(self) -> None:
        for i in range(3):
            mag = dcs.magnitude(self.data[:, i])
            assert isinstance(mag, float)
            self.assertAlmostEqual(mag, self.mag[i])

    def test_single_vector_axis0(self) -> None:
        for i in range(3):
            mag = dcs.magnitude(self.data[:, i], axis=0)
            assert isinstance(mag, float)
            self.assertAlmostEqual(mag, self.mag[i])

    def test_single_vector_bad_axis(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.magnitude(self.data[:, 0], axis=1)
        self.assertEqual(str(context.exception), "axis 1 is out of bounds for array of dimension 1")

    def test_list(self) -> None:
        data = [self.data[:, i] for i in range(self.data.shape[1])]
        mag = dcs.magnitude(data, axis=0)
        np.testing.assert_array_almost_equal(mag, self.mag)


# %% unit
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_unit(unittest.TestCase):
    r"""
    Tests the unit function with the following cases:
        Nominal case
    """

    def setUp(self) -> None:
        self.data = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        hr2 = np.sqrt(2) / 2
        self.norm_data = np.array([[1.0, 0.0, -hr2], [0.0, 0.0, 0.0], [0.0, 0.0, hr2]])

    def test_nominal(self) -> None:
        norm_data = dcs.unit(self.data, axis=0)
        np.testing.assert_array_almost_equal(norm_data, self.norm_data)

    def test_bad_axis(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.unit(self.data, axis=2)
        self.assertEqual(str(context.exception), "axis 2 is out of bounds for array of dimension 2")

    def test_single_vector(self) -> None:
        for i in range(3):
            norm_data = dcs.unit(self.data[:, i])
            np.testing.assert_array_almost_equal(norm_data, self.norm_data[:, i])

    def test_single_vector_axis0(self) -> None:
        for i in range(3):
            norm_data = dcs.unit(self.data[:, i], axis=0)
            np.testing.assert_array_almost_equal(norm_data, self.norm_data[:, i])

    def test_single_vector_bad_axis(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.unit(self.data[:, 0], axis=1)
        self.assertEqual(str(context.exception), "axis 1 is out of bounds for array of dimension 1")

    def test_list(self) -> None:
        data = [self.data[:, i] for i in range(self.data.shape[1])]
        norm_data = dcs.unit(data, axis=0)
        np.testing.assert_array_almost_equal(norm_data, self.norm_data)


# %% modd
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_modd(unittest.TestCase):
    r"""
    Tests the modd function with the following cases:
        Nominal
        Scalar
        List (x2)
        Modify in-place
    """

    def setUp(self) -> None:
        self.x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        self.y = np.array([ 4,  1,  2,  3, 4, 1, 2, 3, 4])  # fmt: skip
        self.mod = 4

    def test_nominal(self) -> None:
        y = dcs.modd(self.x, self.mod)
        np.testing.assert_array_equal(y, self.y)  # type: ignore[arg-type]

    def test_scalar(self) -> None:
        y = dcs.modd(4, 4)
        self.assertEqual(y, 4)

    def test_list1(self) -> None:
        y = dcs.modd([2, 4], 4)
        np.testing.assert_array_equal(y, np.array([2, 4]))  # type: ignore[arg-type]

    def test_list2(self) -> None:
        y = dcs.modd(4, [3, 4])
        np.testing.assert_array_equal(y, np.array([1, 4]))  # type: ignore[arg-type]

    def test_modify_inplace(self) -> None:
        out = np.zeros(self.x.shape, dtype=int)
        dcs.modd(self.x, self.mod, out)
        np.testing.assert_array_equal(out, self.y)


# %% is_np_int
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

    def test_int(self) -> None:
        self.assertTrue(dcs.is_np_int(10))

    def test_float(self) -> None:
        self.assertFalse(dcs.is_np_int(10.0))

    def test_large_int(self) -> None:
        self.assertTrue(dcs.is_np_int(2**62))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_np_int(self) -> None:
        self.assertTrue(dcs.is_np_int(np.array([1, 2, 3])))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_np_float(self) -> None:
        self.assertFalse(dcs.is_np_int(np.array([2.0, np.pi])))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_np_large_int(self) -> None:
        self.assertTrue(dcs.is_np_int(np.array(2**62)))

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_np_uint(self) -> None:
        self.assertTrue(dcs.is_np_int(np.array([1, 2, 3], dtype=np.uint32)))


# %% np_digitize
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
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

    def setUp(self) -> None:
        self.x = np.array([1.1, 2.2, 3.3, 3.3, 5.5, 10])
        self.bins = np.array([-1, 2, 3.1, 4, 4.4, 6, 20])
        self.out = np.array([0, 1, 2, 2, 4, 5], dtype=int)

    def test_nominal(self) -> None:
        out = dcs.np_digitize(self.x, self.bins)
        np.testing.assert_array_equal(out, self.out)

    def test_bad_min(self) -> None:
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([-5, 5]), self.bins)

    def test_bad_max(self) -> None:
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([5, 25]), self.bins)

    def test_bad_both(self) -> None:
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([25, -5, 5]), self.bins)

    def test_empty(self) -> None:
        out = dcs.np_digitize(np.array([]), self.bins)
        self.assertEqual(out.size, 0)

    def test_right(self) -> None:
        out = dcs.np_digitize(self.x, self.bins, right=True)
        np.testing.assert_array_equal(out, self.out)

    def test_bad_right(self) -> None:
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([5, 25]), self.bins, right=True)

    def test_for_nans(self) -> None:
        with self.assertRaises(ValueError):
            dcs.np_digitize(np.array([1, 10, np.nan]), self.bins)


# %% histcounts
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_histcounts(unittest.TestCase):
    r"""
    Tests the histcounts function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.x = np.array([0.2, 6.4, 3.0, 1.6, 0.5])
        self.bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        self.expected = np.array([2, 1, 1, 1])

    def test_nominal(self) -> None:
        hist = dcs.histcounts(self.x, self.bins)
        np.testing.assert_array_equal(hist, self.expected)

    def test_right(self) -> None:
        x = np.array([1, 1, 2, 2, 2])
        bins = np.array([0, 1, 2, 3])
        hist = dcs.histcounts(x, bins, right=False)
        np.testing.assert_array_equal(hist, np.array([0, 2, 3]))
        hist2 = dcs.histcounts(x, bins, right=True)
        np.testing.assert_array_equal(hist2, np.array([2, 3, 0]))

    def test_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            dcs.histcounts(self.x, np.array([100, 1000]))


# %% full_print
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_full_print(unittest.TestCase):
    r"""
    Tests the full_print function with the following cases:
        Nominal
        Small (x2)
    """

    @staticmethod
    def _norm_output(lines: list[str]) -> list[str]:
        out = []
        for line in lines:
            # normalize whitespace
            temp = " ".join(line.strip().split())
            # get rid of spaces near brackets
            temp = temp.replace("[ ", "[")
            temp = temp.replace(" ]", "]")
            out.append(temp)
        return out

    def setUp(self) -> None:
        self.x = np.zeros((10, 5))
        self.x[3, :] = 1.23
        self.x_print = [
            "[[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "...",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]]",
        ]
        self.x_full = [
            "[[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[1.23 1.23 1.23 1.23 1.23]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]",
            "[0. 0. 0. 0. 0.]]",
        ]
        # explicitly set default threshold to 10 (since consoles sometimes use 1000 instead)
        self.orig = np.get_printoptions()
        np.set_printoptions(threshold=10)

    def test_nominal(self) -> None:
        with capture_output() as ctx:
            print(self.x)
        lines = ctx.get_output().split("\n")
        ctx.close()
        # normalize whitespace
        lines = self._norm_output(lines)
        self.assertEqual(lines, self.x_print)
        with capture_output() as ctx:
            with dcs.full_print():
                print(self.x)
        lines = ctx.get_output().split("\n")
        # normalize whitespace
        lines = self._norm_output(lines)
        ctx.close()
        self.assertEqual(lines, self.x_full)

    def test_small1(self) -> None:
        with capture_output() as ctx:
            with dcs.full_print():
                print(np.array(0))
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, "0")

    def test_small2(self) -> None:
        with capture_output() as ctx:
            with dcs.full_print():
                print(np.array([1.35, 1.58]))
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, "[1.35 1.58]")

    def test_keyword_arguments(self) -> None:
        with capture_output() as ctx:
            with dcs.full_print(formatter={"float": lambda x: "{:.1f}".format(x)}):
                print(np.array([1.2345, 1001.555]))
        output = ctx.get_output()
        ctx.close()
        self.assertEqual(output, "[1.2 1001.6]")

    def tearDown(self) -> None:
        # restore the print_options
        np.set_printoptions(**self.orig)


# %% line_wrap
class Test_line_wrap(unittest.TestCase):
    r"""
    Tests the line_wrap function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.text = ("lots of repeated words " * 4).strip()
        self.wrap = 40
        self.min_wrap = 0
        self.indent = 4
        self.out = [
            "lots of repeated words lots of \\",
            "    repeated words lots of repeated \\",
            "    words lots of repeated words",
        ]

    def test_str(self) -> None:
        out = dcs.line_wrap(self.text, self.wrap, self.min_wrap, self.indent)
        self.assertEqual(out, "\n".join(self.out))

    def test_list(self) -> None:
        out = dcs.line_wrap([self.text], self.wrap, self.min_wrap, self.indent)
        self.assertEqual(out, self.out)

    def test_list2(self) -> None:
        out = dcs.line_wrap(3 * ["aaaaaaaaaa bbbbbbbbbb cccccccccc"], wrap=25, min_wrap=15, indent=2)
        self.assertEqual(out, 3 * ["aaaaaaaaaa bbbbbbbbbb \\", "  cccccccccc"])

    def test_min_wrap(self) -> None:
        out = dcs.line_wrap("aaaaaaaaaaaaaaaaaaaa bbbbbbbbbb", 25, 18, 0)
        self.assertEqual(out, "aaaaaaaaaaaaaaaaaaaa \\\nbbbbbbbbbb")

    def test_min_wrap2(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.line_wrap("aaaaaaaaaaaaaaaaaaaa bbbbbbbbbb", 25, 22, 0)
        self.assertEqual(str(context.exception), 'The specified min_wrap:wrap of "22:25" was too small.')


# %% combine_per_year
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
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

    def setUp(self) -> None:
        self.time = np.arange(120)
        self.data = self.time // 12
        self.data2 = np.arange(10)
        self.data3 = np.column_stack((self.data, self.data))
        self.data4 = np.column_stack((self.data2, self.data2))
        self.data5 = np.full(120, np.nan)
        self.func1 = np.nanmean
        self.func2 = np.nansum

    def test_1D(self) -> None:
        data2 = dcs.combine_per_year(self.data, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data2)  # type: ignore[arg-type]

    def test_2D(self) -> None:
        data2 = dcs.combine_per_year(self.data3, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data4)  # type: ignore[arg-type]

    def test_data_is_none(self) -> None:
        data2 = dcs.combine_per_year(None, func=self.func1)
        self.assertEqual(data2, None)

    def test_data_is_all_nan(self) -> None:
        data2 = dcs.combine_per_year(self.data5, func=self.func1)
        self.assertTrue(len(data2) == 10)
        self.assertTrue(np.all(np.isnan(data2)))

    def test_non12_months1d(self) -> None:
        data = np.arange(125) // 12
        data2 = dcs.combine_per_year(data, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data2)  # type: ignore[arg-type]

    def test_non12_months2d(self) -> None:
        data = np.arange(125) // 12
        data3 = np.column_stack((data, data))
        data2 = dcs.combine_per_year(data3, func=self.func1)
        np.testing.assert_array_almost_equal(data2, self.data4)  # type: ignore[arg-type]

    def test_other_funcs(self) -> None:
        data2a = dcs.combine_per_year(self.data, func=self.func1)
        data2b = dcs.combine_per_year(self.data, func=self.func2)
        np.testing.assert_array_almost_equal(12 * data2a, data2b)  # type: ignore[arg-type, operator]

    def test_bad_func1(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.combine_per_year(self.data)  # type: ignore[call-overload]

    def test_bad_func2(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.combine_per_year(self.data, func=1.5)  # type: ignore[call-overload]


# %% execute
class Test_execute(unittest.TestCase):
    r"""
    Tests the execute function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% execute_wrapper
class Test_execute_wrapper(unittest.TestCase):
    r"""
    Tests the execute_wrapper function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% get_env_var
class Test_get_env_var(unittest.TestCase):
    r"""
    Tests the get_env_var function with the following cases:
        Valid key
        Unknown key
        Default key
        Not allowed
    """

    def test_valid(self) -> None:
        home = dcs.get_env_var("HOME")
        self.assertTrue(bool(home))

    def test_bad_key(self) -> None:
        with self.assertRaises(KeyError):
            dcs.get_env_var("Nonexisting_environment_key_name")

    def test_default_key(self) -> None:
        key = dcs.get_env_var("Nonexisting_environment_key_name", default="test")
        self.assertEqual(key, "test")

    def test_not_allowed(self) -> None:
        with patch("dstauffman.utils._ALLOWED_ENVS", {"user", "username"}):
            with self.assertRaises(KeyError):
                dcs.get_env_var("HOME")


# %% get_username
class Test_get_username(unittest.TestCase):
    r"""
    Tests the get_username function with the following cases:
        Windows
        Unix
    """

    def test_windows(self) -> None:
        with patch("dstauffman.utils.IS_WINDOWS", True):
            with patch.dict(os.environ, {"USER": "name", "USERNAME": "name_two"}):
                username = dcs.get_username()
        self.assertEqual(username, "name_two")

    def test_unix(self) -> None:
        with patch("dstauffman.utils.IS_WINDOWS", False):
            with patch.dict(os.environ, {"USER": "name", "USERNAME": "name_two"}):
                username = dcs.get_username()
        self.assertEqual(username, "name")


# %% is_datetime
class Test_is_datetime(unittest.TestCase):
    r"""
    Tests the is_datetime function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% intersect
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_intersect(unittest.TestCase):
    r"""
    Tests the intersect function with the following cases:
        Nominal
        Floats
        Assume unique
    """

    def test_nominal(self) -> None:
        a = np.array([1, 2, 4, 4, 6], dtype=int)
        b = np.array([0, 8, 2, 2, 5, 8, 6, 8, 8], dtype=int)
        (c, ia, ib) = dcs.intersect(a, b, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([2, 6], dtype=int))
        np.testing.assert_array_equal(ia, np.array([1, 4], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 6], dtype=int))

    def test_floats(self) -> None:
        a = np.array([1, 2.5, 4, 6])
        b = np.array([0, 8, 2.5, 4, 6])
        (c, ia, ib) = dcs.intersect(a, b, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6]))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))

    def test_unique(self) -> None:
        a = np.array([1, 2.5, 4, 6])
        b = np.array([0, 8, 2.5, 4, 6])
        (c, ia, ib) = dcs.intersect(a, b, assume_unique=True, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6]))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))
        (c, ia, ib) = dcs.intersect(a, b, tolerance=1e-7, assume_unique=True, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([2.5, 4, 6]))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(ib, np.array([2, 3, 4], dtype=int))

    def test_no_indices(self) -> None:
        a = np.array([1, 2, 4, 4, 6], dtype=int)
        b = np.array([0, 8, 2, 2, 5, 8, 6, 8, 8], dtype=int)
        c = dcs.intersect(a, b)
        np.testing.assert_array_equal(c, np.array([2, 6], dtype=int))

    def test_tolerance(self) -> None:
        a = np.array([1.0, 2.0, 3.1, 3.9, 4.0, 6.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0.12, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([2.0, 3.1, 3.9, 4.0]))
        np.testing.assert_array_equal(ia, np.array([1, 2, 3, 4], dtype=int))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2], dtype=int))

    def test_tolerance_no_ix(self) -> None:
        a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        b = np.array([1.01, 2.02, 3.03, 4.04, 5.05, 6.06, 7.07, 8.08, 9.09])
        c = dcs.intersect(a, b, tolerance=0.055, return_indices=False)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([1.0, 3.0, 5.0]))
        c2 = dcs.intersect(b, a, tolerance=0.055, return_indices=False)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c2, np.array([1.01, 3.03, 5.05]))

    def test_scalars(self) -> None:
        a = 5
        b = 4.9
        c = dcs.intersect(a, b, tolerance=0.5)  # type: ignore[call-overload]
        self.assertEqual(c, 5)

    def test_int(self) -> None:
        a = np.array([0, 4, 10, 20, 30, -40, 30])
        b = np.array([1, 5, 7, 31, -10, -40])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([-40]))
        np.testing.assert_array_equal(ia, np.array([5]))
        np.testing.assert_array_equal(ib, np.array([5]))

    def test_int_even_tol(self) -> None:
        a = np.array([0, 4, 10, 20, 30, -40, 30])
        b = np.array([1, 5, 7, 31, -10, -40])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=2, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 30]))
        np.testing.assert_array_equal(ia, np.array([0, 1, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 3, 5]))

    def test_int_odd_tol(self) -> None:
        a = np.array([0, 4, 10, 20, 30, -40, 30])
        b = np.array([1, 5, 7, 31, -10, -40])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=3, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 10, 30]))
        np.testing.assert_array_equal(ia, np.array([0, 1, 2, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2, 3, 5]))

    def test_int64(self) -> None:
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([-40], dtype=np.int64) + t_offset)
        np.testing.assert_array_equal(ia, np.array([5]))
        np.testing.assert_array_equal(ib, np.array([5]))

    def test_int64_even_tol(self) -> None:
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=2, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 30], dtype=np.int64) + t_offset)
        np.testing.assert_array_equal(ia, np.array([0, 1, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 3, 5]))

    def test_int64_odd_tol(self) -> None:
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=3, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(ia, np.array([0, 1, 2, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2, 3, 5]))
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 10, 30], dtype=np.int64) + t_offset)

    def test_npint64_tol(self) -> None:
        t_offset = 2**62
        a = np.array([0, 4, 10, 20, 30, -40, 30], dtype=np.int64) + t_offset
        b = np.array([1, 5, 7, 31, -10, -40], dtype=np.int64) + t_offset
        (c, ia, ib) = dcs.intersect(a, b, tolerance=np.array(3), return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(ia, np.array([0, 1, 2, 4, 5]))
        np.testing.assert_array_equal(ib, np.array([0, 1, 2, 3, 5]))
        np.testing.assert_array_equal(c, np.array([-40, 0, 4, 10, 30], dtype=np.int64) + t_offset)

    def test_empty(self) -> None:
        a = np.array([], dtype=int)
        b = np.array([1, 2, 3, 4])
        c = dcs.intersect(a, b, tolerance=0.1)  # type: ignore[call-overload]
        self.assertEqual(len(c), 0)

    def test_datetimes(self) -> None:
        date_zero = np.datetime64("2020-06-01 00:00:00", "ms")
        dt = np.arange(0, 11000, 1000).astype("timedelta64[ms]")
        a = date_zero + dt
        dt[3] += 5
        dt[5] -= 30
        b = date_zero + dt
        # no tolerance
        exp = np.array([0, 1, 2, 4, 6, 7, 8, 9, 10])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=0, return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, a[exp])
        np.testing.assert_array_equal(ia, exp)
        np.testing.assert_array_equal(ib, exp)
        # with tolerance
        exp = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
        (c, ia, ib) = dcs.intersect(a, b, tolerance=np.timedelta64(10, "ms"), return_indices=True)  # type: ignore[call-overload]
        np.testing.assert_array_equal(c, a[exp])
        np.testing.assert_array_equal(ia, exp)
        np.testing.assert_array_equal(ib, exp)

    def test_quant_effects(self) -> None:
        # Note: this is an example where something slightly outside of the tolerance can still be matched.
        # TODO: is this really the expected answer or just a limit of this algorithm?
        c = dcs.intersect(225409, 225449, tolerance=30)  # type: ignore[call-overload]
        self.assertEqual(c, 225409)


# %% issorted
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_issorted(unittest.TestCase):
    r"""
    Tests the issorted function with the following cases:
        Sorted
        Not sorted
        Reverse sorted (x2)
        Lists
    """

    def test_sorted(self) -> None:
        x = np.array([1, 3, 3, 5, 7])
        self.assertTrue(dcs.issorted(x))

    def test_not_sorted(self) -> None:
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(dcs.issorted(x))

    def test_reverse_sorted(self) -> None:
        x = np.array([4, np.pi, 1.0, -1.0])
        self.assertFalse(dcs.issorted(x))
        self.assertTrue(dcs.issorted(x, descend=True))

    def test_lists(self) -> None:
        x = [-np.inf, 0, 1, np.pi, 5, np.inf]
        self.assertTrue(dcs.issorted(x))
        self.assertFalse(dcs.issorted(x, descend=True))


# %% zero_order_hold
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_zero_order_hold(unittest.TestCase):
    r"""
    Tests the zero_order_hold function with the following cases:
        Subsample high rate
        Supersample low rate
        xp Not sorted
        x not sorted
        Left extrapolation
        Lists instead of arrays
        Return indices

    Notes
    -----
    #.  Uses scipy.interpolate.interp1d as the gold standard (but it's slower)
    """

    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_subsample(self) -> None:
        xp = np.linspace(0.0, 100 * np.pi, 500000)
        yp = np.sin(2 * np.pi * xp)
        x = np.arange(0.0, 350.0, 0.1)
        func = interp1d(xp, yp, kind="zero", fill_value="extrapolate", assume_sorted=True)
        y_exp = func(x)
        y = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)
        y = dcs.zero_order_hold(x, xp, yp, assume_sorted=True)
        np.testing.assert_array_equal(y, y_exp)

    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_supersample(self) -> None:
        xp = np.array([0.0, 5000.0, 10000.0, 86400.0])
        yp = np.array([0, 1, -2, 0])
        x  = np.arange(0.0, 86400.0)  # fmt: skip
        func = interp1d(xp, yp, kind="zero", fill_value="extrapolate", assume_sorted=True)
        y_exp = func(x)
        y = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)
        y = dcs.zero_order_hold(x, xp, yp, assume_sorted=True)
        np.testing.assert_array_equal(y, y_exp)

    # fmt: off
    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_xp_not_sorted(self) -> None:
        xp    = np.array([0, 10, 5, 15])
        yp    = np.array([0, 1, -2, 3])
        x     = np.array([10, 2, 14,  6,  8, 10, 4, 14, 0, 16])
        y_exp = np.array([ 1, 0,  1, -2, -2,  1, 0,  1, 0,  3])
        y     = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)

    def test_x_not_sorted(self) -> None:
        xp    = np.array([0, 5, 10, 15])
        yp    = np.array([0, -2, 1, 3])
        x     = np.array([10, 2, 14,  6,  8, 10, 4, 14, 0, 16])
        y_exp = np.array([ 1, 0,  1, -2, -2,  1, 0,  1, 0,  3])
        y     = dcs.zero_order_hold(x, xp, yp)
        np.testing.assert_array_equal(y, y_exp)

    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_left_end(self) -> None:
        xp    = np.array([0, 5, 10, 15, 4])
        yp    = np.array([0, 1, -2, 3, 0])
        x     = np.array([-4, -2, 0, 2, 4, 6])
        y_exp = np.array([-5, -5, 0, 0, 0, 1])
        y     = dcs.zero_order_hold(x, xp, yp, left=-5)
        np.testing.assert_array_equal(y, y_exp)

    def test_lists(self) -> None:
        xp    = [0, 5, 10, 15]
        yp    = [0, 1, 2, 3]
        x     = [-4, -2, 0, 2, 4, 6, 20]
        y_exp = [-1, -1, 0, 0, 0, 1, 3]
        y     = dcs.zero_order_hold(x, xp, yp, left=-1)
        np.testing.assert_array_equal(y, y_exp)

    def test_bools(self) -> None:
        xp    = np.array([1, 3, 4, 6, 8, 12], dtype=int)
        yp    = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
        x     = np.arange(15, dtype=int)
        y_exp = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        y     = dcs.zero_order_hold(x, xp, yp, left=False)
        self.assertEqual(y.dtype, bool)
        np.testing.assert_array_equal(y, y_exp)

    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_bools_unsorted(self) -> None:
        xp    = np.array([1, 3, 6, 4, 8, 12], dtype=int)
        yp    = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
        x     = np.arange(15, dtype=int)
        y_exp = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        y     = dcs.zero_order_hold(x, xp, yp, left=False)
        self.assertEqual(y.dtype, bool)
        np.testing.assert_array_equal(y, y_exp)

    def test_indices(self) -> None:
        xp      = np.array([0, 5, 10, 15])
        yp      = np.array([0, 1, 2, 3])
        x       = np.array([-4, -2, 0, 2, 4, 6, 20])
        y_exp   = np.array([np.nan, np.nan, 0, 0, 0, 1, 3])
        ix_exp  = np.array([None, None, 0, 0, 0, 1, 3])
        (y, ix) = dcs.zero_order_hold(x, xp, yp, left=np.nan, return_indices=True)
        np.testing.assert_array_equal(y, y_exp)
        np.testing.assert_array_equal(ix, ix_exp)

    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_indices_not_sorted(self) -> None:
        xp      = np.array([0, 10, 5, 15])
        yp      = np.array([0, 1, 2, 3])
        x       = np.array([-4, -2, 0, 2, 4, 6, 20])
        with self.assertRaises(RuntimeError) as err:
            dcs.zero_order_hold(x, xp, yp, return_indices=True)
        self.assertEqual(str(err.exception), "Data must be sorted in order to ask for indices.")

    def test_missing_scipy(self) -> None:
        xp    = np.array([0, 5, 10, 15, 4])
        yp    = np.array([0, 1, -2, 3, 0])
        x     = np.array([-4, -2, 0, 2, 4, 6])
        with self.assertRaises(RuntimeError) as err:
            with patch("dstauffman.utils.HAVE_SCIPY", False):
                dcs.zero_order_hold(x, xp, yp)
        self.assertEqual(str(err.exception), "You must have scipy available to run this.")
    # fmt: on


# %% linear_interp
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_linear_interp(unittest.TestCase):
    r"""
    Tests the linear_interp function with the following cases:
        Sorted
        Not-sorted
        Extrapolation
    """

    def setUp(self) -> None:
        self.x1 = np.arange(1.0, 5.1, 0.1)  # 40 pts
        self.x2 = np.arange(5.2, 9.2, 0.2)  # 20 pts
        self.x3 = np.arange(9.5, 14.5, 0.5)  # 10 pts
        self.m1 = 2.0
        self.m2 = -1.0
        self.m3 = 5.0
        self.b0 = -8.0
        self.y1 = self.m1 * (self.x1 - self.x1[0]) + self.b0 + self.m1
        self.y2 = self.m2 * (self.x2 - self.x2[0]) + self.y1[-1] + 0.2 * self.m2
        self.y3 = self.m3 * (self.x3 - self.x3[0]) + self.y2[-1] + 0.5 * self.m3
        self.x = np.hstack((self.x1, self.x2, self.x3))
        self.y = np.hstack([self.y1, self.y2, self.y3])
        self.xp = np.array([1.0, 5.0, 9.0, 14.0])
        self.yp: _N
        self.ix: _I
        (self.yp, self.ix, _) = dcs.intersect(self.x, self.xp, return_indices=True, tolerance=1e-10)  # type: ignore[call-overload]
        self.yp = self.y[self.ix]

    def test_sorted(self) -> None:
        y = dcs.linear_interp(self.x, self.xp, self.yp)
        np.testing.assert_array_almost_equal(y, self.y, 12)

    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_unsorted(self) -> None:
        ix = np.arange(self.xp.size)
        while dcs.issorted(ix):
            np.random.shuffle(ix)
        y = dcs.linear_interp(self.x, self.xp[ix], self.yp[ix], assume_sorted=False)
        np.testing.assert_array_almost_equal(y, self.y, 12)

    def test_extrapolate_numpy(self) -> None:
        xp = self.xp.copy()
        yp = self.yp.copy()
        xp[0] = self.x[5]
        yp[0] = self.y[5]
        xp[-1] = self.x[-5]
        yp[-1] = self.y[-5]
        exp = self.y.copy()
        exp[0:5] = 0.5
        exp[-4:] = 1000.0
        with self.assertRaises(ValueError) as context:
            dcs.linear_interp(self.x, xp, yp, left=0.5, right=1000.0)
        self.assertEqual(str(context.exception), "Desired points outside given xp array and extrapolation is False")
        y = dcs.linear_interp(self.x, xp, yp, left=0.5, right=1000.0, extrapolate=True)
        np.testing.assert_array_almost_equal(y, exp, 12)

    @unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
    def test_extrapolation_scipy(self) -> None:
        xp = self.xp.copy()
        yp = self.yp.copy()
        xp[0] = self.x[5]
        yp[0] = self.y[5]
        xp[-1] = self.x[-5]
        yp[-1] = self.y[-5]
        exp = self.y.copy()
        exp[0:5] = 0.7
        exp[-4:] = 750.0
        ix = np.arange(xp.size)
        while dcs.issorted(ix):
            np.random.shuffle(ix)
        with self.assertRaises(ValueError) as context:
            dcs.linear_interp(self.x, xp[ix], yp[ix], assume_sorted=False, extrapolate=False)
        text = str(context.exception)
        self.assertTrue(text.startswith("A value ") and "in x_new is below the interpolation range" in text)
        y = dcs.linear_interp(self.x, xp[ix], yp[ix], assume_sorted=False, extrapolate=True)
        np.testing.assert_array_almost_equal(y, self.y, 12)
        y = dcs.linear_interp(self.x, xp[ix], yp[ix], left=0.7, right=750.0, assume_sorted=False, extrapolate=True)
        np.testing.assert_array_almost_equal(y, exp, 12)


# %% linear_lowpass_interp
@unittest.skipIf(not dcs.HAVE_SCIPY, "Skipping due to missing scipy dependency.")
class Test_linear_lowpass_interp(unittest.TestCase):
    r"""
    Tests the linear_lowpass_interp function with the following cases:
        Nominal
        Alternative filter parameters
        Extrapolation
    """

    def setUp(self) -> None:
        self.x = np.arange(0, 10.1, 0.1)
        self.xp = np.array([0.0, 5.0, 10.0])
        self.yp = np.array([0.0, 5.0, 0.0])
        self.y = np.hstack([np.arange(0.0, 5.0, 0.1), np.arange(5.0, -0.1, -0.1)])

    def test_nominal(self) -> None:
        y = dcs.linear_lowpass_interp(self.x, self.xp, self.yp)
        # test that there was no overshoot at the turn-around
        self.assertTrue(np.all(y < 5.0))

    def test_extrapolation(self) -> None:
        self.xp[-1] = 8.0
        with self.assertRaises(ValueError) as context:
            dcs.linear_lowpass_interp(self.x, self.xp, self.yp)
        text = str(context.exception)
        self.assertTrue(text.startswith("A value ") and "in x_new is above the interpolation range" in text)
        y = dcs.linear_lowpass_interp(self.x, self.xp, self.yp, extrapolate=True)
        self.assertTrue(np.all(y < 5.0))

    def test_filter_parameters(self) -> None:
        y = dcs.linear_lowpass_interp(self.x, self.xp, self.yp, filt_order=4, filt_freq=0.02, filt_samp=0.1)
        self.assertTrue(np.all(y < 5.0))


# %% drop_following_time
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_drop_following_time(unittest.TestCase):
    r"""
    Tests the drop_following_time function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.times = np.arange(10, 60)
        self.ix_drop_start = np.array([5, 15, 17, 25])
        self.drop_starts = self.times[self.ix_drop_start]
        self.dt_drop = 3
        self.exp = np.zeros(self.times.shape, dtype=bool)
        self.exp[np.array([5, 6, 7, 15, 16, 17, 18, 19, 25, 26, 27])] = True
        self.exp_rev = np.zeros(self.times.shape, dtype=bool)
        self.exp_rev[np.array([3, 4, 5, 13, 14, 15, 16, 17, 23, 24, 25])] = True

    def test_nominal(self) -> None:
        drop_mask = dcs.drop_following_time(self.times, self.drop_starts, self.dt_drop)
        np.testing.assert_array_equal(drop_mask, self.exp)

    def test_reversed(self) -> None:
        drop_mask = dcs.drop_following_time(self.times, self.drop_starts, self.dt_drop, reverse=True)
        np.testing.assert_array_equal(drop_mask, self.exp_rev)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
