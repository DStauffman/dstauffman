r"""
Test file for the `numba` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in January 2021.
"""

#%% Imports
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np
    pi = np.pi
    inf = np.inf
else:
    from math import inf, pi

#%% _reduce_shape
class Test__reduce_shape(unittest.TestCase):
    r"""
    Tests the _reduce_shape function with the following cases:
        Nominal
        Bad axis
    """
    def test_nominal(self) -> None:
        shape = (1, 2, 3, 4, 5)
        for axis in range(5):
            out = dcs.numba._reduce_shape(shape, axis)
            expected = shape[:axis] + shape[axis+1:]
            self.assertEqual(tuple(out), expected)

    def test_bad_axis(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.numba._reduce_shape((1, 2), 2)

#%% issorted_ascend
@unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_issorted_ascend(unittest.TestCase):
    r"""
    Tests the issorted_ascend function with the following cases:
        Sorted
        Not sorted
        Lists
    """
    def test_sorted(self) -> None:
        x = np.array([1, 3, 3, 5, 7])
        self.assertTrue(dcs.issorted_ascend(x))
        x2 = np.array([-1, 1, pi, 4])
        self.assertTrue(dcs.issorted_ascend(x2))

    def test_not_sorted(self) -> None:
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(dcs.issorted_ascend(x))

    def test_reverse(self) -> None:
        x = np.array([inf, 4, pi, 1., -1., -inf])
        self.assertFalse(dcs.issorted_ascend(x))

#%% issorted_descend
@unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_issorted_descend(unittest.TestCase):
    r"""
    Tests the issorted_descend function with the following cases:
        Sorted
        Not sorted
        Lists
    """
    def test_sorted(self) -> None:
        x = np.array([7, 5, 3, 3, 1])
        self.assertTrue(dcs.issorted_descend(x))
        x2 = np.array([inf, 4., pi, 1, -1, -inf])
        self.assertTrue(dcs.issorted_descend(x2))

    def test_not_sorted(self) -> None:
        x = np.array([1, 4, 3, 5, 7])
        self.assertFalse(dcs.issorted_descend(x))

    def test_reverse(self) -> None:
        x = np.array([-inf, -1, 1, pi, 4., inf])
        self.assertFalse(dcs.issorted_descend(x))

#%% np_all_axis0, np_all_axis1, np_any_axis0, np_any_axis1
@unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_np_all_axis0(unittest.TestCase):
    r"""
    Tests the np_all_axis0 function with the following cases:
        Nominal
        1D
        Scalar
        0D
        3D
    """
    def test_nominal(self) -> None:
        x = np.array([[True, True, False, False], [True, False, True, False]], dtype=bool)
        np.testing.assert_array_equal(dcs.np_all_axis0(x), np.all(x, axis=0))
        np.testing.assert_array_equal(dcs.np_all_axis1(x), np.all(x, axis=1))
        np.testing.assert_array_equal(dcs.np_any_axis0(x), np.any(x, axis=0))
        np.testing.assert_array_equal(dcs.np_any_axis1(x), np.any(x, axis=1))

    def test_1d(self) -> None:
        self.assertTrue(dcs.np_all_axis0(np.array([True, True], dtype=bool)))
        self.assertFalse(dcs.np_all_axis0(np.array([True, False], dtype=bool)))
        self.assertTrue(dcs.np_any_axis0(np.array([True, False], dtype=bool)))
        self.assertFalse(dcs.np_any_axis0(np.array([False, False], dtype=bool)))

    def test_scalar(self) -> None:
        self.assertTrue(dcs.np_all_axis0(np.array([[True]], dtype=bool)))
        self.assertFalse(dcs.np_all_axis0(np.array([[False]], dtype=bool)))
        self.assertTrue(dcs.np_all_axis1(np.array([[True]], dtype=bool)))
        self.assertFalse(dcs.np_all_axis1(np.array([[False]], dtype=bool)))
        self.assertTrue(dcs.np_any_axis0(np.array([[True]], dtype=bool)))
        self.assertFalse(dcs.np_any_axis0(np.array([[False]], dtype=bool)))
        self.assertTrue(dcs.np_any_axis1(np.array([[True]], dtype=bool)))
        self.assertFalse(dcs.np_any_axis1(np.array([[False]], dtype=bool)))

    def test_0d(self) -> None:
        self.assertTrue(dcs.np_all_axis0(np.array(True, dtype=bool)))
        self.assertFalse(dcs.np_all_axis0(np.array(False, dtype=bool)))
        self.assertTrue(dcs.np_any_axis0(np.array(True, dtype=bool)))
        self.assertFalse(dcs.np_any_axis0(np.array(False, dtype=bool)))

    def test_3d(self) -> None:
        x = np.round(np.random.rand(3, 4, 5)).astype(bool)
        np.testing.assert_array_equal(dcs.np_all_axis0(x), np.all(x, axis=0))
        np.testing.assert_array_equal(dcs.np_all_axis1(x), np.all(x, axis=1))
        np.testing.assert_array_equal(dcs.np_any_axis0(x), np.any(x, axis=0))
        np.testing.assert_array_equal(dcs.np_any_axis1(x), np.any(x, axis=1))

    def test_empty(self) -> None:
        pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
