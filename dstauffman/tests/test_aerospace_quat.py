r"""
Test file for the `quat` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import annotations

from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

from slog import LogLevel

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _Q = np.typing.NDArray[np.float64]

#%% aerospace.QUAT_SIZE
class Test_aerospace_QUAT_SIZE(unittest.TestCase):
    r"""
    Tests the aerospace.QUAT_SIZE function with the following cases:
        Exists and is 4
    """

    def test_exists(self) -> None:
        self.assertEqual(space.QUAT_SIZE, 4)


#%% aerospace.suppress_quat_checks and aerospace.unsupress_quat_checks
class Test_aerospace_suppress_checks(unittest.TestCase):
    r"""
    Tests the suppress_quat_checks and unsupress_quat_checks functions with the following cases:
        Suppress and Unsuppress
    """
    orig_flag: bool

    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_flag = space.quat._USE_ASSERTIONS

    def test_suppress_and_unsupress(self) -> None:
        space.suppress_quat_checks()
        self.assertFalse(space.quat._USE_ASSERTIONS)
        space.unsuppress_quat_checks()
        self.assertTrue(space.quat._USE_ASSERTIONS)

    def tearDown(self) -> None:  # pragma: no cover
        if self.orig_flag:
            space.unsuppress_quat_checks()
        else:
            space.suppress_quat_checks()


#%% aerospace.quat_assertions
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_assertions(unittest.TestCase):
    r"""
    Tests the aerospace.quat_assertions function with the following cases:
        Nominal (x2)
        Array (x2)
        Bad (x7)
    """

    def setUp(self) -> None:
        self.q1 = np.array([0, 0, 0, 1])  # zero quaternion
        self.q2 = np.array([0.5, 0.5, 0.5, 0.5])  # normal 1D quat
        self.q3 = np.array([0.5, -0.5, -0.5, 0.5])  # normal 1D quat with some negative components
        self.q4 = np.column_stack((self.q1, self.q2, self.q3))  # 2D array
        self.q5 = np.array([[0.5], [0.5], [0.5], [0.5]])  # 2D, single quat
        self.q6 = np.array([0, 0, 0, -1])  # zero quaternion with bad scalar
        self.q7 = np.array([-5, -5, 5, 5])  # quat with bad ranges
        self.q8 = np.array([0.5, 0.5j, 0.5, 0.5], dtype=np.complex128)
        self.q9 = np.array([0, 0, 1])  # only a 3 element vector
        self.q10 = np.array([0, 0, 0])  # only a 3 element vector, but with zero magnitude
        self.q11 = np.array([[[0.5], [0.5], [0.5], [0.5]]])  # valid, but 3D instead
        self.q12 = np.column_stack((self.q1, self.q6, self.q7))  # good and bad combined

    def test_nominal1(self) -> None:
        space.quat_assertions(self.q1)

    def test_nominal2(self) -> None:
        space.quat_assertions(self.q2)

    def test_nominal3(self) -> None:
        space.quat_assertions(self.q3)

    def test_array1(self) -> None:
        space.quat_assertions(self.q4)

    def test_array2(self) -> None:
        space.quat_assertions(self.q5)

    def test_bad1(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q6)

    def test_bad2(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q7)

    def test_bad3(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q8)  # type: ignore[arg-type]

    def test_bad4(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q9)

    def test_bad5(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q10)

    def test_bad6(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q11)

    def test_bad7(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q12)

    def test_skip_assertions(self) -> None:
        space.quat_assertions(self.q6, skip_assertions=True)

    def test_nans1d_1(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(np.full(4, np.nan))

    def test_nans1d_2(self) -> None:
        space.quat_assertions(np.full(4, np.nan), allow_nans=True)

    def test_nans1d_3(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_assertions(np.array([0.0, 1.0, np.nan, 0.0]), allow_nans=True)

    def test_nans2d_1(self) -> None:
        self.q4[0, 0] = np.nan
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q4)

    def test_nans2d_2(self) -> None:
        self.q4.fill(np.nan)
        space.quat_assertions(self.q4, allow_nans=True)

    def test_nans2d_3(self) -> None:
        self.q4[:, 1] = np.nan
        space.quat_assertions(self.q4, allow_nans=True)

    def test_nans2d_4(self) -> None:
        self.q4[0, 0] = np.nan
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q4, allow_nans=True)


#%% aerospace.enforce_pos_scalar
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_enforce_pos_scalar(unittest.TestCase):
    r"""
    Tests the aerospace.enforce_pos_scalar function with the following cases:
        Single quat
        Multi quats
        Inplace
    """

    def setUp(self) -> None:
        self.q1_inp = np.array([-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.q1_out = np.array([-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.q2_inp = np.array([0, 0.5, 0, -np.sqrt(3) / 2])
        self.q2_out = np.array([0, -0.5, 0, np.sqrt(3) / 2])

    def test_nominal(self) -> None:
        q1_out = space.enforce_pos_scalar(self.q1_inp)
        np.testing.assert_array_almost_equal(q1_out, self.q1_out)
        q2_out = space.enforce_pos_scalar(self.q2_inp)
        np.testing.assert_array_almost_equal(q2_out, self.q2_out)

    def test_quat_array(self) -> None:
        quat = np.vstack((self.q1_inp, self.q2_inp)).T
        qout = space.enforce_pos_scalar(quat)
        exp = np.vstack((self.q1_out, self.q2_out)).T
        np.testing.assert_array_almost_equal(qout, exp)

    def test_inplace(self) -> None:
        q2_out = space.enforce_pos_scalar(self.q2_inp)
        self.assertGreater(np.max(np.abs(q2_out - self.q2_inp)), 0.1)
        q2_out = space.enforce_pos_scalar(self.q2_inp, inplace=True)
        self.assertLess(np.max(np.abs(q2_out - self.q2_inp)), 1e-8)
        quat = np.vstack((self.q1_inp, self.q2_inp)).T
        qout = space.enforce_pos_scalar(quat, inplace=True)
        self.assertIs(qout, quat)
        exp = np.vstack((self.q1_out, self.q2_out)).T
        np.testing.assert_array_almost_equal(qout, exp)

    def test_all_nans(self) -> None:
        q = space.enforce_pos_scalar(np.full(4, np.nan))
        self.assertTrue(np.all(np.isnan(q)))
        q = space.enforce_pos_scalar(np.full((4, 10), np.nan))
        self.assertTrue(np.all(np.isnan(q)))


#%% aerospace.qrot
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_qrot(unittest.TestCase):
    r"""
    Tests the aerospace.qrot function with the following cases:
        Single input case
        Single axis, multiple angles
        Multiple axes, single angle
        Multiple axes, multiple angles
        Null (x2)
        Vector mismatch (causes AssertionError)
    """

    def setUp(self) -> None:
        # fmt: off
        self.axis      = np.array([1, 2, 3])
        self.angle     = np.pi / 2
        self.angle2    = np.pi / 3
        r2o2           = np.sqrt(2) / 2
        r3o2           = np.sqrt(3) / 2
        self.quat      = np.array([[r2o2, 0, 0, r2o2], [0, r2o2, 0, r2o2], [0, 0, r2o2, r2o2]])
        self.quat2     = np.array([[0.5, 0, 0, r3o2], [0, 0.5, 0, r3o2], [0, 0, 0.5, r3o2]])
        self.null      = np.array([], dtype=int)
        self.null_quat = np.zeros((4, 0))
        # fmt: on

    def test_single_inputs(self) -> None:
        for i in range(len(self.axis)):
            quat = space.qrot(self.axis[i], self.angle)
            self.assertEqual(quat.ndim, 1)
            np.testing.assert_array_almost_equal(quat, self.quat[i, :])

    def test_single_axis(self) -> None:
        for i in range(len(self.axis)):
            quat = space.qrot(self.axis[i], np.array([self.angle, self.angle2]))
            self.assertEqual(quat.ndim, 2)
            np.testing.assert_array_almost_equal(quat, np.column_stack((self.quat[i, :], self.quat2[i, :])))

    def test_single_angle(self) -> None:
        quat = space.qrot(self.axis, self.angle)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, self.quat.T)

    def test_all_vector_inputs(self) -> None:
        quat = space.qrot(self.axis, np.array([self.angle, self.angle, self.angle2]))
        np.testing.assert_array_almost_equal(quat, np.column_stack((self.quat[0, :], self.quat[1, :], self.quat2[2, :])))

    def test_null1(self) -> None:
        quat = space.qrot(self.axis[0], self.null)
        np.testing.assert_array_almost_equal(quat, self.null_quat)

    def test_null2(self) -> None:
        quat = space.qrot(self.null, self.angle)
        np.testing.assert_array_almost_equal(quat, self.null_quat)

    def test_vector_mismatch(self) -> None:
        with self.assertRaises(AssertionError):
            space.qrot(self.axis, np.array([self.angle, self.angle2]))

    def test_larger_angle(self) -> None:
        quat = space.qrot(1, 5.1 * np.pi)
        self.assertGreater(quat[3], 0)


#%% aerospace.quat_from_axis_angle
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_from_axis_angle(unittest.TestCase):
    r"""
    Tests the aerospace.quat_from_axis_angle function with the following cases:
        Single axis (x3)
        Multiple axis
    """

    def test_axis1(self) -> None:
        angle = np.arange(0, 2 * np.pi, 0.01) - np.pi
        quat = space.quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), angle)
        exp = space.qrot(1, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_axis2(self) -> None:
        angle = np.arange(0, 2 * np.pi, 0.01) - np.pi
        quat = space.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), angle)
        exp = space.qrot(2, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_axis3(self) -> None:
        angle = np.arange(0, 2 * np.pi, 0.01) - np.pi
        quat = space.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)
        exp = space.qrot(3, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_single_inputs(self) -> None:
        axis = np.sqrt([9 / 50, 16 / 50, 0.5])  # unit([3, 4, 5])
        angle = 1e-6 * np.sqrt(50)
        quat = space.quat_from_axis_angle(axis, angle)
        exp = space.quat_mult(space.quat_mult(space.qrot(1, 3e-6), space.qrot(2, 4e-6)), space.qrot(3, 5e-6))
        np.testing.assert_array_almost_equal(quat, exp, 10)

    def test_axes_single_angle(self) -> None:
        axis = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [np.sqrt(9 / 50), np.sqrt(16 / 50), np.sqrt(0.5)]]
        ).T
        angle = 1e-6 * np.sqrt(50)
        quat = space.quat_from_axis_angle(axis, angle)
        # fmt: off
        exp = np.column_stack([
            space.qrot(np.array([1, 2, 3]), angle),
            space.quat_mult(space.quat_mult(space.qrot(1, 3e-6), space.qrot(2, 4e-6)), space.qrot(3, 5e-6)),
        ])
        # fmt: on
        np.testing.assert_array_almost_equal(quat, exp, 10)

    def test_multi_axis_angle(self) -> None:
        axis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], np.full(3, np.sqrt(3) / 3)]).T
        angle = np.array([-0.5, 1.5, 5.5, 0.0])
        quat = space.quat_from_axis_angle(axis, angle)
        exp = space.qrot(np.array([1, 2, 3, 1]), angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_null_axis(self) -> None:
        quat = space.quat_from_axis_angle(np.zeros(3), 0.1)
        np.testing.assert_array_equal(quat, np.array([0.0, 0.0, 0.0, 1.0]))

    def test_null_axis_2d(self) -> None:
        axis = np.zeros((3, 4))
        axis[1, 1] = 1.0
        quat = space.quat_from_axis_angle(axis, np.array([0.1, 0.2, 5.3, 0.4]))
        null = np.array([0.0, 0.0, 0.0, 1.0])
        exp = np.column_stack([null, space.qrot(2, 0.2), null, null])
        np.testing.assert_array_equal(quat, exp)


#%% aerospace.quat_angle_diff
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_angle_diff(unittest.TestCase):
    r"""
    Tests the aerospace.quat_angle_diff function with the following cases:
        Nominal (x2)
        Array (x3)
        Zero diff (x4)
        Null (x4)
    """

    def setUp(self) -> None:
        # fmt: off
        self.quat1 = np.array([0.5, 0.5, 0.5, 0.5])
        self.dq1   = space.qrot(1, 0.001)
        self.dq2   = space.qrot(2, 0.05)
        self.dqq1  = space.quat_mult(self.dq1, self.quat1)
        self.dqq2  = space.quat_mult(self.dq2, self.quat1)
        self.theta = np.array([0.001, 0.05])
        self.comp  = np.array([[0.001, 0], [0, 0.05], [0, 0]])
        self.null: _Q = np.array([])
        self.null_quat = np.zeros((4, 0))
        # fmt: on

    def test_nominal1(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.quat1, self.dqq1)
        np.testing.assert_array_almost_equal(theta, self.theta[0])
        np.testing.assert_array_almost_equal(comp, self.comp[:, 0])

    def test_nominal2(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.quat1, self.dqq2)
        np.testing.assert_array_almost_equal(theta, self.theta[1])
        np.testing.assert_array_almost_equal(comp, self.comp[:, 1])

    def test_array1(self) -> None:
        (theta, comp) = space.quat_angle_diff(np.column_stack((self.dqq1, self.dqq2)), self.quat1)
        np.testing.assert_array_almost_equal(theta, self.theta)
        np.testing.assert_array_almost_equal(comp, -self.comp)

    def test_array2(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.quat1, np.column_stack((self.dqq1, self.dqq2)))
        np.testing.assert_array_almost_equal(theta, self.theta)
        np.testing.assert_array_almost_equal(comp, self.comp)

    def test_array3(self) -> None:
        (theta, comp) = space.quat_angle_diff(
            np.column_stack((self.quat1, self.quat1, self.dqq1, self.dqq2)),
            np.column_stack((self.dqq1, self.dqq2, self.quat1, self.quat1)),
        )
        np.testing.assert_array_almost_equal(theta, self.theta[[0, 1, 0, 1]])
        np.testing.assert_array_almost_equal(comp, self.comp[:, [0, 1, 0, 1]] * np.array([1, 1, -1, -1]))

    def test_zero_diff1(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.quat1, self.quat1)
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff2(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.quat1, np.column_stack((self.quat1, self.quat1)))
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff3(self) -> None:
        (theta, comp) = space.quat_angle_diff(np.column_stack((self.quat1, self.quat1)), self.quat1)
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff4(self) -> None:
        temp = np.column_stack((self.quat1, self.dq1, self.dq2, self.dqq1, self.dqq2))
        (theta, comp) = space.quat_angle_diff(temp, temp)
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_null1(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.quat1, self.null)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0,))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))

    def test_null2(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.quat1, self.null_quat)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0,))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))

    def test_null3(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.null, self.quat1)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0,))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))

    def test_null4(self) -> None:
        (theta, comp) = space.quat_angle_diff(self.null_quat, self.quat1)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0,))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))


#%% aerospace.quat_from_euler
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_from_euler(unittest.TestCase):
    r"""
    Tests the aerospace.quat_from_euler function with the following cases:
        Nominal (x2 different values)
        Default sequence
        Repeated axis
        Shorter than normal sequence
        Single rotation sequence (x2 for actual scalar)
        Longer than normal rotation sequence
        Array cases (x3 2D, 2D with unit len, and >2D for error)
    """

    def setUp(self) -> None:
        # fmt: off
        self.a      = np.array([0.01, 0.02, 0.03])
        self.b      = np.array([0.04, 0.05, 0.06])
        self.angles = np.column_stack((self.a, self.b))
        self.seq    = np.array([3, 2, 1])
        self.quat   = np.array([
            [0.01504849, 0.03047982],
            [0.00992359, 0.02438147],
            [0.00514916, 0.02073308],
            [0.99982426, 0.99902285],
        ])
        # fmt: on

    def test_nominal1(self) -> None:
        quat = space.quat_from_euler(self.a, self.seq)
        np.testing.assert_array_almost_equal(quat, self.quat[:, 0])
        self.assertEqual(quat.ndim, 1)

    def test_nominal2(self) -> None:
        quat = space.quat_from_euler(self.b, self.seq)
        np.testing.assert_array_almost_equal(quat, self.quat[:, 1])
        self.assertEqual(quat.ndim, 1)

    def test_default_seq(self) -> None:
        quat = space.quat_from_euler(self.a)
        temp = space.quat_mult(space.quat_mult(space.qrot(3, self.a[0]), space.qrot(1, self.a[1])), space.qrot(2, self.a[2]))
        np.testing.assert_array_almost_equal(quat, temp)
        self.assertEqual(quat.ndim, 1)

    def test_repeated(self) -> None:
        quat1 = space.quat_from_euler(np.hstack((self.a, self.a)), seq=np.array([1, 1, 1, 1, 1, 1]))
        quat2 = space.qrot(1, 2 * np.sum(self.a))
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_short(self) -> None:
        quat1 = space.quat_from_euler(self.a[0:2], self.seq[0:2])
        quat2 = space.quat_mult(space.qrot(self.seq[0], self.a[0]), space.qrot(self.seq[1], self.a[1]))
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_single1(self) -> None:
        quat1 = space.quat_from_euler(self.a[0], self.seq[0])
        quat2 = space.qrot(self.seq[0], self.a[0])
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_single2(self) -> None:
        quat1 = space.quat_from_euler(0.01, 3)
        quat2 = space.qrot(3, 0.01)
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_long(self) -> None:
        quat1 = space.quat_from_euler(np.hstack((self.a, self.b)), seq=np.hstack((self.seq, self.seq)))
        quat2 = space.quat_mult(self.quat[:, 0], self.quat[:, 1])
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_array1(self) -> None:
        quat = space.quat_from_euler(self.angles, self.seq)
        np.testing.assert_array_almost_equal(quat, self.quat)
        self.assertEqual(quat.ndim, 2)

    def test_array2(self) -> None:
        quat = space.quat_from_euler(np.expand_dims(self.a, axis=1), self.seq)
        np.testing.assert_array_almost_equal(quat, np.expand_dims(self.quat[:, 0], axis=1))
        self.assertEqual(quat.ndim, 2)

    def test_array3(self) -> None:
        with self.assertRaises(ValueError):
            space.quat_from_euler(np.zeros((3, 3, 1)))


#%% aerospace.quat_interp
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_interp(unittest.TestCase):
    r"""
    Tests the aerospace.quat_interp function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.time = np.array([1, 3, 5])
        self.quat = np.column_stack((space.qrot(1, 0), space.qrot(1, np.pi / 2), space.qrot(1, np.pi)))
        self.ti   = np.array([1, 2, 4.5, 5])  # fmt: skip
        self.qout = np.column_stack(
            (space.qrot(1, 0), space.qrot(1, np.pi / 4), space.qrot(1, 3.5 / 4 * np.pi), space.qrot(1, np.pi))
        )
        self.ti_extra = np.array([0, 1, 2, 4.5, 5, 10])

    def test_nominal(self) -> None:
        qout = space.quat_interp(self.time, self.quat, self.ti)
        np.testing.assert_array_almost_equal(qout, self.qout)

    def test_empty(self) -> None:
        qout = space.quat_interp(self.time, self.quat, np.array([]))
        self.assertEqual(qout.size, 0)

    def test_scalar1(self) -> None:
        qout = space.quat_interp(self.time, self.quat, self.ti[0])
        np.testing.assert_array_almost_equal(qout, np.expand_dims(self.qout[:, 0], 1))

    def test_scalar2(self) -> None:
        qout = space.quat_interp(self.time, self.quat, self.ti[1])
        np.testing.assert_array_almost_equal(qout, self.qout[:, 1])

    def test_extra1(self) -> None:
        with self.assertRaises(ValueError):
            space.quat_interp(self.time, self.quat, self.ti_extra, inclusive=True)

    def test_extra2(self) -> None:
        with patch("dstauffman.aerospace.quat.logger") as mock_logger:
            qout = space.quat_interp(self.time, self.quat, self.ti_extra, inclusive=False)
        np.testing.assert_array_almost_equal(qout[:, 1:-1], self.qout)
        np.testing.assert_array_equal(qout[:, [0, -1]], np.nan)
        mock_logger.log.assert_called_with(LogLevel.L8, "Desired time not found within input time vector.")


#%% aerospace.quat_inv
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_inv(unittest.TestCase):
    r"""
    Tests the aerospace.quat_inv function with the following cases:
        Single quat (x2 different quats)
        Quat array
        Null (x2 different null sizes)
    """

    def setUp(self) -> None:
        self.q1_inp = space.qrot(1, np.pi / 2)
        self.q1_out = np.array([-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.q2_inp = space.qrot(2, np.pi / 3)
        self.q2_out = np.array([0, -0.5, 0, np.sqrt(3) / 2])
        self.q3_inp = np.column_stack((self.q1_inp, self.q2_inp))
        self.q3_out = np.column_stack((self.q1_out, self.q2_out))
        self.null: _Q = np.array([])
        self.null_quat = np.ones((space.QUAT_SIZE, 0))

    def test_single_quat1(self) -> None:
        q1_inv = space.quat_inv(self.q1_inp)
        np.testing.assert_array_almost_equal(q1_inv, self.q1_out)
        self.assertEqual(q1_inv.ndim, 1)
        np.testing.assert_array_equal(q1_inv.shape, self.q1_out.shape)

    def test_single_quat2(self) -> None:
        q2_inv = space.quat_inv(self.q2_inp)
        np.testing.assert_array_almost_equal(q2_inv, self.q2_out)
        self.assertEqual(q2_inv.ndim, 1)
        np.testing.assert_array_equal(q2_inv.shape, self.q2_out.shape)

    def test_quat_array(self) -> None:
        q3_inv = space.quat_inv(self.q3_inp)
        np.testing.assert_array_almost_equal(q3_inv, self.q3_out)
        self.assertEqual(q3_inv.ndim, 2)
        np.testing.assert_array_equal(q3_inv.shape, self.q3_out.shape)

    def test_null_input1(self) -> None:
        null_inv = space.quat_inv(self.null_quat)
        np.testing.assert_array_equal(null_inv, self.null_quat)
        np.testing.assert_array_equal(null_inv.shape, self.null_quat.shape)

    def test_null_input2(self) -> None:
        null_inv = space.quat_inv(self.null)
        np.testing.assert_array_equal(null_inv, self.null)
        np.testing.assert_array_equal(null_inv.shape, self.null.shape)

    def test_inplace_single(self) -> None:
        q1_inv = space.quat_inv(self.q1_inp)
        self.assertIsNot(q1_inv, self.q1_inp)
        np.testing.assert_array_almost_equal(q1_inv, self.q1_out)
        q1_inv = space.quat_inv(self.q1_inp, inplace=True)
        self.assertIs(q1_inv, self.q1_inp)
        np.testing.assert_array_almost_equal(q1_inv, self.q1_out)

    def test_inplace_array(self) -> None:
        q3_inv = space.quat_inv(self.q3_inp)
        self.assertIsNot(q3_inv, self.q3_inp)
        np.testing.assert_array_almost_equal(q3_inv, self.q3_out)
        q3_inv = space.quat_inv(self.q3_inp, inplace=True)
        self.assertIs(q3_inv, self.q3_inp)
        np.testing.assert_array_almost_equal(q3_inv, self.q3_out)

    def test_all_nans(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_inv(np.full(4, np.nan))
        q = space.quat_inv(np.full(4, np.nan), allow_nans=True)
        self.assertTrue(np.all(np.isnan(q)))
        with self.assertRaises(AssertionError):
            space.quat_inv(np.full((4, 5), np.nan))
        q = space.quat_inv(np.full((4, 5), np.nan), allow_nans=True)
        self.assertTrue(np.all(np.isnan(q)))
        self.q3_inp[:, 0] = np.nan
        self.q3_out[:, 0] = np.nan
        q3_inv = space.quat_inv(self.q3_inp, allow_nans=True)
        np.testing.assert_array_almost_equal(q3_inv, self.q3_out)


#%% aerospace.quat_mult
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_mult(unittest.TestCase):
    r"""
    Tests the aerospace.quat_mult function with the following cases:
        Single quat (x2 different quats)
        Reverse order
        Quat array times scalar (x2 orders + x1 array-array)
        Null (x8 different null size and order permutations)
    """

    def setUp(self) -> None:
        self.q1 = space.qrot(1, np.pi / 2)
        self.q2 = space.qrot(2, -np.pi)
        self.q3 = space.qrot(3, np.pi / 3)
        self.q4 = np.array([0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0])  # q1*q2
        self.q5 = np.array([0.5, -np.sqrt(3) / 2, 0, 0])  # q2*q3
        self.q6 = np.array([0.5, 0.5, 0.5, 0.5])  # q6 * q6 = q6**-1, and triggers negative scalar component
        self.q_array_in1 = np.column_stack((self.q1, self.q2))
        self.q_array_in2 = np.column_stack((self.q2, self.q3))
        self.q_array_out = np.column_stack((self.q4, self.q5))
        self.null: _Q = np.array([])
        self.null_quat = np.ones((space.QUAT_SIZE, 0))

    def test_nominal1(self) -> None:
        quat = space.quat_mult(self.q1, self.q2)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q4)
        np.testing.assert_array_equal(quat.shape, self.q4.shape)

    def test_nominal2(self) -> None:
        quat = space.quat_mult(self.q2, self.q3)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q5)
        np.testing.assert_array_equal(quat.shape, self.q5.shape)

    def test_nominal3(self) -> None:
        quat = space.quat_mult(self.q6, self.q6)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, space.quat_inv(self.q6))
        np.testing.assert_array_equal(quat.shape, self.q6.shape)

    def test_reverse(self) -> None:
        quat1 = space.quat_mult(self.q2, self.q1)
        quat2 = space.quat_inv(space.quat_mult(space.quat_inv(self.q1), space.quat_inv(self.q2)))
        np.testing.assert_array_almost_equal(quat1, quat2)

    def test_array_scalar1(self) -> None:
        quat = space.quat_mult(self.q_array_in1, self.q2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat[:, 0], self.q4)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_array_scalar2(self) -> None:
        quat = space.quat_mult(self.q1, self.q_array_in2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat[:, 0], self.q4)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_array_scalar3(self) -> None:
        quat = space.quat_mult(self.q6, np.column_stack((self.q6, self.q6)))
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, np.column_stack((space.quat_inv(self.q6), space.quat_inv(self.q6))))
        np.testing.assert_array_equal(quat.shape, (4, 2))

    def test_array(self) -> None:
        quat = space.quat_mult(self.q_array_in1, self.q_array_in2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, self.q_array_out)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_null_input1(self) -> None:
        quat = space.quat_mult(self.null_quat, self.q2)
        np.testing.assert_array_equal(quat, self.null_quat)
        np.testing.assert_array_equal(quat.shape, self.null_quat.shape)

    def test_null_input2(self) -> None:
        quat = space.quat_mult(self.null, self.q2)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input3(self) -> None:
        quat = space.quat_mult(self.q1, self.null_quat)
        np.testing.assert_array_equal(quat, self.null_quat)
        np.testing.assert_array_equal(quat.shape, self.null_quat.shape)

    def test_null_input4(self) -> None:
        quat = space.quat_mult(self.q2, self.null)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input5(self) -> None:
        quat = space.quat_mult(self.null_quat, self.null_quat)
        np.testing.assert_array_equal(quat, self.null_quat)
        np.testing.assert_array_equal(quat.shape, self.null_quat.shape)

    def test_null_input6(self) -> None:
        quat = space.quat_mult(self.null, self.null)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input7(self) -> None:
        quat = space.quat_mult(self.null_quat, self.null)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input8(self) -> None:
        quat = space.quat_mult(self.null, self.null_quat)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)


#%% aerospace.quat_norm
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_norm(unittest.TestCase):
    r"""
    Tests the aerospace.quat_norm function with the following cases:
        Single quat (x3 different quats)
        Quat array
        Null (x2 different null sizes)
    """

    def setUp(self) -> None:
        self.q1_inp = space.qrot(1, np.pi / 2)
        self.q1_out = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.q2_inp = space.qrot(2, np.pi / 3)
        self.q2_out = np.array([0, 0.5, 0, np.sqrt(3) / 2])
        self.q3_inp = np.array([0.1, 0, 0, 1])
        self.q3_out = np.array([0.09950372, 0, 0, 0.99503719])
        self.q4_inp = np.column_stack((self.q1_inp, self.q2_inp, self.q3_inp))
        self.q4_out = np.column_stack((self.q1_out, self.q2_out, self.q3_out))
        self.null: _Q = np.array([])
        self.null_quat = np.ones((space.QUAT_SIZE, 0))

    def test_nominal1(self) -> None:
        quat_norm = space.quat_norm(self.q1_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q1_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q1_out.shape)

    def test_nominal2(self) -> None:
        quat_norm = space.quat_norm(self.q2_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q2_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q2_out.shape)

    def test_nominal3(self) -> None:
        quat_norm = space.quat_norm(self.q3_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q3_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q3_out.shape)

    def test_array(self) -> None:
        quat_norm = space.quat_norm(self.q4_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q4_out)
        self.assertEqual(quat_norm.ndim, 2)
        np.testing.assert_array_equal(quat_norm.shape, self.q4_out.shape)

    def test_null_input1(self) -> None:
        quat_norm = space.quat_norm(self.null_quat)
        np.testing.assert_array_equal(quat_norm, self.null_quat)
        np.testing.assert_array_equal(quat_norm.shape, self.null_quat.shape)

    def test_null_input2(self) -> None:
        quat_norm = space.quat_norm(self.null)
        np.testing.assert_array_equal(quat_norm, self.null)
        np.testing.assert_array_equal(quat_norm.shape, self.null.shape)

    def test_inplace_single(self) -> None:
        quat_norm = space.quat_norm(self.q1_inp)
        self.assertIsNot(quat_norm, self.q1_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q1_out)
        quat_norm = space.quat_norm(self.q1_inp, inplace=True)
        self.assertIs(quat_norm, self.q1_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q1_out)

    def test_inplace_array(self) -> None:
        quat_norm = space.quat_norm(self.q4_inp)
        self.assertIsNot(quat_norm, self.q4_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q4_out)
        quat_norm = space.quat_norm(self.q4_inp, inplace=True)
        self.assertIs(quat_norm, self.q4_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q4_out)

    def test_all_nans(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_norm(np.full(4, np.nan))
        q = space.quat_norm(np.full(4, np.nan), allow_nans=True)
        self.assertTrue(np.all(np.isnan(q)))
        with self.assertRaises(AssertionError):
            space.quat_norm(np.full((4, 5), np.nan))
        q = space.quat_norm(np.full((4, 5), np.nan), allow_nans=True)
        self.assertTrue(np.all(np.isnan(q)))
        self.q4_inp[:, 0] = np.nan
        self.q4_out[:, 0] = np.nan
        q4_inv = space.quat_norm(self.q4_inp, allow_nans=True)
        np.testing.assert_array_almost_equal(q4_inv, self.q4_out)

    def test_some_nans(self) -> None:
        self.q4_inp[1, 2] = np.nan
        self.q4_out[:, 2] = np.nan
        with self.assertRaises(AssertionError):
            space.quat_norm(self.q4_inp)
        quat_norm = space.quat_norm(self.q4_inp, allow_nans=True)
        np.testing.assert_array_almost_equal(quat_norm, self.q4_out)
        self.assertEqual(quat_norm.ndim, 2)
        np.testing.assert_array_equal(quat_norm.shape, self.q4_out.shape)


#%% aerospace.quat_prop
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_prop(unittest.TestCase):
    r"""
    Tests the aerospace.quat_prop function with the following cases:
        Nominal case
        Negative scaler
        No renormalization case (Raises norm AttributeError)
        No renormalization with suppressed warning
    """

    def setUp(self) -> None:
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.delta_ang = np.array([0.01, 0.02, 0.03])
        self.quat_new = np.array([0.00499912522962, 0.00999825045924, 0.01499737568886, 0.99982504592411])
        self.quat_unnorm = np.array([0.005, 0.01, 0.015, 1.0])

    def test_nominal(self) -> None:
        quat = space.quat_prop(self.quat, self.delta_ang)
        np.testing.assert_array_almost_equal(quat, self.quat_new, 12)

    def test_negative_scalar(self) -> None:
        quat = space.quat_prop(np.array([1.0, 0.0, 0.0, 0.0]), self.delta_ang)
        self.assertGreater(quat[3], 0)
        quat = space.quat_prop(np.array([1.0, 0.0, 0.0, 0.0]), -self.delta_ang)
        self.assertGreater(quat[3], 0)

    def test_no_renorm(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_prop(self.quat, self.delta_ang, renorm=False)

    def test_no_renorm_suppress(self) -> None:
        quat = space.quat_prop(self.quat, self.delta_ang, renorm=False, skip_assertions=True)
        np.testing.assert_array_almost_equal(quat, self.quat_unnorm, 12)


#%% aerospace.quat_times_vector
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_times_vector(unittest.TestCase):
    r"""
    Tests the aerospace.quat_times_vector function with the following cases:
        Nominal
        Array inputs
    """

    def setUp(self) -> None:
        # TODO: confirm that this is enough to test the correctness of the function
        self.quat = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).T
        self.vec = np.array([[1, 0, 0], [2, 0, 0]]).T
        self.out = np.array([[-1, 2], [0, 0], [0, 0]])

    def test_nominal(self) -> None:
        for i in range(2):
            vec = space.quat_times_vector(self.quat[:, i], self.vec[:, i])
            np.testing.assert_array_almost_equal(vec, self.out[:, i])

    def test_array(self) -> None:
        vec = space.quat_times_vector(self.quat, self.vec)
        np.testing.assert_array_almost_equal(vec, self.out)

    def test_vector_array(self) -> None:
        quat = self.quat[:, 0]
        vec1 = space.quat_times_vector(quat, self.vec)
        vec2 = space.quat_times_vector(np.vstack((quat, quat)).T, self.vec)
        np.testing.assert_array_almost_equal(vec1, vec2)

    def test_array_vector(self) -> None:
        vec = self.vec[:, 0]
        vec1 = space.quat_times_vector(self.quat, vec)
        vec2 = space.quat_times_vector(self.quat, np.vstack((vec, vec)).T)
        np.testing.assert_array_almost_equal(vec1, vec2)


#%% aerospace.quat_to_euler
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_to_euler(unittest.TestCase):
    r"""
    Tests the aerospace.quat_to_euler function with the following cases:
        Nominal
        Zero quat
        All valid sequences
        All invalid sequences
        Bad length sequence
    """

    def setUp(self) -> None:
        self.quat = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]]).T
        self.seq = [3, 1, 2]
        self.euler = np.array([[0.0, -np.pi, 0.0], [0.0, 0.0, -np.pi / 2], [np.pi, -0.0, 0.0]])
        self.zero_quat = np.array([0, 0, 0, 1])
        self.all_sequences = {
            (1, 1, 1),
            (1, 1, 2),
            (1, 1, 3),
            (1, 2, 1),
            (1, 2, 2),
            (1, 2, 3),
            (1, 3, 1),
            (1, 3, 2),
            (1, 3, 3),
            (2, 1, 1),
            (2, 1, 2),
            (2, 1, 3),
            (2, 2, 1),
            (2, 2, 2),
            (2, 2, 3),
            (2, 3, 1),
            (2, 3, 2),
            (2, 3, 3),
            (3, 1, 1),
            (3, 1, 2),
            (3, 1, 3),
            (3, 2, 1),
            (3, 2, 2),
            (3, 2, 3),
            (3, 3, 1),
            (3, 3, 2),
            (3, 3, 3),
        }
        self.valid_sequences = {(1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2), (2, 1, 3), (3, 2, 1)}
        self.bad_sequences = self.all_sequences - self.valid_sequences

    def test_nominal(self) -> None:
        euler = space.quat_to_euler(self.quat, self.seq)
        np.testing.assert_array_almost_equal(euler, self.euler)

    def test_zero_quat(self) -> None:
        euler = space.quat_to_euler(self.zero_quat)
        np.testing.assert_array_equal(euler, np.zeros(3))

    def test_all_valid(self) -> None:
        # TODO: this doesn't confirm that all of these give the correct answer, but just don't crash
        for this_seq in self.valid_sequences:
            euler = space.quat_to_euler(self.zero_quat, np.array(this_seq))
            np.testing.assert_array_equal(euler, np.zeros(3))

    def test_all_invalid(self) -> None:
        for this_seq in self.bad_sequences:
            with self.assertRaises(ValueError):
                space.quat_to_euler(self.zero_quat, np.array(this_seq))

    def test_bad_sequence(self) -> None:
        with self.assertRaises(AssertionError):
            space.quat_to_euler(self.zero_quat, np.array([1, 2]))


#%% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
