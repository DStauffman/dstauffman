r"""
Test file for the `quat_opt` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in February 2021.

"""

# %% Imports
import unittest

from nubs import HAVE_NUMBA

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np


# %% aerospace.qrot_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_qrot_single(unittest.TestCase):
    r"""
    Tests the aerospace.qrot_single function with the following cases:
        Single input case
    """

    def setUp(self) -> None:
        # fmt: off
        self.axis   = np.array([1, 2, 3])
        self.angle  = np.pi / 2
        self.angle2 = np.pi / 3
        r2o2        = np.sqrt(2) / 2
        r3o2        = np.sqrt(3) / 2
        self.quat   = np.array([[r2o2, 0, 0, r2o2], [0, r2o2, 0, r2o2], [0, 0, r2o2, r2o2]])
        self.quat2  = np.array([[0.5, 0, 0, r3o2], [0, 0.5, 0, r3o2], [0, 0, 0.5, r3o2]])
        # fmt: on

    def test_single_inputs(self) -> None:
        for i in range(len(self.axis)):
            quat = space.qrot_single(self.axis[i], self.angle)
            self.assertEqual(quat.ndim, 1)
            np.testing.assert_array_almost_equal(quat, self.quat[i, :])
            quat = space.qrot_single(self.axis[i], self.angle2)
            self.assertEqual(quat.ndim, 1)
            np.testing.assert_array_almost_equal(quat, self.quat2[i, :])

    def test_larger_angle(self) -> None:
        quat = space.qrot_single(1, 5.1 * np.pi)
        self.assertGreater(quat[3], 0)


# %% aerospace.quat_from_axis_angle_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_from_axis_angle_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_from_axis_angle_single function with the following cases:
        Single axis (x3)
        Multiple axis
    """

    def test_axis1(self) -> None:
        angle = 5.0 / 180.0 * np.pi
        quat = space.quat_from_axis_angle_single(np.array([1.0, 0.0, 0.0]), angle)
        exp = space.qrot_single(1, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_axis2(self) -> None:
        angle = 110.0 / 180.0 * np.pi
        quat = space.quat_from_axis_angle_single(np.array([0.0, 1.0, 0.0]), angle)
        exp = space.qrot_single(2, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_axis3(self) -> None:
        angle = -45.0 / 180.0 * np.pi
        quat = space.quat_from_axis_angle_single(np.array([0.0, 0.0, 1.0]), angle)
        exp = space.qrot_single(3, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_multiple(self) -> None:
        axis = np.sqrt([9 / 50, 16 / 50, 0.5])  # unit([3, 4, 5])
        angle = 1e-6 * np.sqrt(50)
        quat = space.quat_from_axis_angle_single(axis, angle)
        exp = space.quat_mult_single(
            space.quat_mult_single(space.qrot_single(1, 3e-6), space.qrot_single(2, 4e-6)), space.qrot_single(3, 5e-6)
        )
        np.testing.assert_array_almost_equal(quat, exp, 10)

    def test_null_axis(self) -> None:
        quat = space.quat_from_axis_angle_single(np.zeros(3), 0.1)
        np.testing.assert_array_equal(quat, np.array([0.0, 0.0, 0.0, 1.0]))


# %% aerospace.quat_angle_diff_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_angle_diff_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_angle_diff_single function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        # fmt: off
        self.quat1 = np.array([0.5, 0.5, 0.5, 0.5])
        self.dq1   = space.qrot(1, 0.001)
        self.dq2   = space.qrot(2, 0.05)
        self.dqq1  = space.quat_mult(self.dq1, self.quat1)
        self.dqq2  = space.quat_mult(self.dq2, self.quat1)
        self.comp1 = np.array([0.001, 0.0, 0.0])
        self.comp2 = np.array([0.0, 0.05, 0.0])
        # fmt: on

    def test_nominal1(self) -> None:
        comp = space.quat_angle_diff_single(self.quat1, self.dqq1)
        np.testing.assert_array_almost_equal(comp, self.comp1)

    def test_nominal2(self) -> None:
        comp = space.quat_angle_diff_single(self.quat1, self.dqq2)
        np.testing.assert_array_almost_equal(comp, self.comp2)

    def test_zero_diff(self) -> None:
        comp = space.quat_angle_diff_single(self.quat1, self.quat1)
        np.testing.assert_array_almost_equal(comp, 0)


# %% aerospace.quat_from_rotation_vector_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_from_rotation_vector_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_from_rotation_vector_single function with the following cases:
        Single axis (x3)
        Multiple axis
    """

    def test_axis1(self) -> None:
        angle = 5.0 / 180.0 * np.pi
        quat = space.quat_from_rotation_vector_single(angle * np.array([1.0, 0.0, 0.0]))
        exp = space.qrot_single(1, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_axis2(self) -> None:
        angle = 110.0 / 180.0 * np.pi
        quat = space.quat_from_rotation_vector_single(angle * np.array([0.0, 1.0, 0.0]))
        exp = space.qrot_single(2, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_axis3(self) -> None:
        angle = -45.0 / 180.0 * np.pi
        quat = space.quat_from_rotation_vector_single(angle * np.array([0.0, 0.0, 1.0]))
        exp = space.qrot_single(3, angle)
        np.testing.assert_array_almost_equal(quat, exp, 14)

    def test_multiple(self) -> None:
        axis = np.sqrt([9 / 50, 16 / 50, 0.5])  # unit([3, 4, 5])
        angle = 1e-6 * np.sqrt(50)
        quat = space.quat_from_rotation_vector_single(angle * axis)
        exp = space.quat_mult_single(
            space.quat_mult_single(space.qrot_single(1, 3e-6), space.qrot_single(2, 4e-6)), space.qrot_single(3, 5e-6)
        )
        np.testing.assert_array_almost_equal(quat, exp, 10)

    def test_null_axis(self) -> None:
        quat = space.quat_from_rotation_vector_single(np.zeros(3))
        np.testing.assert_array_equal(quat, np.array([0.0, 0.0, 0.0, 1.0]))


# %% aerospace.quat_interp_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_interp_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_interp_single function with the following cases:
        Nominal
        Extrapolation
    """

    def setUp(self) -> None:
        self.time = np.array([1.0, 3.0, 5.0])
        self.quat = np.vstack((space.qrot_single(1, 0), space.qrot_single(1, np.pi / 2), space.qrot_single(1, np.pi))).T
        self.ti = np.array([1.0, 2.0, 4.5, 5.0])
        self.qout = np.column_stack(
            (
                space.qrot_single(1, 0),
                space.qrot_single(1, np.pi / 4),
                space.qrot_single(1, 3.5 / 4 * np.pi),
                space.qrot_single(1, np.pi),
            )
        )

    def test_nominal(self) -> None:
        ix = np.array([0, 1])
        qout = space.quat_interp_single(self.time[ix], self.quat[:, ix], self.ti[0])
        np.testing.assert_array_almost_equal(qout, self.qout[:, 0])
        ix = np.array([0, 1])
        qout = space.quat_interp_single(self.time[ix], self.quat[:, ix], self.ti[1])
        np.testing.assert_array_almost_equal(qout, self.qout[:, 1])
        ix = np.array([1, 2])
        qout = space.quat_interp_single(self.time[ix], self.quat[:, ix], self.ti[2])
        np.testing.assert_array_almost_equal(qout, self.qout[:, 2])
        ix = np.array([1, 2])
        qout = space.quat_interp_single(self.time[ix], self.quat[:, ix], self.ti[3])
        if qout[0] > 0:
            np.testing.assert_array_almost_equal(qout, self.qout[:, 3])  # pragma: no cover
        else:
            np.testing.assert_array_almost_equal(qout, -self.qout[:, 3])

    def test_extrapolation(self) -> None:
        with self.assertRaises(ValueError):
            space.quat_interp_single(self.time, self.quat, 10.0)


# %% aerospace.quat_inv_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_inv_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_inv_single function with the following cases:
        Single quat (x2 different quats)
        In-place
    """

    def setUp(self) -> None:
        self.q1_inp = space.qrot_single(1, np.pi / 2)
        self.q1_out = np.array([-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.q2_inp = space.qrot_single(2, np.pi / 3)
        self.q2_out = np.array([0, -0.5, 0, np.sqrt(3) / 2])

    def test_single_quat1(self) -> None:
        q1_inv = space.quat_inv_single(self.q1_inp)
        np.testing.assert_array_almost_equal(q1_inv, self.q1_out)
        self.assertEqual(q1_inv.ndim, 1)
        np.testing.assert_array_equal(q1_inv.shape, self.q1_out.shape)

    def test_single_quat2(self) -> None:
        q2_inv = space.quat_inv_single(self.q2_inp)
        np.testing.assert_array_almost_equal(q2_inv, self.q2_out)
        self.assertEqual(q2_inv.ndim, 1)
        np.testing.assert_array_equal(q2_inv.shape, self.q2_out.shape)

    def test_inplace(self) -> None:
        q1_inv = space.quat_inv_single(self.q1_inp)
        self.assertGreater(np.max(np.abs(q1_inv - self.q1_inp)), 0.1)
        q1_inv = space.quat_inv_single(self.q1_inp, inplace=True)
        self.assertLess(np.max(np.abs(q1_inv - self.q1_inp)), 1e-8)


# %% aerospace.quat_mult_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_mult_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_mult_single function with the following cases:
        Single quat (x2 different quats)
        Reverse order
        Quat array times scalar (x2 orders)
        Inplace
    """

    def setUp(self) -> None:
        self.q1 = space.qrot_single(1, np.pi / 2)
        self.q2 = space.qrot_single(2, -np.pi)
        self.q3 = space.qrot_single(3, np.pi / 3)
        self.q4 = np.array([0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0])  # q1*q2
        self.q5 = np.array([0.5, -np.sqrt(3) / 2, 0, 0])  # q2*q3
        self.q6 = np.array([0.5, 0.5, 0.5, 0.5])  # q6 * q6 = q6**-1, and triggers negative scalar component

    def test_nominal1(self) -> None:
        quat = space.quat_mult_single(self.q1, self.q2)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q4)
        np.testing.assert_array_equal(quat.shape, self.q4.shape)

    def test_nominal2(self) -> None:
        quat = space.quat_mult_single(self.q2, self.q3)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q5)
        np.testing.assert_array_equal(quat.shape, self.q5.shape)

    def test_nominal3(self) -> None:
        quat = space.quat_mult_single(self.q6, self.q6)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, space.quat_inv_single(self.q6))
        np.testing.assert_array_equal(quat.shape, self.q6.shape)

    def test_reverse(self) -> None:
        quat1 = space.quat_mult_single(self.q2, self.q1)
        quat2 = space.quat_inv_single(space.quat_mult_single(space.quat_inv_single(self.q1), space.quat_inv_single(self.q2)))
        np.testing.assert_array_almost_equal(quat1, quat2)

    def test_inplace(self) -> None:
        quat = space.quat_mult_single(self.q1, self.q2)
        self.assertGreater(np.max(np.abs(quat - self.q1)), 0.1)
        quat = space.quat_mult_single(self.q1, self.q2, inplace=True)
        self.assertIs(quat, self.q1)
        self.assertLess(np.max(np.abs(quat - self.q4)), 1e-8)


# %% aerospace.quat_norm_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_norm_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_norm_single function with the following cases:
        Single quat (x3 different quats)
        Inplace
    """

    def setUp(self) -> None:
        self.q1_inp = space.qrot_single(1, np.pi / 2)
        self.q1_out = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.q2_inp = space.qrot_single(2, np.pi / 3)
        self.q2_out = np.array([0, 0.5, 0, np.sqrt(3) / 2])
        self.q3_inp = np.array([0.1, 0, 0, 1])
        self.q3_out = np.array([0.09950372, 0, 0, 0.99503719])

    def test_nominal1(self) -> None:
        quat_norm = space.quat_norm_single(self.q1_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q1_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q1_out.shape)

    def test_nominal2(self) -> None:
        quat_norm = space.quat_norm_single(self.q2_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q2_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q2_out.shape)

    def test_nominal3(self) -> None:
        quat_norm = space.quat_norm_single(self.q3_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q3_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q3_out.shape)

    def test_inplace(self) -> None:
        quat_norm = space.quat_norm_single(self.q3_inp)
        self.assertGreater(np.max(np.abs(quat_norm - self.q3_inp)), 0.004)
        quat_norm = space.quat_norm_single(self.q3_inp, inplace=True)
        self.assertIs(quat_norm, self.q3_inp)
        self.assertLess(np.max(np.abs(quat_norm - self.q3_inp)), 1e-8)


# %% aerospace.quat_prop_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_prop_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_prop_single function with the following cases:
        Nominal case (Linear and Non-linear, with and without renorm)
        Negative scalar
        Inplace
        Inplace approximation
    """

    def setUp(self) -> None:
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.delta_ang = np.array([0.01, 0.02, 0.03])
        self.quat_new_approx = np.array([0.005, 0.01, 0.015, 1.0])
        self.quat_new_appx_norm = np.array([0.00499912522962, 0.00999825045924, 0.01499737568886, 0.99982504592411])
        self.quat_new = np.array([0.004999708338, 0.009999416677, 0.014999125015, 0.999825005104])

    def test_nominal(self) -> None:
        quat = space.quat_prop_single(self.quat, self.delta_ang)
        np.testing.assert_array_almost_equal(quat, self.quat_new, 12)
        quat_norm = space.quat_prop_single(self.quat, self.delta_ang, use_approx=True, renorm=False)
        np.testing.assert_array_almost_equal(quat_norm, self.quat_new_approx, 12)
        quat_norm = space.quat_prop_single(self.quat, self.delta_ang, use_approx=True)
        np.testing.assert_array_almost_equal(quat_norm, self.quat_new_appx_norm, 12)

    def test_negative_scalar(self) -> None:
        quat = space.quat_prop_single(np.array([1.0, 0.0, 0.0, 0.0]), self.delta_ang)
        self.assertGreater(quat[3], 0)
        quat = space.quat_prop_single(np.array([1.0, 0.0, 0.0, 0.0]), -self.delta_ang)
        self.assertGreater(quat[3], 0)

    def test_inplace(self) -> None:
        quat = space.quat_prop_single(self.quat, self.delta_ang)
        self.assertGreater(np.max(np.abs(quat - self.quat)), 0.004)
        quat = space.quat_prop_single(self.quat, self.delta_ang, inplace=True)
        self.assertIs(quat, self.quat)
        self.assertLess(np.max(np.abs(quat - self.quat_new)), 1e-12)

    def test_inplace_approx(self) -> None:
        quat = space.quat_prop_single(self.quat, self.delta_ang, use_approx=True, inplace=True)
        self.assertIs(quat, self.quat)
        self.assertLess(np.max(np.abs(quat - self.quat_new_appx_norm)), 1e-12)


# %% aerospace.quat_times_vector_single
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_times_vector_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_times_vector_single function with the following cases:
        Integers
        Nominal
        Inplace
    """

    def setUp(self) -> None:
        # TODO: confirm that this is enough to test the correctness of the function
        self.quat = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).T
        self.vec = np.array([[1, 0, 0], [2, 0, 0]]).T
        self.out = np.array([[-1, 2], [0, 0], [0, 0]])

    def test_integers(self) -> None:
        # Expected to fail until numba supports @ for matrix multiplication for integers.
        for i in range(2):  # pragma: no cover
            vec = space.quat_times_vector_single(self.quat[:, i], self.vec[:, i])
            np.testing.assert_array_almost_equal(vec, self.out[:, i])

    if HAVE_NUMBA:
        test_integers = unittest.expectedFailure(test_integers)

    def test_nominal(self) -> None:
        for i in range(2):
            vec = space.quat_times_vector_single(self.quat[:, i].astype(float), self.vec[:, i].astype(float))
            np.testing.assert_array_almost_equal(vec, self.out[:, i].astype(float))

    def test_inplace(self) -> None:
        q = self.quat[:, 0].astype(float)
        v = self.vec[:, 0].astype(float)
        vec = space.quat_times_vector_single(q, v)
        self.assertGreater(np.max(np.abs(vec - v)), 0.004)
        vec = space.quat_times_vector_single(q, v, inplace=True)
        self.assertIs(vec, v)
        self.assertLess(np.max(np.abs(vec - self.out[:, 0])), 1e-8)


# %% aerospace.quat_to_dcm
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_quat_to_dcm(unittest.TestCase):
    r"""
    Tests the aerospace.quat_to_dcm function with the following cases:
        Nominal case
    """

    def setUp(self) -> None:
        self.quat = np.array([0.5, -0.5, 0.5, 0.5])
        self.dcm = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

    def test_nominal(self) -> None:
        dcm = space.quat_to_dcm(self.quat)
        np.testing.assert_array_almost_equal(dcm, self.dcm)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
