r"""
Test file for the `quat_keras` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in October 2024.

"""

# %% Imports
import unittest

from dstauffman import HAVE_KERAS, HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_KERAS:
    import keras.ops as ops
if HAVE_NUMPY:
    import numpy as np


# %% aerospace.enforce_pos_scalar_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_enforce_pos_scalar_keras(unittest.TestCase):
    r"""
    Tests the aerospace.enforce_pos_scalar_keras function with the following cases:
        Single quat
        Multi quats
    """

    def setUp(self) -> None:
        self.q1_inp = ops.array([-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2], dtype="float64")
        self.q1_out = ops.array([-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2], dtype="float64")
        self.q2_inp = ops.array([0, 0.5, 0, -np.sqrt(3) / 2], dtype="float64")
        self.q2_out = ops.array([0, -0.5, 0, np.sqrt(3) / 2], dtype="float64")

    def test_nominal(self) -> None:
        q1_out = space.enforce_pos_scalar_keras(self.q1_inp)
        np.testing.assert_array_almost_equal(q1_out, self.q1_out)
        q2_out = space.enforce_pos_scalar_keras(self.q2_inp)
        np.testing.assert_array_almost_equal(q2_out, self.q2_out)

    def test_quat_array(self) -> None:
        quat = ops.stack([self.q1_inp, self.q2_inp])
        qout = space.enforce_pos_scalar_keras(quat)
        exp = ops.stack([self.q1_out, self.q2_out])
        np.testing.assert_array_almost_equal(qout, exp)


# %% aerospace.qrot_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_qrot(unittest.TestCase):
    r"""
    Tests the aerospace.qrot_keras function with the following cases:
        Single input case
        Single axis, multiple angles
        Multiple axes, single angle
        Multiple axes, multiple angles
        Null (x2)
        Vector mismatch (causes AssertionError)
    """

    def setUp(self) -> None:
        # fmt: off
        self.axis      = ops.array([1, 2, 3])
        self.angle     = ops.array(np.pi / 2, dtype="float64")
        self.angle2    = ops.array(np.pi / 3, dtype="float64")
        r2o2           = ops.array(np.sqrt(2) / 2, dtype="float64")
        r3o2           = ops.array(np.sqrt(3) / 2, dtype="float64")
        self.quat      = ops.array([[r2o2, 0, 0, r2o2], [0, r2o2, 0, r2o2], [0, 0, r2o2, r2o2]])
        self.quat2     = ops.array([[0.5, 0, 0, r3o2], [0, 0.5, 0, r3o2], [0, 0, 0.5, r3o2]])
        # fmt: on

    def test_single_inputs(self) -> None:
        for i in range(len(self.axis)):
            quat = space.qrot_keras(self.axis[i], self.angle)
            self.assertEqual(quat.ndim, 1)
            np.testing.assert_array_almost_equal(quat, self.quat[i, :])

    # def test_single_axis(self) -> None:
    #     for i in range(len(self.axis)):
    #         quat = space.qrot_keras(self.axis[i], np.array([self.angle, self.angle2]))
    #         self.assertEqual(quat.ndim, 2)
    #         np.testing.assert_array_almost_equal(quat, ops.stack([self.quat[i, :], self.quat2[i, :]]))

    # def test_single_angle(self) -> None:
    #     quat = space.qrot_keras(self.axis, self.angle)
    #     self.assertEqual(quat.ndim, 2)
    #     np.testing.assert_array_almost_equal(quat, self.quat.T)

    # def test_all_vector_inputs(self) -> None:
    #     quat = space.qrot_keras(self.axis, np.array([self.angle, self.angle, self.angle2]))
    #     np.testing.assert_array_almost_equal(quat, ops.stack([self.quat[0, :], self.quat[1, :], self.quat2[2, :]]))

    # def test_vector_mismatch(self) -> None:
    #     with self.assertRaises(AssertionError):
    #         space.qrot_keras(self.axis, np.array([self.angle, self.angle2]))

    def test_larger_angle(self) -> None:
        quat = space.qrot_keras(1, 5.1 * np.pi)
        self.assertGreater(quat[3], 0)


# %% aerospace.quat_inv_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_quat_inv_keras(unittest.TestCase):
    r"""
    Tests the aerospace.quat_inv_keras function with the following cases:
        Single quat (x2 different quats)
        Array of quats
    """

    def setUp(self) -> None:
        self.q1_inp = space.qrot_keras(1, np.pi / 2)
        self.q1_out = ops.array([-ops.sqrt(2) / 2, 0, 0, ops.sqrt(2) / 2])
        self.q2_inp = space.qrot_keras(2, np.pi / 3)
        self.q2_out = ops.array([0, -0.5, 0, ops.sqrt(3) / 2])
        self.q3_inp = ops.stack([self.q1_inp, self.q2_inp])
        self.q3_out = ops.stack([self.q1_out, self.q2_out])

    def test_single_quat1(self) -> None:
        q1_inv = space.quat_inv_keras(self.q1_inp)
        np.testing.assert_array_almost_equal(q1_inv, self.q1_out)
        self.assertEqual(q1_inv.ndim, 1)
        np.testing.assert_array_equal(q1_inv.shape, self.q1_out.shape)

    def test_single_quat2(self) -> None:
        q2_inv = space.quat_inv_keras(self.q2_inp)
        np.testing.assert_array_almost_equal(q2_inv, self.q2_out)
        self.assertEqual(q2_inv.ndim, 1)
        np.testing.assert_array_equal(q2_inv.shape, self.q2_out.shape)

    def test_quat_array(self) -> None:
        q3_inv = space.quat_inv_keras(self.q3_inp)
        np.testing.assert_array_almost_equal(q3_inv, self.q3_out)
        self.assertEqual(q3_inv.ndim, 2)
        np.testing.assert_array_equal(q3_inv.shape, self.q3_out.shape)


# %% aerospace.quat_norm_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_quat_norm_keras(unittest.TestCase):
    r"""
    Tests the aerospace.quat_norm_keras function with the following cases:
        Single quat (x3 different quats)
        Inplace
    """

    def setUp(self) -> None:
        self.q1_inp = space.qrot_keras(1, np.pi / 2)
        self.q1_out = ops.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.q2_inp = space.qrot_keras(2, np.pi / 3)
        self.q2_out = ops.array([0, 0.5, 0, np.sqrt(3) / 2])
        self.q3_inp = ops.array([0.1, 0, 0, 1])
        self.q3_out = ops.array([0.09950372, 0, 0, 0.99503719])
        self.q4_inp = ops.stack([self.q1_inp, self.q2_inp, self.q3_inp])
        self.q4_out = ops.stack([self.q1_out, self.q2_out, self.q3_out])

    def test_nominal1(self) -> None:
        quat_norm = space.quat_norm_keras(self.q1_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q1_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q1_out.shape)

    def test_nominal2(self) -> None:
        quat_norm = space.quat_norm_keras(self.q2_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q2_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q2_out.shape)

    def test_nominal3(self) -> None:
        quat_norm = space.quat_norm_keras(self.q3_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q3_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q3_out.shape)

    def test_array(self) -> None:
        quat_norm = space.quat_norm_keras(self.q4_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q4_out)
        self.assertEqual(quat_norm.ndim, 2)
        np.testing.assert_array_equal(quat_norm.shape, self.q4_out.shape)


# %% aerospace.quat_mult_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_quat_mult_keras(unittest.TestCase):
    r"""
    Tests the aerospace.quat_mult_keras function with the following cases:
        Single quat (x2 different quats)
        Reverse order
        Quat array times scalar (x2 orders)
    """

    def setUp(self) -> None:
        self.q1 = space.qrot_keras(1, ops.array(np.pi / 2, dtype="float64"))
        self.q2 = space.qrot_keras(2, ops.array(-np.pi, dtype="float64"))
        self.q3 = space.qrot_keras(3, ops.array(np.pi / 3, dtype="float64"))
        self.q4 = ops.array([0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0])  # q1*q2
        self.q5 = ops.array([0.5, -np.sqrt(3) / 2, 0, 0])  # q2*q3
        self.q6 = ops.array([0.5, 0.5, 0.5, 0.5])  # q6 * q6 = q6**-1, and triggers negative scalar component
        self.q_array_in1 = ops.stack([self.q1, self.q2])
        self.q_array_in2 = ops.stack([self.q2, self.q3])
        self.q_array_out = ops.stack([self.q4, self.q5])

    def test_nominal1(self) -> None:
        quat = space.quat_mult_keras(self.q1, self.q2)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q4)
        np.testing.assert_array_equal(quat.shape, self.q4.shape)

    def test_nominal2(self) -> None:
        quat = space.quat_mult_keras(self.q2, self.q3)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q5)
        np.testing.assert_array_equal(quat.shape, self.q5.shape)

    def test_nominal3(self) -> None:
        quat = space.quat_mult_keras(self.q6, self.q6)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, space.quat_inv_keras(self.q6))
        np.testing.assert_array_equal(quat.shape, self.q6.shape)

    def test_reverse(self) -> None:
        quat1 = space.quat_mult_keras(self.q2, self.q1)
        quat2 = space.quat_inv_keras(space.quat_mult_keras(space.quat_inv_keras(self.q1), space.quat_inv_keras(self.q2)))
        np.testing.assert_array_almost_equal(quat1, quat2)

    def test_array_scalar1(self) -> None:
        quat = space.quat_mult_keras(self.q_array_in1, self.q2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat[0, :], self.q4)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_array_scalar2(self) -> None:
        quat = space.quat_mult_keras(self.q1, self.q_array_in2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat[0, :], self.q4)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_array_scalar3(self) -> None:
        quat = space.quat_mult_keras(self.q6, ops.stack([self.q6, self.q6]))
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, ops.stack([space.quat_inv_keras(self.q6), space.quat_inv_keras(self.q6)]))
        np.testing.assert_array_equal(quat.shape, (2, 4))

    def test_array(self) -> None:
        quat = space.quat_mult_keras(self.q_array_in1, self.q_array_in2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, self.q_array_out)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)


# %% aerospace.quat_prop_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_quat_prop_keras(unittest.TestCase):
    r"""
    Tests the aerospace.quat_prop_keras function with the following cases:
        Nominal
        Array (x3)
    """

    def setUp(self) -> None:
        self.quat = ops.array([0.0, 0.0, 0.0, 1.0], dtype="float64")
        self.delta_ang = ops.array([0.01, 0.02, 0.03], dtype="float64")
        self.new_quat1 = ops.array([0.004999708338, 0.009999416677, 0.014999125015, 0.999825005104], dtype="float64")
        self.new_quat2 = ops.array([-0.004999708338, -0.009999416677, -0.014999125015, 0.999825005104], dtype="float64")

    def test_nominal(self) -> None:
        quat = space.quat_prop_keras(self.quat, self.delta_ang)
        np.testing.assert_array_almost_equal(quat, self.new_quat1, 12)
        quat = space.quat_prop_keras(self.quat, -self.delta_ang)
        np.testing.assert_array_almost_equal(quat, self.new_quat2, 12)

    def test_negative_scalar(self) -> None:
        quat = space.quat_prop_keras(ops.array([1.0, 0.0, 0.0, 0.0]), ops.array([0.01, 0.02, 0.03]))
        self.assertGreater(quat[3], 0)
        quat = space.quat_prop_keras(ops.array([1.0, 0.0, 0.0, 0.0]), ops.array([-0.01, -0.02, -0.03]))
        self.assertGreater(quat[3], 0)

    def test_array(self) -> None:
        quat = space.quat_prop_keras(ops.stack([self.quat, self.quat]), ops.stack([self.delta_ang, -self.delta_ang]))
        np.testing.assert_array_almost_equal(quat, ops.stack([self.new_quat1, self.new_quat2]), 12)


# %% aerospace.quat_times_vector_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_quat_times_vector_keras(unittest.TestCase):
    r"""
    Tests the aerospace.quat_times_vector_keras function with the following cases:
        Integers
        Nominal
    """

    def setUp(self) -> None:
        # TODO: confirm that this is enough to test the correctness of the function
        self.quat = ops.array([[0, 1, 0, 0], [1, 0, 0, 0]])
        self.vec = ops.array([[1, 0, 0], [2, 0, 0]])
        self.out = ops.array([[-1, 0, 0], [2, 0, 0]])

    def test_integers(self) -> None:
        for i in range(2):
            vec = space.quat_times_vector_keras(self.quat[i, :], self.vec[i, :])
            np.testing.assert_array_almost_equal(vec, self.out[i, :])

    def test_nominal(self) -> None:
        for i in range(2):
            vec = space.quat_times_vector_keras(
                ops.array(self.quat[i, :], dtype="float64"), ops.array(self.vec[i, :], dtype="float64")
            )
            np.testing.assert_array_almost_equal(vec, ops.array(self.out[i, :], dtype="float64"))

    def test_array(self) -> None:
        vec = space.quat_times_vector_keras(self.quat, self.vec)
        np.testing.assert_array_almost_equal(vec, self.out)

    def test_vector_array(self) -> None:
        quat = self.quat[0, :]
        vec1 = space.quat_times_vector_keras(quat, self.vec)
        vec2 = space.quat_times_vector_keras(ops.stack([quat, quat]), self.vec)
        np.testing.assert_array_almost_equal(vec1, vec2)

    def test_array_vector(self) -> None:
        vec = self.vec[0, :]
        vec1 = space.quat_times_vector_keras(self.quat, vec)
        vec2 = space.quat_times_vector_keras(self.quat, ops.stack([vec, vec]))
        np.testing.assert_array_almost_equal(vec1, vec2)


# %% aerospace.quat_angle_diff_keras
@unittest.skipIf(not HAVE_KERAS, "Skipping due to missing keras dependency.")
class Test_aerospace_quat_angle_diff_keras(unittest.TestCase):
    r"""
    Tests the aerospace.quat_angle_diff_keras function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        # fmt: off
        self.quat1 = ops.array([0.5, 0.5, 0.5, 0.5], dtype="float64")
        self.dq1   = space.qrot_keras(1, ops.array(0.001, dtype="float64"))
        self.dq2   = space.qrot_keras(2, ops.array(0.05, dtype="float64"))
        self.dqq1  = space.quat_mult_keras(self.dq1, self.quat1)
        self.dqq2  = space.quat_mult_keras(self.dq2, self.quat1)
        self.comp  = ops.array([[0.001, 0.0, 0.0], [0.0, 0.05, 0.0]], dtype="float64")
        # fmt: on

    def test_nominal1(self) -> None:
        comp = space.quat_angle_diff_keras(self.quat1, self.dqq1)
        np.testing.assert_array_almost_equal(comp, self.comp[0, :])

    def test_nominal2(self) -> None:
        comp = space.quat_angle_diff_keras(self.quat1, self.dqq2)
        np.testing.assert_array_almost_equal(comp, self.comp[1, :])

    def test_array1(self) -> None:
        comp = space.quat_angle_diff_keras(ops.stack([self.dqq1, self.dqq2]), self.quat1)
        np.testing.assert_array_almost_equal(comp, -self.comp)

    def test_array2(self) -> None:
        comp = space.quat_angle_diff_keras(self.quat1, ops.stack([self.dqq1, self.dqq2]))
        np.testing.assert_array_almost_equal(comp, self.comp)

    def test_array3(self) -> None:
        comp = space.quat_angle_diff_keras(
            ops.stack([self.quat1, self.quat1, self.dqq1, self.dqq2]),
            ops.stack([self.dqq1, self.dqq2, self.quat1, self.quat1]),
        )
        exp = ops.tile(self.comp, (2, 1)) * ops.array([[1], [1], [-1], [-1]], dtype="float64")
        np.testing.assert_array_almost_equal(comp, exp)

    def test_zero_diff1(self) -> None:
        comp = space.quat_angle_diff_keras(self.quat1, self.quat1)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff2(self) -> None:
        comp = space.quat_angle_diff_keras(self.quat1, ops.stack([self.quat1, self.quat1]))
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff3(self) -> None:
        comp = space.quat_angle_diff_keras(ops.stack([self.quat1, self.quat1]), self.quat1)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff4(self) -> None:
        temp = ops.stack([self.quat1, self.dq1, self.dq2, self.dqq1, self.dqq2])
        comp = space.quat_angle_diff_keras(temp, temp)
        np.testing.assert_array_almost_equal(comp, 0)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
