r"""
Test file for the `quat` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import unittest

import numpy as np

from dstauffman import capture_output
import dstauffman.aerospace as space

#%% aerospace.QUAT_SIZE
class Test_aerospace_QUAT_SIZE(unittest.TestCase):
    r"""
    Tests the aerospace.QUAT_SIZE function with the following cases:
        Exists and is 4
    """
    def test_exists(self):
        self.assertEqual(space.QUAT_SIZE, 4)

#%% aerospace.USE_ASSERTIONS
class Test_aerospace_USE_ASSERTIONS(unittest.TestCase):
    r"""
    Tests the aerospace.USE_ASSERTIONS function with the following cases:
        Exists and is boolean
    """
    def test_exists(self):
        self.assertTrue(isinstance(space.USE_ASSERTIONS, bool))

#%% aerospace.quat_assertions
class Test_aerospace_quat_assertions(unittest.TestCase):
    r"""
    Tests the aerospace.quat_assertions function with the following cases:
        Nominal (x2)
        Array (x2)
        Bad (x7)
    """
    def setUp(self):
        self.q1 = np.array([0, 0, 0, 1]) # zero quaternion
        self.q2 = np.array([0.5, 0.5, 0.5, 0.5]) # normal 1D quat
        self.q3 = np.array([0.5, -0.5, -0.5, 0.5]) # normal 1D quat with some negative components
        self.q4 = np.column_stack((self.q1, self.q2, self.q3)) # 2D array
        self.q5 = np.array([[0.5],[0.5],[0.5],[0.5]]) # 2D, single quat
        self.q6 = np.array([0, 0, 0, -1]) # zero quaternion with bad scalar
        self.q7 = np.array([-5, -5, 5, 5]) # quat with bad ranges
        self.q8 = np.array([0.5, 0.5j, 0.5, 0.5], dtype=np.complex128)
        self.q9 = np.array([0, 0, 1]) # only a 3 element vector
        self.q10 = np.array([0, 0, 0]) # only a 3 element vector, but with zero magnitude
        self.q11 = np.array([[[0.5],[0.5],[0.5],[0.5]]]) # valid, but 3D instead
        self.q12 = np.column_stack((self.q1, self.q6, self.q7)) # good and bad combined

    def test_nominal1(self):
        space.quat_assertions(self.q1)

    def test_nominal2(self):
        space.quat_assertions(self.q2)

    def test_nominal3(self):
        space.quat_assertions(self.q3)

    def test_array1(self):
        space.quat_assertions(self.q4)

    def test_array2(self):
        space.quat_assertions(self.q5)

    def test_bad1(self):
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q6)

    def test_bad2(self):
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q7)

    def test_bad3(self):
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q8)

    def test_bad4(self):
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q9)

    def test_bad5(self):
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q10)

    def test_bad6(self):
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q11)

    def test_bad7(self):
        with self.assertRaises(AssertionError):
            space.quat_assertions(self.q12)

    def test_skip_assertions(self):
        space.quat_assertions(self.q6, skip_assertions=True)

#%% aerospace.qrot
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
    def setUp(self):
        self.axis   = np.array([1, 2, 3])
        self.angle  = np.pi/2
        self.angle2 = np.pi/3
        r2o2        = np.sqrt(2)/2
        r3o2        = np.sqrt(3)/2
        self.quat   = np.array([[r2o2, 0, 0, r2o2], [0, r2o2, 0, r2o2], [0, 0, r2o2, r2o2]])
        self.quat2  = np.array([[ 0.5, 0, 0, r3o2], [0,  0.5, 0, r3o2], [0, 0,  0.5, r3o2]])
        self.null   = np.array([], dtype=int)
        self.null_quat = np.zeros((4, 0))

    def test_single_inputs(self):
        for i in range(len(self.axis)):
            quat = space.qrot(self.axis[i], self.angle)
            self.assertEqual(quat.ndim, 1)
            np.testing.assert_array_almost_equal(quat, self.quat[i, :])

    def test_single_axis(self):
        for i in range(len(self.axis)):
            quat = space.qrot(self.axis[i], np.array([self.angle, self.angle2]))
            self.assertEqual(quat.ndim, 2)
            np.testing.assert_array_almost_equal(quat, np.column_stack((self.quat[i, :], self.quat2[i, :])))

    def test_single_angle(self):
        quat = space.qrot(self.axis, self.angle)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, self.quat.T)

    def test_all_vector_inputs(self):
        quat = space.qrot(self.axis, np.array([self.angle, self.angle, self.angle2]))
        np.testing.assert_array_almost_equal(quat, np.column_stack((self.quat[0,:], self.quat[1,:], self.quat2[2,:])))

    def test_null1(self):
        quat = space.qrot(self.axis[0], self.null)
        np.testing.assert_array_almost_equal(quat, self.null_quat)

    def test_null2(self):
        quat = space.qrot(self.null, self.angle)
        np.testing.assert_array_almost_equal(quat, self.null_quat)

    def test_vector_mismatch(self):
        with self.assertRaises(AssertionError):
            space.qrot(self.axis, np.array([self.angle, self.angle2]))

#%% aerospace.quat_angle_diff
class Test_aerospace_quat_angle_diff(unittest.TestCase):
    r"""
    Tests the aerospace.quat_angle_diff function with the following cases:
        Nominal (x2)
        Array (x3)
        Zero diff (x4)
        Null (x4)
    """
    def setUp(self):
        self.quat1 = np.array([0.5, 0.5, 0.5, 0.5])
        self.dq1   = space.qrot(1, 0.001)
        self.dq2   = space.qrot(2, 0.05)
        self.dqq1  = space.quat_mult(self.dq1, self.quat1)
        self.dqq2  = space.quat_mult(self.dq2, self.quat1)
        self.theta = np.array([0.001, 0.05])
        self.comp  = np.array([[0.001, 0], [0, 0.05], [0, 0]])
        self.null   = np.array([])
        self.null_quat = np.zeros((4, 0))

    def test_nominal1(self):
        (theta, comp) = space.quat_angle_diff(self.quat1, self.dqq1)
        np.testing.assert_array_almost_equal(theta, self.theta[0])
        np.testing.assert_array_almost_equal(comp, self.comp[:, 0])

    def test_nominal2(self):
        (theta, comp) = space.quat_angle_diff(self.quat1, self.dqq2)
        np.testing.assert_array_almost_equal(theta, self.theta[1])
        np.testing.assert_array_almost_equal(comp, self.comp[:, 1])

    def test_array1(self):
        (theta, comp) = space.quat_angle_diff(np.column_stack((self.dqq1, self.dqq2)), self.quat1)
        np.testing.assert_array_almost_equal(theta, self.theta)
        np.testing.assert_array_almost_equal(comp, -self.comp)

    def test_array2(self):
        (theta, comp) = space.quat_angle_diff(self.quat1, np.column_stack((self.dqq1, self.dqq2)))
        np.testing.assert_array_almost_equal(theta, self.theta)
        np.testing.assert_array_almost_equal(comp, self.comp)

    def test_array3(self):
        (theta, comp) = space.quat_angle_diff(np.column_stack((self.quat1, self.quat1, self.dqq1, self.dqq2)), \
            np.column_stack((self.dqq1, self.dqq2, self.quat1, self.quat1)))
        np.testing.assert_array_almost_equal(theta, self.theta[[0, 1, 0, 1]])
        np.testing.assert_array_almost_equal(comp, self.comp[:,[0, 1, 0, 1]] * np.array([1, 1, -1, -1]))

    def test_zero_diff1(self):
        (theta, comp) = space.quat_angle_diff(self.quat1, self.quat1)
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff2(self):
        (theta, comp) = space.quat_angle_diff(self.quat1, np.column_stack((self.quat1, self.quat1)))
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff3(self):
        (theta, comp) = space.quat_angle_diff(np.column_stack((self.quat1, self.quat1)), self.quat1)
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_zero_diff4(self):
        temp = np.column_stack((self.quat1, self.dq1, self.dq2, self.dqq1, self.dqq2))
        (theta, comp) = space.quat_angle_diff(temp, temp)
        np.testing.assert_array_almost_equal(theta, 0)
        np.testing.assert_array_almost_equal(comp, 0)

    def test_null1(self):
        (theta, comp) = space.quat_angle_diff(self.quat1, self.null)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0, ))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))

    def test_null2(self):
        (theta, comp) = space.quat_angle_diff(self.quat1, self.null_quat)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0, ))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))

    def test_null3(self):
        (theta, comp) = space.quat_angle_diff(self.null, self.quat1)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0, ))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))

    def test_null4(self):
        (theta, comp) = space.quat_angle_diff(self.null_quat, self.quat1)
        self.assertEqual(theta.size, 0)
        self.assertEqual(theta.shape, (0, ))
        self.assertEqual(comp.size, 0)
        self.assertEqual(comp.shape, (3, 0))

#%% aerospace.quat_from_euler
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
    def setUp(self):
        self.a      = np.array([0.01, 0.02, 0.03])
        self.b      = np.array([0.04, 0.05, 0.06])
        self.angles = np.column_stack((self.a, self.b))
        self.seq    = np.array([3, 2, 1])
        self.quat   = np.array([\
            [0.01504849, 0.03047982],
            [0.00992359, 0.02438147],
            [0.00514916, 0.02073308],
            [0.99982426, 0.99902285]])

    def test_nominal1(self):
        quat = space.quat_from_euler(self.a, self.seq)
        np.testing.assert_array_almost_equal(quat, self.quat[:,0])
        self.assertEqual(quat.ndim, 1)

    def test_nominal2(self):
        quat = space.quat_from_euler(self.b, self.seq)
        np.testing.assert_array_almost_equal(quat, self.quat[:,1])
        self.assertEqual(quat.ndim, 1)

    def test_default_seq(self):
        quat = space.quat_from_euler(self.a)
        temp = space.quat_mult(space.quat_mult(space.qrot(3, self.a[0]), space.qrot(1, self.a[1])), space.qrot(2, self.a[2]))
        np.testing.assert_array_almost_equal(quat, temp)
        self.assertEqual(quat.ndim, 1)

    def test_repeated(self):
        quat1 = space.quat_from_euler(np.hstack((self.a, self.a)), seq=np.array([1, 1, 1, 1, 1, 1]))
        quat2 = space.qrot(1, 2*np.sum(self.a))
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_short(self):
        quat1 = space.quat_from_euler(self.a[0:2], self.seq[0:2])
        quat2 = space.quat_mult(space.qrot(self.seq[0], self.a[0]), space.qrot(self.seq[1], self.a[1]))
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_single1(self):
        quat1 = space.quat_from_euler(self.a[0], self.seq[0])
        quat2 = space.qrot(self.seq[0], self.a[0])
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_single2(self):
        quat1 = space.quat_from_euler(0.01, 3)
        quat2 = space.qrot(3, 0.01)
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_long(self):
        quat1 = space.quat_from_euler(np.hstack((self.a, self.b)), seq=np.hstack((self.seq, self.seq)))
        quat2 = space.quat_mult(self.quat[:,0], self.quat[:,1])
        np.testing.assert_array_almost_equal(quat1, quat2)
        self.assertEqual(quat1.ndim, 1)

    def test_array1(self):
        quat = space.quat_from_euler(self.angles, self.seq)
        np.testing.assert_array_almost_equal(quat, self.quat)
        self.assertEqual(quat.ndim, 2)

    def test_array2(self):
        quat = space.quat_from_euler(np.expand_dims(self.a, axis=1), self.seq)
        np.testing.assert_array_almost_equal(quat, np.expand_dims(self.quat[:,0], axis=1))
        self.assertEqual(quat.ndim, 2)

    def test_array3(self):
        with self.assertRaises(ValueError):
            space.quat_from_euler(np.zeros((3,3,1)))

#%% aerospace.quat_interp
class Test_aerospace_quat_interp(unittest.TestCase):
    r"""
    Tests the aerospace.quat_interp function with the following cases:
        TBD
    """
    def setUp(self):
        self.time = np.array([1, 3, 5])
        self.quat = np.column_stack((space.qrot(1, 0), space.qrot(1, np.pi/2), space.qrot(1, np.pi)))
        self.ti   = np.array([1, 2, 4.5, 5])
        self.qout = np.column_stack((space.qrot(1, 0), space.qrot(1, np.pi/4), space.qrot(1, 3.5/4*np.pi), space.qrot(1, np.pi)))
        self.ti_extra = np.array([0, 1, 2, 4.5, 5, 10])

    def test_nominal(self):
        qout = space.quat_interp(self.time, self.quat, self.ti)
        np.testing.assert_array_almost_equal(qout, self.qout)

    def test_empty(self):
        qout = space.quat_interp(self.time, self.quat, np.array([]))
        self.assertEqual(qout.size, 0)

    def test_scalar1(self):
        qout = space.quat_interp(self.time, self.quat, self.ti[0])
        np.testing.assert_array_almost_equal(qout, np.expand_dims(self.qout[:,0],1))

    def test_scalar2(self):
        qout = space.quat_interp(self.time, self.quat, self.ti[1])
        np.testing.assert_array_almost_equal(qout, np.expand_dims(self.qout[:,1],1))

    def test_extra1(self):
        with self.assertRaises(ValueError):
            space.quat_interp(self.time, self.quat, self.ti_extra, inclusive=True)

    def test_extra2(self):
        with capture_output() as out:
            qout = space.quat_interp(self.time, self.quat, self.ti_extra, inclusive=False)
        output = out.getvalue().strip()
        out.close()
        np.testing.assert_array_almost_equal(qout[:, 1:-1], self.qout)
        np.testing.assert_array_equal(qout[:,[0, -1]], np.nan)
        self.assertEqual(output, 'Desired time not found within input time vector.')

#%% aerospace.quat_inv
class Test_aerospace_quat_inv(unittest.TestCase):
    r"""
    Tests the aerospace.quat_inv function with the following cases:
        Single quat (x2 different quats)
        Quat array
        Null (x2 different null sizes)
    """
    def setUp(self):
        self.q1_inp = space.qrot(1, np.pi/2)
        self.q1_out = np.array([-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        self.q2_inp = space.qrot(2, np.pi/3)
        self.q2_out = np.array([0, -0.5, 0, np.sqrt(3)/2])
        self.q3_inp = np.column_stack((self.q1_inp, self.q2_inp))
        self.q3_out = np.column_stack((self.q1_out, self.q2_out))
        self.null   = np.array([])
        self.null_quat = np.ones((space.QUAT_SIZE, 0))

    def test_single_quat1(self):
        q1_inv = space.quat_inv(self.q1_inp)
        np.testing.assert_array_almost_equal(q1_inv, self.q1_out)
        self.assertEqual(q1_inv.ndim, 1)
        np.testing.assert_array_equal(q1_inv.shape, self.q1_out.shape)

    def test_single_quat2(self):
        q2_inv = space.quat_inv(self.q2_inp)
        np.testing.assert_array_almost_equal(q2_inv, self.q2_out)
        self.assertEqual(q2_inv.ndim, 1)
        np.testing.assert_array_equal(q2_inv.shape, self.q2_out.shape)

    def test_quat_array(self):
        q3_inv = space.quat_inv(self.q3_inp)
        np.testing.assert_array_almost_equal(q3_inv, self.q3_out)
        self.assertEqual(q3_inv.ndim, 2)
        np.testing.assert_array_equal(q3_inv.shape, self.q3_out.shape)

    def test_null_input1(self):
        null_inv = space.quat_inv(self.null_quat)
        np.testing.assert_array_equal(null_inv, self.null_quat)
        np.testing.assert_array_equal(null_inv.shape, self.null_quat.shape)

    def test_null_input2(self):
        null_inv = space.quat_inv(self.null)
        np.testing.assert_array_equal(null_inv, self.null)
        np.testing.assert_array_equal(null_inv.shape, self.null.shape)

#%% aerospace.quat_mult
class Test_aerospace_quat_mult(unittest.TestCase):
    r"""
    Tests the aerospace.quat_mult function with the following cases:
        Single quat (x2 different quats)
        Reverse order
        Quat array times scalar (x2 orders + x1 array-array)
        Null (x8 different null size and order permutations)
    """
    def setUp(self):
        self.q1 = space.qrot(1, np.pi/2)
        self.q2 = space.qrot(2, -np.pi)
        self.q3 = space.qrot(3, np.pi/3)
        self.q4 = np.array([ 0, -np.sqrt(2)/2, np.sqrt(2)/2, 0]) # q1*q2
        self.q5 = np.array([0.5, -np.sqrt(3)/2, 0, 0]) # q2*q3
        self.q6 = np.array([0.5, 0.5, 0.5, 0.5]) # q6 * q6 = q6**-1, and triggers negative scalar component
        self.q_array_in1 = np.column_stack((self.q1, self.q2))
        self.q_array_in2 = np.column_stack((self.q2, self.q3))
        self.q_array_out = np.column_stack((self.q4, self.q5))
        self.null        = np.array([])
        self.null_quat   = np.ones((space.QUAT_SIZE, 0))

    def test_nominal1(self):
        quat = space.quat_mult(self.q1, self.q2)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q4)
        np.testing.assert_array_equal(quat.shape, self.q4.shape)

    def test_nominal2(self):
        quat = space.quat_mult(self.q2, self.q3)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, self.q5)
        np.testing.assert_array_equal(quat.shape, self.q5.shape)

    def test_nominal3(self):
        quat = space.quat_mult(self.q6, self.q6)
        self.assertEqual(quat.ndim, 1)
        np.testing.assert_array_almost_equal(quat, space.quat_inv(self.q6))
        np.testing.assert_array_equal(quat.shape, self.q6.shape)

    def test_reverse(self):
        quat1 = space.quat_mult(self.q2, self.q1)
        quat2 = space.quat_inv(space.quat_mult(space.quat_inv(self.q1), space.quat_inv(self.q2)))
        np.testing.assert_array_almost_equal(quat1, quat2)

    def test_array_scalar1(self):
        quat = space.quat_mult(self.q_array_in1, self.q2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat[:,0], self.q4)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_array_scalar2(self):
        quat = space.quat_mult(self.q1, self.q_array_in2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat[:,0], self.q4)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_array_scalar3(self):
        quat = space.quat_mult(self.q6, np.column_stack((self.q6, self.q6)))
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, np.column_stack((space.quat_inv(self.q6), space.quat_inv(self.q6))))
        np.testing.assert_array_equal(quat.shape, (4, 2))

    def test_array(self):
        quat = space.quat_mult(self.q_array_in1, self.q_array_in2)
        self.assertEqual(quat.ndim, 2)
        np.testing.assert_array_almost_equal(quat, self.q_array_out)
        np.testing.assert_array_equal(quat.shape, self.q_array_out.shape)

    def test_null_input1(self):
        quat = space.quat_mult(self.null_quat, self.q2)
        np.testing.assert_array_equal(quat, self.null_quat)
        np.testing.assert_array_equal(quat.shape, self.null_quat.shape)

    def test_null_input2(self):
        quat = space.quat_mult(self.null, self.q2)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input3(self):
        quat = space.quat_mult(self.q1, self.null_quat)
        np.testing.assert_array_equal(quat, self.null_quat)
        np.testing.assert_array_equal(quat.shape, self.null_quat.shape)

    def test_null_input4(self):
        quat = space.quat_mult(self.q2, self.null)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input5(self):
        quat = space.quat_mult(self.null_quat, self.null_quat)
        np.testing.assert_array_equal(quat, self.null_quat)
        np.testing.assert_array_equal(quat.shape, self.null_quat.shape)

    def test_null_input6(self):
        quat = space.quat_mult(self.null, self.null)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input7(self):
        quat = space.quat_mult(self.null_quat, self.null)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

    def test_null_input8(self):
        quat = space.quat_mult(self.null, self.null_quat)
        np.testing.assert_array_equal(quat, self.null)
        np.testing.assert_array_equal(quat.shape, self.null.shape)

#%% aerospace.quat_norm
class Test_aerospace_quat_norm(unittest.TestCase):
    r"""
    Tests the aerospace.quat_norm function with the following cases:
        Single quat (x3 different quats)
        Quat array
        Null (x2 different null sizes)
    """
    def setUp(self):
        self.q1_inp = space.qrot(1, np.pi/2)
        self.q1_out = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        self.q2_inp = space.qrot(2, np.pi/3)
        self.q2_out = np.array([0, 0.5, 0, np.sqrt(3)/2])
        self.q3_inp = np.array([0.1, 0, 0, 1])
        self.q3_out = np.array([0.09950372, 0, 0, 0.99503719])
        self.q4_inp = np.column_stack((self.q1_inp, self.q2_inp, self.q3_inp))
        self.q4_out = np.column_stack((self.q1_out, self.q2_out, self.q3_out))
        self.null   = np.array([])
        self.null_quat = np.ones((space.QUAT_SIZE, 0))

    def test_nominal1(self):
        quat_norm = space.quat_norm(self.q1_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q1_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q1_out.shape)

    def test_nominal2(self):
        quat_norm = space.quat_norm(self.q2_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q2_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q2_out.shape)

    def test_nominal3(self):
        quat_norm = space.quat_norm(self.q3_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q3_out)
        self.assertEqual(quat_norm.ndim, 1)
        np.testing.assert_array_equal(quat_norm.shape, self.q3_out.shape)

    def test_array(self):
        quat_norm = space.quat_norm(self.q4_inp)
        np.testing.assert_array_almost_equal(quat_norm, self.q4_out)
        self.assertEqual(quat_norm.ndim, 2)
        np.testing.assert_array_equal(quat_norm.shape, self.q4_out.shape)

    def test_null_input1(self):
        quat_norm = space.quat_norm(self.null_quat)
        np.testing.assert_array_equal(quat_norm, self.null_quat)
        np.testing.assert_array_equal(quat_norm.shape, self.null_quat.shape)

    def test_null_input2(self):
        quat_norm = space.quat_norm(self.null)
        np.testing.assert_array_equal(quat_norm, self.null)
        np.testing.assert_array_equal(quat_norm.shape, self.null.shape)

#%% aerospace.quat_prop
class Test_aerospace_quat_prop(unittest.TestCase):
    r"""
    Tests the aerospace.quat_prop function with the following cases:
        Nominal case
        No renormalization case (Raises norm AttributeError)
    """
    def setUp(self):
        self.quat      = np.array([0, 0, 0, 1])
        self.delta_ang = np.array([0.01, 0.02, 0.03])
        self.quat_new  = np.array([0.00499913, 0.00999825, 0.01499738, 0.99982505])

    def test_nominal(self):
        quat = space.quat_prop(self.quat, self.delta_ang)
        np.testing.assert_array_almost_equal(quat, self.quat_new)

    def test_negative_scalar(self):
        quat = space.quat_prop(np.array([1, 0, 0, 0]), self.delta_ang)
        self.assertGreater(quat[3], 0)
        quat = space.quat_prop(np.array([1, 0, 0, 0]), -self.delta_ang)
        self.assertGreater(quat[3], 0)

    def test_no_renorm(self):
        with self.assertRaises(AssertionError):
            space.quat_prop(self.quat, self.delta_ang, renorm=False)

#%% aerospace.quat_times_vector
class Test_aerospace_quat_times_vector(unittest.TestCase):
    r"""
    Tests the aerospace.quat_times_vector function with the following cases:
        Nominal
        Array inputs
    """
    def setUp(self):
        # TODO: confirm that this is enough to test the correctness of the function
        self.quat = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).T
        self.vec  = np.array([[1, 0, 0], [2, 0, 0]]).T
        self.out  = np.array([[-1, 2], [0, 0], [0, 0]])

    def test_nominal(self):
        for i in range(2):
            vec = space.quat_times_vector(self.quat[:, i], self.vec[:, i])
            np.testing.assert_array_almost_equal(vec, self.out[:, i])

    def test_array(self):
        vec = space.quat_times_vector(self.quat, self.vec)
        np.testing.assert_array_almost_equal(vec, self.out)

#%% aerospace.quat_to_dcm
class Test_aerospace_quat_to_dcm(unittest.TestCase):
    r"""
    Tests the aerospace.quat_to_dcm function with the following cases:
        Nominal case
    """
    def setUp(self):
        self.quat = np.array([0.5, -0.5, 0.5, 0.5])
        self.dcm  = np.array([\
            [ 0.,  0.,  1.],
            [-1.,  0.,  0.],
            [ 0., -1.,  0.]])

    def test_nominal(self):
        dcm = space.quat_to_dcm(self.quat)
        np.testing.assert_array_almost_equal(dcm, self.dcm)

#%% aerospace.quat_to_euler
class Test_aerospace_quat_to_euler(unittest.TestCase):
    r"""
    Tests the aerospace.quat_to_euler function with the following cases:
        Nominal
        Zero quat
        All valid sequences
        All invalid sequences
        Bad length sequence
    """
    def setUp(self):
        self.quat  = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2]]).T
        self.seq   = [3, 1, 2]
        self.euler = np.array([\
            [ 0.   , -np.pi, 0.       ],
            [ 0.   ,  0.   , -np.pi/2 ],
            [ np.pi, -0.   , 0.       ]])
        self.zero_quat = np.array([0, 0, 0, 1])
        self.all_sequences = {\
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2), (1, 3, 3), \
            (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 3, 1), (2, 3, 2), (2, 3, 3), \
            (3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 3, 1), (3, 3, 2), (3, 3, 3)}
        self.valid_sequences = {(1,2,3), (2,3,1), (3,1,2), (1,3,2), (2,1,3), (3,2,1)}
        self.bad_sequences = self.all_sequences - self.valid_sequences

    def test_nominal(self):
        euler = space.quat_to_euler(self.quat, self.seq)
        np.testing.assert_array_almost_equal(euler, self.euler)

    def test_zero_quat(self):
        euler = space.quat_to_euler(self.zero_quat)
        np.testing.assert_array_equal(euler, np.zeros(3))

    def test_all_valid(self):
        # TODO: this doesn't confirm that all of these give the correct answer, but just don't crash
        for this_seq in self.valid_sequences:
            euler = space.quat_to_euler(self.zero_quat, np.array(this_seq))
            np.testing.assert_array_equal(euler, np.zeros(3))

    def test_all_invalid(self):
        for this_seq in self.bad_sequences:
            with self.assertRaises(ValueError):
                space.quat_to_euler(self.zero_quat, np.array(this_seq))

    def test_bad_sequence(self):
        with self.assertRaises(AssertionError):
            space.quat_to_euler(self.zero_quat, np.array([1, 2]))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
