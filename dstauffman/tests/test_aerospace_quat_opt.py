r"""
Test file for the `quat_opt` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in February 2021.
"""

#%% Imports
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np

#%% aerospace.qrot_single
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_qrot_single(unittest.TestCase):
    r"""
    Tests the aerospace.qrot_single function with the following cases:
        Single input case
    """
    def setUp(self) -> None:
        self.axis   = np.array([1, 2, 3])
        self.angle  = np.pi/2
        self.angle2 = np.pi/3
        r2o2        = np.sqrt(2)/2
        r3o2        = np.sqrt(3)/2
        self.quat   = np.array([[r2o2, 0, 0, r2o2], [0, r2o2, 0, r2o2], [0, 0, r2o2, r2o2]])
        self.quat2  = np.array([[ 0.5, 0, 0, r3o2], [0,  0.5, 0, r3o2], [0, 0,  0.5, r3o2]])

    def test_single_inputs(self) -> None:
        for i in range(len(self.axis)):
            quat = space.qrot_single(self.axis[i], self.angle)
            self.assertEqual(quat.ndim, 1)
            np.testing.assert_array_almost_equal(quat, self.quat[i, :])
            quat = space.qrot_single(self.axis[i], self.angle2)
            self.assertEqual(quat.ndim, 1)
            np.testing.assert_array_almost_equal(quat, self.quat2[i, :])

#%% aerospace.quat_interp_single
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_quat_interp_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_interp_single function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.time = np.array([1., 3., 5.])
        self.quat = np.vstack((space.qrot_single(1, 0), space.qrot_single(1, np.pi/2), \
            space.qrot_single(1, np.pi))).T
        self.ti   = np.array([1., 2., 4.5, 5.])
        self.qout = np.column_stack((space.qrot_single(1, 0), space.qrot_single(1, np.pi/4), \
            space.qrot_single(1, 3.5/4*np.pi), space.qrot_single(1, np.pi)))

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
            np.testing.assert_array_almost_equal(qout, self.qout[:, 3])
        else:
            np.testing.assert_array_almost_equal(qout, -self.qout[:, 3])

    def test_extra(self) -> None:
        with self.assertRaises(ValueError):
            space.quat_interp_single(self.time, self.quat, 10.)

#%% aerospace.quat_inv_single
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_quat_inv_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_inv_single function with the following cases:
        Single quat (x2 different quats)
    """
    def setUp(self) -> None:
        self.q1_inp = space.qrot_single(1, np.pi/2)
        self.q1_out = np.array([-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        self.q2_inp = space.qrot_single(2, np.pi/3)
        self.q2_out = np.array([0, -0.5, 0, np.sqrt(3)/2])

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

#%% aerospace.quat_mult_single
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_quat_mult_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_mult_single function with the following cases:
        Single quat (x2 different quats)
        Reverse order
        Quat array times scalar (x2 orders)
    """
    def setUp(self) -> None:
        self.q1 = space.qrot_single(1, np.pi/2)
        self.q2 = space.qrot_single(2, -np.pi)
        self.q3 = space.qrot_single(3, np.pi/3)
        self.q4 = np.array([ 0, -np.sqrt(2)/2, np.sqrt(2)/2, 0]) # q1*q2
        self.q5 = np.array([0.5, -np.sqrt(3)/2, 0, 0]) # q2*q3
        self.q6 = np.array([0.5, 0.5, 0.5, 0.5]) # q6 * q6 = q6**-1, and triggers negative scalar component

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

#%% aerospace.quat_norm_single
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_quat_norm_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_norm_single function with the following cases:
        Single quat (x3 different quats)
    """
    def setUp(self) -> None:
        self.q1_inp = space.qrot_single(1, np.pi/2)
        self.q1_out = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        self.q2_inp = space.qrot_single(2, np.pi/3)
        self.q2_out = np.array([0, 0.5, 0, np.sqrt(3)/2])
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

#%% aerospace.quat_times_vector_single
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_quat_times_vector_single(unittest.TestCase):
    r"""
    Tests the aerospace.quat_times_vector_single function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        # TODO: confirm that this is enough to test the correctness of the function
        self.quat = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).T
        self.vec  = np.array([[1, 0, 0], [2, 0, 0]]).T
        self.out  = np.array([[-1, 2], [0, 0], [0, 0]])

    def test_integers(self) -> None:
        for i in range(2):
            vec = space.quat_times_vector_single(self.quat[:, i], self.vec[:, i])
            np.testing.assert_array_almost_equal(vec, self.out[:, i])

    def test_nominal(self) -> None:
        for i in range(2):
            vec = space.quat_times_vector_single(self.quat[:, i].astype(float), self.vec[:, i].astype(float))
            np.testing.assert_array_almost_equal(vec, self.out[:, i].astype(float))

#%% aerospace.quat_to_dcm
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_quat_to_dcm(unittest.TestCase):
    r"""
    Tests the aerospace.quat_to_dcm function with the following cases:
        Nominal case
    """
    def setUp(self) -> None:
        self.quat = np.array([0.5, -0.5, 0.5, 0.5])
        self.dcm  = np.array([\
            [ 0.,  0.,  1.],
            [-1.,  0.,  0.],
            [ 0., -1.,  0.]])

    def test_nominal(self) -> None:
        dcm = space.quat_to_dcm(self.quat)
        np.testing.assert_array_almost_equal(dcm, self.dcm)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)