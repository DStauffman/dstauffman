# -*- coding: utf-8 -*-
r"""
Test file for the `quat` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import numpy as np
import unittest
import dstauffman as dcs

#%% qrot
class Test_qrot(unittest.TestCase):
    r"""
    Tests the qrot function with the following cases:
        Single input case
        Single axis, multiple angles
        Multiple axes, single angle (Not working)
        Multiple axes, multiple angles (Not working)
    """
    def setUp(self):
        self.axis   = np.array([1, 2, 3])
        self.angle  = np.pi/2
        self.angle2 = np.pi/3
        r2o2        = np.sqrt(2)/2
        r3o2        = np.sqrt(3)/2
        self.quat   = np.array([[r2o2, 0, 0, r2o2], [0, r2o2, 0, r2o2], [0, 0, r2o2, r2o2]])
        self.quat2  = np.array([[ 0.5, 0, 0, r3o2], [0,  0.5, 0, r3o2], [0, 0,  0.5, r3o2]])

    def test_single_inputs(self):
        for i in range(len(self.axis)):
            quat = dcs.qrot(self.axis[i], self.angle)
            np.testing.assert_almost_equal(quat, self.quat[i, :])

    def test_single_axis(self):
        for i in range(len(self.axis)):
            quat = dcs.qrot(self.axis[i], np.array([self.angle, self.angle2]))
            np.testing.assert_almost_equal(quat, dcs.concat_vectors(self.quat[i, :], self.quat2[i, :]))

    #def test_single_angle(self):
    #    quat = dcs.qrot(self.axis, self.angle)
    #    np.testing.assert_almost_equal(quat, self.quat)

    #def all_vector_inputs(self):
    #    quat = dcs.qrot(self.axis, np.array([self.angle, self.angle2]))
    #    np.testing.assert_almost_equal(quat, dcs.concat_vectors(self.quat, self.quat2))

#%% quat_angle_diff
class test_quat_angle_diff(unittest.TestCase):
    r"""
    Tests the quat_angle_diff function with the following cases:
        TBD
    """
    def setUp(self):
        self.quat1 = np.array([0.5, 0.5, 0.5, 0.5])
        self.dq1   = dcs.qrot(1, 0.001)
        self.dq2   = dcs.qrot(2, 0.05)
        self.quat2 = dcs.concat_vectors(dcs.quat_mult(self.dq1, self.quat1), dcs.quat_mult(self.dq2, self.quat1))
        self.theta = np.array([0.001, 0.05])
        self.comp  = np.array([[0.001, 0], [0, 0.05], [0, 0]])

    def test_nominal(self):
        (theta, comp) = dcs.quat_angle_diff(self.quat1, self.quat2)
        np.testing.assert_almost_equal(theta, self.theta)
        np.testing.assert_almost_equal(comp, self.comp)

#%% quat_from_euler
class test_quat_from_euler(unittest.TestCase):
    r"""
    Tests the quat_from_euler function with the following cases:
        TBD
    """
    def setUp(self):
        self.a      = np.array([0.01, 0.02, 0.03])
        self.b      = np.array([0.04, 0.05, 0.06])
        self.angles = dcs.concat_vectors(self.a, self.b)
        self.seq    = np.array([3, 2, 1])
        self.quat   = np.array([\
            [0.01504849, 0.03047982],
            [0.00992359, 0.02438147],
            [0.00514916, 0.02073308],
            [0.99982426, 0.99902285]])

    def test_nominal(self):
        quat = dcs.quat_from_euler(self.angles, self.seq)
        np.testing.assert_almost_equal(quat, self.quat)

#%% quat_interp
class test_quat_interp(unittest.TestCase):
    r"""
    Tests the quat_interp function with the following cases:
        TBD
    """
    def setUp(self):
        self.time  = np.array([1, 3, 5])
        self.quat  = np.array([[0, 0, 0, 1], [0, 0, 0.1961, 0.9806], [0.5, -0.5, -0.5, 0.5]]).T
        self.ti    = np.array([1, 2, 4.5, 5])
        self.qout  = np.array([\
            [ 0., 0.        ,  0.41748298,  0.5 ],
            [ 0., 0.        , -0.41748298, -0.5 ],
            [ 0., 0.09852786, -0.35612893, -0.5 ],
            [ 1., 0.99513429,  0.72428455,  0.5 ]])

    def test_nominal(self):
        qout = dcs.quat_interp(self.time, self.quat, self.ti)
        np.testing.assert_almost_equal(qout, self.qout)

#%% quat_inv
class test_quat_inv(unittest.TestCase):
    r"""
    Tests the quat_inv function with the following cases:
        TBD
    """
    def setUp(self):
        self.q1 = dcs.qrot(1, np.pi/2)
        self.q2 = np.array([-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])

    def test_nominal(self):
        q2 = dcs.quat_inv(self.q1)
        np.testing.assert_almost_equal(q2, self.q2)


#%% quat_mult
class test_quat_mult(unittest.TestCase):
    r"""
    Tests the quat_mult function with the following cases:
        TBD
    """
    def setUp(self):
        self.a = dcs.qrot(1, np.pi/2)
        self.b = dcs.qrot(2, np.pi)
        self.c = np.array([ 0, np.sqrt(2)/2, -np.sqrt(2)/2, 0])

    def test_nominal(self):
        c = dcs.quat_mult(self.a, self.b)
        np.testing.assert_almost_equal(c, self.c)

#%% quat_norm
class test_quat_norm(unittest.TestCase):
    r"""
    Tests the quat_norm function with the following cases:
        TBD
    """
    def setUp(self):
        self.x = np.array([0.1, 0, 0, 1])
        self.y = np.array([0.09950372, 0, 0, 0.99503719])

    def test_nominal(self):
        y = dcs.quat_norm(self.x)
        np.testing.assert_almost_equal(y, self.y)

#%% quat_prop
class test_quat_prop(unittest.TestCase):
    r"""
    Tests the quat_prop function with the following cases:
        TBD
    """
    def setUp(self):
        self.quat      = np.array([0, 0, 0, 1])
        self.delta_ang = np.array([0.01, 0.02, 0.03])
        self.quat_new  = np.array([0.00499913, 0.00999825, 0.01499738, 0.99982505])

    def test_nominal(self):
        quat_new = dcs.quat_prop(self.quat, self.delta_ang)
        np.testing.assert_almost_equal(quat_new, self.quat_new)

#%% quat_times_vector
class test_quat_times_vector(unittest.TestCase):
    r"""
    Tests the quat_times_vector function with the following cases:
        TBD
    """
    def setUp(self):
        self.quat = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).T
        self.v = np.array([[1, 0, 0], [2, 0, 0]]).T
        self.vec = np.array([[-1, 2], [0, 0], [0, 0]])

    def test_nominal(self):
        vec = dcs.quat_times_vector(self.quat, self.v)
        np.testing.assert_almost_equal(vec, self.vec)

#%% quat_to_dcm
class test_quat_to_dcm(unittest.TestCase):
    r"""
    Tests the quat_to_dcm function with the following cases:
        TBD
    """
    def setUp(self):
        self.quat = np.array([0.5, -0.5, 0.5, 0.5])
        self.dcm  = np.array([\
            [ 0.,  0.,  1.],
            [-1.,  0.,  0.],
            [ 0., -1.,  0.]])

    def test_nominal(self):
        dcm = dcs.quat_to_dcm(self.quat)
        np.testing.assert_almost_equal(dcm, self.dcm)

#%% quat_to_euler
class test_quat_to_euler(unittest.TestCase):
    r"""
    Tests the quat_to_euler function with the following cases:
        TBD
    """
    def setUp(self):
        self.quat  = np.array([[0, 1, 0, 0], [0, 0, 1, 0]]).T
        self.seq   = [3, 1, 2]
        self.euler = np.array([\
            [-0.        , -3.14159265],
            [ 0.        ,  0.        ],
            [ 3.14159265, -0.        ]])

    def test_nominal(self):
        euler = dcs.quat_to_euler(self.quat, self.seq)
        np.testing.assert_almost_equal(euler, self.euler)

#%% concat_vectors
class test_concat_vectors(unittest.TestCase):
    r"""
    Tests the concat_vectors function with the following cases:
        TBD
    """
    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([4, 5, 6])
        self.c = np.array([[1, 4], [2, 5],[3, 6]])

    def test_nominal(self):
        c = dcs.concat_vectors(self.a, self.b)
        np.testing.assert_almost_equal(c, self.c)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
