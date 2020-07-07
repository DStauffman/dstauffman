r"""
Test file for the `vectors` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import unittest

import numpy as np

import dstauffman.spacecraft as space

#%% rot
class Test_rot(unittest.TestCase):
    r"""
    Tests the rot function with the following cases:
        Reference 1, single axis
        Reference 2, single axis
    """
    def setUp(self):
        self.angle  = np.pi/6 # 30 deg
        self.angle2 = 3*np.pi/4 # 135 deg
        r2o2        = np.sqrt(2)/2
        r3o2        = np.sqrt(3)/2

        # reference 1
        self.r1x = np.array([[1., 0., 0.], [0., r3o2, 0.5], [0., -0.5, r3o2]], dtype=float)
        self.r1y = np.array([[r3o2, 0., -0.5], [0., 1., 0.], [0.5, 0., r3o2]], dtype=float)
        self.r1z = np.array([[r3o2, 0.5, 0.], [-0.5, r3o2, 0.], [0., 0., 1.]], dtype=float)

        # reference 2
        self.r2x = np.array([[1., 0., 0.], [0., -r2o2, r2o2], [0., -r2o2, -r2o2]], dtype=float)
        self.r2y = np.array([[-r2o2, 0., -r2o2], [0., 1., 0], [r2o2, 0., -r2o2]], dtype=float)
        self.r2z = np.array([[-r2o2, r2o2, 0.], [-r2o2, -r2o2, 0.], [0., 0., 1.]], dtype=float)

        self.tolerance = 1e-12

    def test_ref1(self):
        out1 = space.rot(1, self.angle)
        out2 = space.rot(2, self.angle)
        out3 = space.rot(3, self.angle)
        np.testing.assert_allclose(out1, self.r1x, atol=self.tolerance)
        np.testing.assert_allclose(out2, self.r1y, atol=self.tolerance)
        np.testing.assert_allclose(out3, self.r1z, atol=self.tolerance)

    def test_ref2(self):
        out1 = space.rot(1, self.angle2)
        out2 = space.rot(2, self.angle2)
        out3 = space.rot(3, self.angle2)
        np.testing.assert_allclose(out1, self.r2x, atol=self.tolerance)
        np.testing.assert_allclose(out2, self.r2y, atol=self.tolerance)
        np.testing.assert_allclose(out3, self.r2z, atol=self.tolerance)

    def test_bad_axis(self):
        with self.assertRaises(ValueError) as context:
            space.rot(4, self.angle)
            self.assertEqual(str(context.exception), 'Unexpected value for axis of: "4".')
        with self.assertRaises(ValueError):
            space.rot(np.pi/2, 2)

#%% vec_cross
class Test_vec_cross(unittest.TestCase):
    r"""
    Tests the vec_cross function with the following cases:
        Nominal
    """
    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([-2, -3, -4])
        self.c = np.cross(self.a, self.b)

    def test_nominal(self):
        mat = space.vec_cross(self.a)
        c = mat @ self.b
        np.testing.assert_array_equal(c, self.c)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
