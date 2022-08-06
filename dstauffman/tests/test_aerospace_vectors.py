r"""
Test file for the `vectors` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import unittest

from dstauffman import HAVE_NUMPY
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np

#%% aerospace.rot
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_rot(unittest.TestCase):
    r"""
    Tests the aerospace.rot function with the following cases:
        Reference 1, single axis
        Reference 2, single axis
    """

    def setUp(self) -> None:
        self.angle = np.pi / 6  # 30 deg
        self.angle2 = 3 * np.pi / 4  # 135 deg
        r2o2 = np.sqrt(2) / 2
        r3o2 = np.sqrt(3) / 2

        # reference 1
        self.r1x = np.array([[1.0, 0.0, 0.0], [0.0, r3o2, 0.5], [0.0, -0.5, r3o2]])
        self.r1y = np.array([[r3o2, 0.0, -0.5], [0.0, 1.0, 0.0], [0.5, 0.0, r3o2]])
        self.r1z = np.array([[r3o2, 0.5, 0.0], [-0.5, r3o2, 0.0], [0.0, 0.0, 1.0]])

        # reference 2
        self.r2x = np.array([[1.0, 0.0, 0.0], [0.0, -r2o2, r2o2], [0.0, -r2o2, -r2o2]])
        self.r2y = np.array([[-r2o2, 0.0, -r2o2], [0.0, 1.0, 0], [r2o2, 0.0, -r2o2]])
        self.r2z = np.array([[-r2o2, r2o2, 0.0], [-r2o2, -r2o2, 0.0], [0.0, 0.0, 1.0]])

        self.tolerance = 1e-12

    def test_ref1(self) -> None:
        out1 = space.rot(1, self.angle)
        out2 = space.rot(2, self.angle)
        out3 = space.rot(3, self.angle)
        np.testing.assert_allclose(out1, self.r1x, atol=self.tolerance)
        np.testing.assert_allclose(out2, self.r1y, atol=self.tolerance)
        np.testing.assert_allclose(out3, self.r1z, atol=self.tolerance)

    def test_ref2(self) -> None:
        out1 = space.rot(1, self.angle2)
        out2 = space.rot(2, self.angle2)
        out3 = space.rot(3, self.angle2)
        np.testing.assert_allclose(out1, self.r2x, atol=self.tolerance)
        np.testing.assert_allclose(out2, self.r2y, atol=self.tolerance)
        np.testing.assert_allclose(out3, self.r2z, atol=self.tolerance)

    def test_bad_axis(self) -> None:
        with self.assertRaises(ValueError) as context:
            space.rot(4, self.angle)
        self.assertEqual(str(context.exception), "Unexpected value for axis.")
        with self.assertRaises(ValueError):
            space.rot(np.pi / 2, 2)


#%% aerospace.drot
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_drot(unittest.TestCase):
    r"""
    Tests the aerospace.drot function with the following cases:
        TBD
    """
    pass  # TODO: write this


#%% aerospace.vec_cross
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_vec_cross(unittest.TestCase):
    r"""
    Tests the aerospace.vec_cross function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.a = np.array([1, 2, 3])
        self.b = np.array([-2, -3, -4])
        self.c = np.cross(self.a, self.b)

    def test_nominal(self) -> None:
        mat = space.vec_cross(self.a)
        c = mat @ self.b
        np.testing.assert_array_equal(c, self.c)

    def test_floats(self) -> None:
        mat = space.vec_cross(self.a.astype(float))
        c = mat @ self.b.astype(float)
        np.testing.assert_array_equal(c, self.c)


#%% aerospace.vec_angle
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_vec_angle(unittest.TestCase):
    r"""
    Tests the aerospace.vec_angle function with the following cases:
        Small angle
        Small angle without cross
        Nominal
        Vectorized (x3)
    """

    def setUp(self) -> None:
        self.vec1 = np.array([1.0, 0.0, 0.0])
        self.vec2 = space.rot(2, 1e-5) @ self.vec1
        self.vec3 = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0])
        self.exp1 = 1e-5
        self.exp2 = np.pi / 4
        self.exp3 = 2 * np.pi / 3

    def test_small(self) -> None:
        angle = space.vec_angle(self.vec1, self.vec2)
        self.assertAlmostEqual(angle, self.exp1, 14)

    def test_bad_small(self) -> None:
        angle = space.vec_angle(self.vec1, self.vec2, use_cross=False)
        self.assertTrue(np.abs(angle - self.exp1) > 1e-14)

    def test_nominal(self) -> None:
        angle = space.vec_angle(self.vec1, self.vec3)
        self.assertAlmostEqual(angle, self.exp2, 14)
        angle = space.vec_angle(self.vec1, self.vec3, use_cross=False)
        self.assertAlmostEqual(angle, self.exp2, 14)

    def test_vectorized(self) -> None:
        matrix1 = np.vstack((self.vec1, self.vec2, self.vec3, self.vec1)).T
        matrix2 = np.vstack((self.vec2, self.vec1, self.vec1, self.vec1)).T
        exp1 = np.array([0.0, self.exp1, self.exp2, 0.0])
        exp2 = np.array([self.exp1, self.exp1, self.exp2, 0.0])
        angle = space.vec_angle(matrix1, self.vec1)
        np.testing.assert_almost_equal(angle, exp1, 14)
        angle = space.vec_angle(self.vec1, matrix1)
        np.testing.assert_almost_equal(angle, exp1, 14)
        angle = space.vec_angle(matrix1, matrix2)
        np.testing.assert_almost_equal(angle, exp2, 14)
        angle = space.vec_angle(matrix1, matrix2, use_cross=False)
        np.testing.assert_almost_equal(angle, exp2, 12)

    def test_list(self) -> None:
        list1 = [self.vec1, self.vec2, self.vec3, self.vec1]
        list2 = [self.vec2, self.vec1, self.vec1, self.vec1]
        exp1 = np.array([0.0, self.exp1, self.exp2, 0.0])
        exp2 = np.array([self.exp1, self.exp1, self.exp2, 0.0])
        angle = space.vec_angle(list1, self.vec1)
        np.testing.assert_almost_equal(angle, exp1, 14)
        angle = space.vec_angle(self.vec1, list1)
        np.testing.assert_almost_equal(angle, exp1, 14)
        angle = space.vec_angle(list1, list2)
        np.testing.assert_almost_equal(angle, exp2, 14)
        angle = space.vec_angle(list1, list2, use_cross=False)
        np.testing.assert_almost_equal(angle, exp2, 12)

    def test_tuple(self) -> None:
        tuple1 = (self.vec1, self.vec2, self.vec3, self.vec1)
        tuple2 = (self.vec2, self.vec1, self.vec1, self.vec1)
        exp1 = np.array((0.0, self.exp1, self.exp2, 0.0))
        exp2 = np.array((self.exp1, self.exp1, self.exp2, 0.0))
        angle = space.vec_angle(tuple1, self.vec1)
        np.testing.assert_almost_equal(angle, exp1, 14)
        angle = space.vec_angle(self.vec1, tuple1)
        np.testing.assert_almost_equal(angle, exp1, 14)
        angle = space.vec_angle(tuple1, tuple2)
        np.testing.assert_almost_equal(angle, exp2, 14)
        angle = space.vec_angle(tuple1, tuple2, use_cross=False)
        np.testing.assert_almost_equal(angle, exp2, 12)

    def test_not_normalized(self) -> None:
        angle = space.vec_angle(np.array([0, 2.0, 0]), np.array([0.0, -5.0, 5.0]), normalized=False, use_cross=True)
        self.assertAlmostEqual(angle, 3 * np.pi / 4, 14)
        angle = space.vec_angle(np.array([0, 2.0, 0]), np.array([0.0, -5.0, 5.0]), normalized=False, use_cross=False)
        self.assertAlmostEqual(angle, 3 * np.pi / 4, 14)

    def test_4d_vector(self) -> None:
        angle = space.vec_angle(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]), use_cross=False)
        self.assertAlmostEqual(angle, np.pi / 2)
        with self.assertRaises(ValueError):
            space.vec_angle(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]), use_cross=True)

    def test_2d_vector(self) -> None:
        angle = space.vec_angle(np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2]), use_cross=False)
        self.assertAlmostEqual(angle, self.exp3, 14)
        angle = space.vec_angle(np.array([1.0, 0.0]), np.array([-0.5, -np.sqrt(3) / 2]), use_cross=False)
        self.assertAlmostEqual(angle, self.exp3, 14)
        angle = space.vec_angle(np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2]), use_cross=True)
        self.assertAlmostEqual(angle, self.exp3, 14)
        angle = space.vec_angle(np.array([1.0, 0.0]), np.array([-0.5, -np.sqrt(3) / 2]), use_cross=True)
        self.assertAlmostEqual(angle, self.exp3, 14)


#%% aerospace.cart2sph
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_cart2sph(unittest.TestCase):
    r"""
    Tests the aerospace.cart2sph function with the following cases:
        Single inputs
        Vector inputs
        Full loop
    """

    def setUp(self) -> None:
        self.xyz1 = (1.0, 0.0, 0.0)
        self.aer1 = (0.0, 0.0, 1.0)
        self.xyz2 = (0.0, -1.0, 0.0)
        self.aer2 = (-np.pi / 2, 0, 1)

    def test_single(self) -> None:
        (az, el, rad) = space.cart2sph(*self.xyz1)
        self.assertAlmostEqual(az, self.aer1[0], 14)
        self.assertAlmostEqual(el, self.aer1[1], 14)
        self.assertAlmostEqual(rad, self.aer1[2], 14)
        (az, el, rad) = space.cart2sph(*self.xyz2)
        self.assertAlmostEqual(az, self.aer2[0], 14)
        self.assertAlmostEqual(el, self.aer2[1], 14)
        self.assertAlmostEqual(rad, self.aer2[2], 14)

    def test_vectors(self) -> None:
        x = np.array([self.xyz1[0], self.xyz2[0]])
        y = np.array([self.xyz1[1], self.xyz2[1]])
        z = np.array([self.xyz1[2], self.xyz2[2]])
        exp_az = np.array([self.aer1[0], self.aer2[0]])
        exp_el = np.array([self.aer1[1], self.aer2[1]])
        exp_rad = np.array([self.aer1[2], self.aer2[2]])
        (az, el, rad) = space.cart2sph(x, y, z)
        np.testing.assert_array_almost_equal(az, exp_az, 14)
        np.testing.assert_array_almost_equal(el, exp_el, 14)
        np.testing.assert_array_almost_equal(rad, exp_rad, 14)

    def test_loop(self) -> None:
        num = 10
        x = 3 * np.random.rand(num)
        y = -4 * np.random.rand(num)
        z = 10 * np.random.rand(num)
        (az, el, rad) = space.cart2sph(x, y, z)
        (x2, y2, z2) = space.sph2cart(az, el, rad)
        np.testing.assert_array_almost_equal(x, x2, 14)
        np.testing.assert_array_almost_equal(y, y2, 14)
        np.testing.assert_array_almost_equal(z, z2, 14)


#%% aerospace.sph2cart
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_sph2cart(unittest.TestCase):
    r"""
    Tests the aerospace.sph2cart function with the following cases:
        Single inputs
        Vector inputs
        Full loop
    """

    def setUp(self) -> None:
        self.xyz1 = (1.0, 0.0, 0.0)
        self.aer1 = (0.0, 0.0, 1.0)
        self.xyz2 = (0.0, -1.0, 0.0)
        self.aer2 = (-np.pi / 2, 0, 1)

    def test_single(self) -> None:
        (x, y, z) = space.sph2cart(*self.aer1)
        self.assertAlmostEqual(x, self.xyz1[0], 14)
        self.assertAlmostEqual(y, self.xyz1[1], 14)
        self.assertAlmostEqual(z, self.xyz1[2], 14)
        (x, y, z) = space.sph2cart(*self.aer2)
        self.assertAlmostEqual(x, self.xyz2[0], 14)
        self.assertAlmostEqual(y, self.xyz2[1], 14)
        self.assertAlmostEqual(z, self.xyz2[2], 14)

    def test_vectors(self) -> None:
        az = np.array([self.aer1[0], self.aer2[0]])
        el = np.array([self.aer1[1], self.aer2[1]])
        rad = np.array([self.aer1[2], self.aer2[2]])
        exp_x = np.array([self.xyz1[0], self.xyz2[0]])
        exp_y = np.array([self.xyz1[1], self.xyz2[1]])
        exp_z = np.array([self.xyz1[2], self.xyz2[2]])
        (x, y, z) = space.sph2cart(az, el, rad)
        np.testing.assert_array_almost_equal(x, exp_x, 14)
        np.testing.assert_array_almost_equal(y, exp_y, 14)
        np.testing.assert_array_almost_equal(z, exp_z, 14)

    def test_loop(self) -> None:
        num = 10
        az = 2 * np.pi * np.random.rand(num) - np.pi
        el = np.pi / 2 * np.random.rand(num)
        rad = 10 * np.random.rand(num)
        (x, y, z) = space.sph2cart(az, el, rad)
        (az2, el2, rad2) = space.cart2sph(x, y, z)
        np.testing.assert_array_almost_equal(az, az2, 14)
        np.testing.assert_array_almost_equal(el, el2, 14)
        np.testing.assert_array_almost_equal(rad, rad2, 14)


#%% aerospace.rv2dcm
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_rv2dcm(unittest.TestCase):
    r"""
    Tests the aerospace.rv2dcm function with the following cases:
        Each axis
        Non-unity vector
    """

    def test_simple_rots(self) -> None:
        for axis in range(1, 4):
            for angle in [0.0, 1.0, -1.0, np.pi / 6, np.pi / 4, np.pi, 2 * np.pi, -3 * np.pi / 2]:
                vec = np.zeros(3)
                vec[axis - 1] = angle
                dcm = space.rv2dcm(vec)
                exp = space.rot(axis, angle)
                np.testing.assert_array_almost_equal(dcm, exp, decimal=14)

    def test_complex(self) -> None:
        vec = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
        dcm = space.rv2dcm(vec)
        mag = np.sqrt(np.sum(vec**2))
        quat = space.quat_norm(np.hstack((vec / mag * np.sin(mag / 2), np.cos(mag / 2))))
        exp = space.quat_to_dcm(quat)
        np.testing.assert_array_almost_equal(dcm, exp, decimal=14)


#%% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
