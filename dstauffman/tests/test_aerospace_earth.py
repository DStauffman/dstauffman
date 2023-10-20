r"""
Test file for the `earth` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

# %% Imports
import unittest

from dstauffman import DEG2RAD, HAVE_NUMPY, M2FT
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np


# %% aerospace.geod2ecf
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_geod2ecf(unittest.TestCase):
    r"""
    Tests the aerospace.geod2ecf function with the following cases:
        Nominal
        Output vectors
        Input vectors
        ALl vectors
        Scalars
        English Units
        Round-trip
        Bad Units
        Bad Output
    """

    def setUp(self) -> None:
        self.a = 6378137.0  # WGS84 elliptical Earth semimajor axis
        f = 1 / 298.257223563
        self.b = self.a * (1 - f)
        self.lla = np.array(
            [
                [0, 0, 0],  # Test all zeros
                [0, np.pi / 4, 0],  # Test 45 degrees at equator
                [np.pi / 2, 0, 0],  # Test North Pole
                [-np.pi / 2, 0, 100],  # Test South Pole
                [np.pi / 2, np.pi / 4, -100],
            ]
        ).T  # Longitude shouldn't matter at North Pole
        self.xyz = np.array(
            [
                [self.a, 0, 0],
                [self.a / np.sqrt(2), self.a / np.sqrt(2), 0],
                [0, 0, self.b],
                [0, 0, -self.b - 100],
                [0, 0, self.b - 100],
            ]
        ).T
        self.decimal = 8

    def test_3d(self) -> None:
        xyz = space.geod2ecf(self.lla)
        np.testing.assert_array_almost_equal(xyz, self.xyz, decimal=self.decimal)

    def test_output_vectors(self) -> None:
        (x, y, z) = space.geod2ecf(self.lla, output="split")
        np.testing.assert_array_almost_equal(x, self.xyz[0, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(y, self.xyz[1, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(z, self.xyz[2, :], decimal=self.decimal)

    def test_input_vectors(self) -> None:
        xyz = space.geod2ecf(self.lla[0, :], self.lla[1, :], self.lla[2, :])
        np.testing.assert_array_almost_equal(xyz, self.xyz, decimal=self.decimal)

    def test_all_vectors(self) -> None:
        (x, y, z) = space.geod2ecf(self.lla[0, :], self.lla[1, :], self.lla[2, :], output="split")
        np.testing.assert_array_almost_equal(x, self.xyz[0, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(y, self.xyz[1, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(z, self.xyz[2, :], decimal=self.decimal)

    def test_scalars(self) -> None:
        for ix in range(self.lla.shape[1]):
            xyz = space.geod2ecf(self.lla[:, ix])
            np.testing.assert_array_almost_equal(xyz, self.xyz[:, ix], decimal=self.decimal)
            xyz = space.geod2ecf(self.lla[0, ix], self.lla[1, ix], self.lla[2, ix])
            np.testing.assert_array_almost_equal(xyz, self.xyz[:, ix], decimal=self.decimal)
            (x, y, z) = space.geod2ecf(self.lla[:, ix], output="split")
            np.testing.assert_array_almost_equal(xyz, self.xyz[:, ix], decimal=self.decimal)
            (x, y, z) = space.geod2ecf(self.lla[:, ix], output="split")
            np.testing.assert_array_almost_equal(xyz, self.xyz[:, ix], decimal=self.decimal)

    def test_english_units(self) -> None:
        self.lla[2, :] *= M2FT
        xyz = space.geod2ecf(self.lla, units="ft")
        np.testing.assert_array_almost_equal(xyz, M2FT * self.xyz, decimal=self.decimal)

    def test_closed_loop(self) -> None:
        lats = DEG2RAD * np.arange(-90.0, 91.0, 1.0)  # TODO: sofair has problems at the poles!
        num_lats = lats.size
        longs = DEG2RAD * np.linspace(-180.0, 180.0, num_lats)
        num_alts = 10
        alts = 1000.0 * np.logspace(0.0, 4.0, num_alts)

        err_shape = (num_lats, num_alts)
        lat_errs = np.empty(err_shape)
        lon_errs = np.empty(err_shape)
        alt_errs = np.empty(err_shape)
        for ix in range(num_alts):
            xyz = space.geod2ecf(np.vstack((lats, longs, np.full(num_lats, alts[ix]))))
            assert isinstance(xyz, np.ndarray)  # TODO: typing should figure this out based on overloads
            lla = space.ecf2geod(xyz, algorithm="gersten")
            assert isinstance(lla, np.ndarray)  # TODO: typing should figure this out based on overloads
            lat_errs[:, ix] = (lla[0, :] - lats) * self.a
            lon_errs[:, ix] = (lla[1, :] - longs) * self.a
            alt_errs[:, ix] = lla[2, :] - alts[ix]

        np.testing.assert_allclose(lat_errs, np.zeros(err_shape), atol=0.5)
        np.testing.assert_allclose(lon_errs, np.zeros(err_shape), atol=1e-6)
        np.testing.assert_allclose(alt_errs, np.zeros(err_shape), atol=0.1)

    def test_bad_units(self) -> None:
        with self.assertRaises(ValueError) as err:
            space.geod2ecf(self.lla, units="furlong")
        self.assertEqual(str(err.exception), 'Unexpected value for units: "furlong"')

    def test_bad_output(self) -> None:
        with self.assertRaises(ValueError) as err:
            space.geod2ecf(self.lla, output="matrix")
        self.assertEqual(str(err.exception), 'Unexpected value for output: "matrix"')


# %% aerospace.ecf2geod
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_ecf2geod(unittest.TestCase):
    r"""
    Tests the aerospace.ecf2geod function with the following cases:
        Nominal
        Output vectors
        Input vectors
        ALl vectors
        Scalars
        English Units
        Bad Units
        Bad Output
        Bad Algorithm
    """

    def setUp(self) -> None:
        self.a = 6378137.0  # WGS84 elliptical Earth semimajor axis
        f = 1 / 298.257223563
        self.b = self.a * (1 - f)
        self.lla = np.array(
            [
                [0, 0, 0],  # Test all zeros
                [0, np.pi / 4, 0],  # Test 45 degrees at equator
                [np.pi / 2, 0, 0],  # Test North Pole
                [-np.pi / 2, 0, 100],
            ]
        ).T  # Test South Pole
        self.xyz = np.array(
            [[self.a, 0, 0], [self.a / np.sqrt(2), self.a / np.sqrt(2), 0], [0, 0, self.b], [0, 0, -self.b - 100]]
        ).T
        self.decimal = 8

    def test_3d(self) -> None:
        lla = space.ecf2geod(self.xyz)
        np.testing.assert_array_almost_equal(lla, self.lla, decimal=self.decimal)

    def test_output_vectors(self) -> None:
        (lat, lon, alt) = space.ecf2geod(self.xyz, output="split")
        np.testing.assert_array_almost_equal(lat, self.lla[0, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(lon, self.lla[1, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(alt, self.lla[2, :], decimal=self.decimal)

    def test_input_vectors(self) -> None:
        lla = space.ecf2geod(self.xyz[0, :], self.xyz[1, :], self.xyz[2, :])
        np.testing.assert_array_almost_equal(lla, self.lla, decimal=self.decimal)

    def test_all_vectors(self) -> None:
        (lat, lon, alt) = space.ecf2geod(self.xyz[0, :], self.xyz[1, :], self.xyz[2, :], output="split")
        np.testing.assert_array_almost_equal(lat, self.lla[0, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(lon, self.lla[1, :], decimal=self.decimal)
        np.testing.assert_array_almost_equal(alt, self.lla[2, :], decimal=self.decimal)

    def test_scalars(self) -> None:
        for ix in range(self.xyz.shape[1]):
            lla = space.ecf2geod(self.xyz[:, ix])
            np.testing.assert_array_almost_equal(lla, self.lla[:, ix], decimal=self.decimal)
            lla = space.ecf2geod(self.xyz[0, ix], self.xyz[1, ix], self.xyz[2, ix])
            np.testing.assert_array_almost_equal(lla, self.lla[:, ix], decimal=self.decimal)
            (lat, lon, alt) = space.ecf2geod(self.xyz[:, ix], output="split")
            np.testing.assert_array_almost_equal(lla, self.lla[:, ix], decimal=self.decimal)
            (lat, lon, alt) = space.ecf2geod(self.xyz[:, ix], output="split")
            np.testing.assert_array_almost_equal(lla, self.lla[:, ix], decimal=self.decimal)

    def test_english_units(self) -> None:
        lla = space.ecf2geod(M2FT * self.xyz, units="ft")
        self.lla[2, :] *= M2FT
        np.testing.assert_array_almost_equal(lla, self.lla, decimal=self.decimal)

    def test_bad_units(self) -> None:
        with self.assertRaises(ValueError) as err:
            space.ecf2geod(self.xyz, units="furlong")
        self.assertEqual(str(err.exception), 'Unexpected value for units: "furlong"')

    def test_bad_algorithm(self) -> None:
        with self.assertRaises(ValueError) as err:
            space.ecf2geod(self.xyz, algorithm="bad")
        self.assertEqual(str(err.exception), 'Unknown algorithm: "bad"')

    @unittest.expectedFailure
    def test_alt_algorithm(self) -> None:
        # TODO: need to fix the poles in sofair algorithm
        lla = space.ecf2geod(self.xyz, algorithm="sofair")
        np.testing.assert_array_almost_equal(lla, self.lla, decimal=self.decimal)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
