r"""
Test file for the `orbits` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

#%% Imports
import unittest

from dstauffman import get_data_dir, HAVE_NUMPY, read_text_file
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np

#%% aerospace.OrbitType
class Test_aerospace_OrbitType(unittest.TestCase):
    r"""
    Tests the aerospace.OrbitType class with the following cases:
        Nominal
    """
    def test_nominal(self) -> None:
        orbit_types = space.OrbitType
        self.assertEqual(orbit_types.list_of_names(), ['uninitialized', 'elliptic', 'parabolic', 'hyperbolic'])

#%% aerospace.Elements
class Test_aerospace_Elements(unittest.TestCase):
    r"""
    Tests the aerospace.Elements class with the following cases:
        Initialization
    """
    def setUp(self) -> None:
        self.floats = {'a', 'e', 'i', 'W', 'w', 'vo', 'p', 'uo', 'P', 'lo', 'T'}
        self.ints = {'type'}
        self.bools = {'equatorial', 'circular'}
        self.dates = {'t'}
        self.all = self.floats | self.ints | self.bools | self.dates
        self.num = 10
        self.nans = np.full(self.num, np.nan)

    def test_init(self) -> None:
        elements = space.Elements()
        self.assertIsInstance(elements, space.Elements)
        for key in self.floats:
            self.assertIsNone(getattr(elements, key))
        for key in self.ints:
            self.assertEqual(getattr(elements, key), 0)
        for key in self.bools:
            self.assertFalse(getattr(elements, key))
        for key in self.dates:
            self.assertTrue(np.isnat(getattr(elements, key)))
        self.assertEqual(vars(elements).keys(), self.all)

    def test_equality(self) -> None:
        elements1 = space.Elements()
        elements2 = space.Elements()
        self.assertEqual(elements1, elements2)

    def test_preallocated(self) -> None:
        elements = space.Elements(self.num)
        for key in self.floats:
            np.testing.assert_array_equal(getattr(elements, key), self.nans)
        for key in self.ints:
            np.testing.assert_array_equal(getattr(elements, key), np.zeros(self.num, dtype=int))
        for key in self.bools:
            np.testing.assert_array_equal(getattr(elements, key), np.zeros(self.num, dtype=bool))
        for key in self.dates:
            self.assertTrue(np.all(np.isnat(getattr(elements, key))))
            self.assertEqual(getattr(elements, key).shape, (self.num, ))
        self.assertEqual(vars(elements).keys(), self.all)

#%% aerospace.jd_to_numpy
class Test_aerospace_jd_to_numpy(unittest.TestCase):
    r"""
    Tests the aerospace.jd_to_numpy function with the following cases:
        Nominal
    """
    def test_nominal(self) -> None:
        date = space.jd_to_numpy(2451545.0)
        self.assertEqual(date, np.datetime64('2000-01-01T00:00:00'))

#%% aerospace.two_line_elements
class Test_aerospace_two_line_elements(unittest.TestCase):
    r"""
    Tests the aerospace.two_line_elements function with the following cases:
        Single
        File
    """
    def test_single(self) -> None:
        line1 = '1 25544U 98067A   06132.29375000  .00013633  00000-0  92740-4 0  9181'
        line2 = '2 25544  51.6383  12.2586 0009556 188.7367 320.5459 15.75215761427503'
        elements = space.two_line_elements(line1, line2)
        exp            = space.Elements()
        exp.a          = 6722342.198683569
        exp.e          = 0.0009556
        exp.i          = 0.9012583551325879
        exp.W          = 0.21395293168497687
        exp.w          = 3.294076834348782
        exp.vo         = 5.593365747043137
        exp.p          = None
        exp.uo         = 8.88744258139192
        exp.P          = 3.5080297660337587
        exp.lo         = 9.101395513076895
        exp.T          = None
        exp.type       = space.OrbitType.elliptic
        exp.equatorial = False
        exp.circular   = False
        exp.t          = space.jd_to_numpy(2453867.79375)
        self.assertEqual(elements, exp)

    def test_file(self) -> None:
        lines = read_text_file(get_data_dir() / 'gps-ops_2021_07_28.txt').split('\n')
        for i in range(len(lines) // 3):  # TODO: vectorize this function so this loop can be eliminated
            this_name = lines[3*i]
            line1 = lines[3*i + 1]
            line2 = lines[3*i + 2]
            elements = space.two_line_elements(line1, line2)
            self.assertTrue(this_name.startswith('GPS '))
            # TODO: do something with this data

#%% aerospace.rv_2_oe
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_aerospace_rv_2_oe(unittest.TestCase):
    r"""
    Tests the aerospace.rv_2_oe function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.r1 = np.array([1., 0., 0.])
        self.v1 = np.array([0., 1., 0.])
        self.a1 = 1.0

        self.r2 = np.array([space.EARTH['a']+1_000_000., 0., 0.])
        self.v2 = np.array([0., 7777., 0.])
        self.a2 = 8378813.850732614

        self.floats = {'a', 'e', 'i', 'W', 'w', 'vo', 'p', 'uo', 'P', 'lo', 'T'}
        self.ints = {'type'}
        self.bools = {'equatorial', 'circular'}
        self.dates = {'t'}

    def test_nominal(self) -> None:
        elements1 = space.rv_2_oe(self.r1, self.v1)
        self.assertEqual(elements1.a, self.a1)
        for key in self.floats:
            self.assertIsInstance(getattr(elements1, key), float, f'Bad key: {key}')
        for key in self.ints:
            self.assertIsInstance(getattr(elements1, key), int, f'Bad key: {key}')
        for key in self.bools:
            self.assertIsInstance(getattr(elements1, key), bool, f'Bad key: {key}')
        for key in self.dates:
            self.assertIsInstance(getattr(elements1, key), np.datetime64, f'Bad key: {key}')

        elements2 = space.rv_2_oe(self.r2, self.v2, mu=space.MU_EARTH)
        self.assertEqual(elements2.a, self.a2)

    def test_vectorized(self) -> None:
        r = np.column_stack((self.r1, self.r2))
        v = np.column_stack((self.v1, self.v2))
        mu = np.array([1., space.MU_EARTH])
        elements = space.rv_2_oe(r, v, mu=mu)
        exp_a = np.array([self.a1, self.a2])
        np.testing.assert_array_equal(elements.a, exp_a)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
