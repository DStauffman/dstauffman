r"""
Test file for the `orbits` module of the "dstauffman.aerospace" library.

Notes
-----
#.  Written by David C. Stauffer in July 2021.

"""

# %% Imports
import unittest

from slog import read_text_file

from dstauffman import get_data_dir, HAVE_NUMPY, NP_DATETIME_FORM, NP_NAT
import dstauffman.aerospace as space

if HAVE_NUMPY:
    import numpy as np


# %% aerospace.OrbitType
class Test_aerospace_OrbitType(unittest.TestCase):
    r"""
    Tests the aerospace.OrbitType class with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        orbit_types = space.OrbitType
        self.assertEqual(orbit_types.list_of_names(), ["uninitialized", "elliptic", "parabolic", "hyperbolic"])


# %% aerospace.Elements
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_Elements(unittest.TestCase):
    r"""
    Tests the aerospace.Elements class with the following cases:
        Initialization
    """

    def setUp(self) -> None:
        self.floats = {"a", "e", "i", "W", "w", "vo", "p", "uo", "P", "lo", "T"}
        self.ints = {"type"}
        self.bools = {"equatorial", "circular"}
        self.dates = {"t"}
        self.all = self.floats | self.ints | self.bools | self.dates
        self.num = 10
        self.nans = np.full(self.num, np.nan)

    def test_init(self) -> None:
        elements = space.Elements()
        self.assertIsInstance(elements, space.Elements)
        for key in self.all:
            self.assertEqual(getattr(elements, key).size, 0)
        for key in self.floats:
            self.assertEqual(getattr(elements, key).dtype, float)
        for key in self.ints:
            self.assertEqual(getattr(elements, key).dtype, int)
        for key in self.bools:
            self.assertEqual(getattr(elements, key).dtype, bool)
        for key in self.dates:
            self.assertEqual(getattr(elements, key).dtype, NP_DATETIME_FORM)
        self.assertEqual(vars(elements).keys(), self.all)

    def test_equality(self) -> None:
        elements1 = space.Elements()
        elements2 = space.Elements()
        self.assertEqual(elements1, elements2)

    def test_length(self) -> None:
        elements1 = space.Elements()
        elements2 = space.Elements(1)
        elements3 = space.Elements(5)
        self.assertEqual(len(elements1), 0)
        self.assertEqual(len(elements2), 1)
        self.assertEqual(len(elements3), 5)

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
            self.assertEqual(getattr(elements, key).shape, (self.num,))
        self.assertEqual(vars(elements).keys(), self.all)

    def test_indexing(self) -> None:
        elements = space.Elements(0)
        with self.assertRaises(IndexError) as err:
            elements[0]
        self.assertEqual(str(err.exception), "index 0 is out of bounds for axis 0 with size 0")
        elements_full = space.Elements(self.num)
        elements_full.a[:] = np.arange(self.num, dtype=float)  # type: ignore[index]
        exp = space.Elements(1)
        exp.a = 3.0
        elements3 = elements_full[3]
        self.assertEqual(elements3, exp)


# %% aerospace.orbits._zero_divide
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_orbits__zero_divide(unittest.TestCase):
    r"""
    Tests the aerospace.orbits._zero_divide function with the following cases:
        Non-zero denominator
        Zero denominator
        All zeros
        Vector of zeros
    """

    def test_division(self) -> None:
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([2.0, 4.0, 6.0])
        out = space.orbits._zero_divide(x, y)
        exp = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(out, exp)

    def test_zero_denominator(self) -> None:
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([2.0, 0.0, 6.0])
        out = space.orbits._zero_divide(x, y)
        exp = np.array([5.0, 0.0, 5.0])
        np.testing.assert_array_equal(out, exp)

    def test_all_zeros(self) -> None:
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([2.0, 4.0, 6.0])
        out = space.orbits._zero_divide(x, y)
        exp = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(out, exp)

    def test_division_by_zero_vector(self) -> None:
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([0.0, 0.0, 0.0])
        out = space.orbits._zero_divide(x, y)
        exp = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(out, exp)


# %% aerospace.orbits._inf_divide
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_orbits__inf_divide(unittest.TestCase):
    r"""
    Tests the aerospace.orbits._inf_divide function with the following cases:
        Normal division
        Division by zero
        Division of zero by a number
        Division of two zeros
        Division with arrays
    """

    def test_division(self) -> None:
        out = space.orbits._inf_divide(10.0, 2.0)
        exp = 5.0
        self.assertEqual(out, exp)

    def test_divide_by_zero(self) -> None:
        out = space.orbits._inf_divide(10.0, 0.0)
        self.assertTrue(np.isinf(out))

    def test_zero_divide(self) -> None:
        out = space.orbits._inf_divide(0.0, 2.0)
        exp = 0.0
        self.assertEqual(out, exp)

    def test_zero_zero(self) -> None:
        out = space.orbits._inf_divide(0.0, 0.0)
        exp = np.inf
        self.assertEqual(out, exp)

    def test_arrays(self) -> None:
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([2.0, 0.0, 3.0])
        exp = np.array([5.0, np.inf, 10.0])
        out = space.orbits._inf_divide(x, y)
        np.testing.assert_array_equal(out, exp)


# %% aerospace.orbits._inf_multiply
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_orbits__inf_multiply(unittest.TestCase):
    r"""
    Tests the aerospace.orbits._inf_multiply function with the following cases:
        Finite numbers
        Infinite numbers
        zero
        inf times zero
        NaNs
    """

    def test_finite_numbers(self) -> None:
        # Test multiplication of finite numbers
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        out = space.orbits._inf_multiply(x, y)
        exp = np.array([4.0, 10.0, 18.0])
        np.testing.assert_array_equal(out, exp)

    def test_infinite_numbers(self) -> None:
        # Test multiplication involving infinite numbers
        x = np.array([1.0, np.inf, 3.0])
        y = np.array([4.0, 5.0, np.inf])
        out = space.orbits._inf_multiply(x, y)
        exp = np.array([4.0, np.inf, np.inf])
        np.testing.assert_array_equal(out, exp)

    def test_zero(self) -> None:
        # Test multiplication involving zero
        x = np.array([0.0, 2.0, 0.0])
        y = np.array([4.0, 0.0, np.inf])
        out = space.orbits._inf_multiply(x, y)
        exp = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(out, exp)

    def test_inf_times_zero(self) -> None:
        # Specifically test inf * 0
        x = np.array([np.inf, 2.0, np.inf])
        y = np.array([0.0, 0.0, 0.0])
        out = space.orbits._inf_multiply(x, y)
        exp = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(out, exp)

    def test_nan(self) -> None:
        x = np.array([np.nan, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        out = space.orbits._inf_multiply(x, y)
        exp = np.array([np.nan, 10.0, 18.0])
        np.testing.assert_array_equal(out, exp)


# %% aerospace.orbits._fix_instab
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_orbits__fix_instab(unittest.TestCase):
    r"""
    Tests the aerospace.orbits._fix_instab function with the following cases:
        Near position one
        Near negative one
        Unchanged values
        Empty array
        Zero precision
    """

    def test_fix_instab_positive(self) -> None:
        """Test fixing instabilities near positive one."""
        x = np.array([0.5, 1.0, 1.00001, 2.0])
        precision = 0.001
        exp = np.array([0.5, 1.0, 1.0, 2.0])
        space.orbits._fix_instab(x, precision)
        np.testing.assert_array_equal(x, exp)

    def test_fix_instab_negative(self) -> None:
        """Test fixing instabilities near negative one."""
        x = np.array([-0.5, -1.0, -1.00001, -2.0])
        precision = 0.001
        exp = np.array([-0.5, -1.0, -1.0, -2.0])
        space.orbits._fix_instab(x, precision)
        np.testing.assert_array_equal(x, exp)

    def test_fix_instab_no_change(self) -> None:
        """Test that values outside the instability range are not changed."""
        x = np.array([0.5, 0.8, 0.9, 1.1])
        precision = 0.001
        exp = np.array([0.5, 0.8, 0.9, 1.1])
        space.orbits._fix_instab(x, precision)
        np.testing.assert_array_equal(x, exp)

    def test_fix_instab_empty_array(self) -> None:
        """Test that an empty array is not modified."""
        x = np.array([])
        precision = 0.001
        exp = np.array([])
        space.orbits._fix_instab(x, precision)
        np.testing.assert_array_equal(x, exp)

    def test_fix_instab_precision_zero(self) -> None:
        """Test that a precision of zero does not modify the array."""
        x = np.array([0.5, 1.0, 1.00001, 2.0])
        precision = 0.0
        exp = np.array([0.5, 1.0, 1.00001, 2.0])
        space.orbits._fix_instab(x, precision)
        np.testing.assert_array_equal(x, exp)


# %% aerospace.two_line_elements
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_two_line_elements(unittest.TestCase):
    r"""
    Tests the aerospace.two_line_elements function with the following cases:
        Single
        File
    """

    def test_single(self) -> None:
        line1 = "1 25544U 98067A   06132.29375000  .00013633  00000-0  92740-4 0  9181"
        line2 = "2 25544  51.6383  12.2586 0009556 188.7367 320.5459 15.75215761427503"
        elements = space.two_line_elements(line1, line2)
        # fmt: off
        exp            = space.Elements(1)
        exp.a          = 6722154.278502964
        exp.e          = 0.0009556
        exp.i          = 0.9012583551325879
        exp.W          = 0.21395293168497687
        exp.w          = 3.294076834348782
        exp.vo         = 5.593365747043137
        exp.p          = 6722148.1400242
        exp.uo         = 8.88744258139192
        exp.P          = 3.5080297660337587
        exp.lo         = 9.101395513076895
        exp.T          = 5484.96289455295
        exp.type       = space.OrbitType.elliptic
        exp.equatorial = False
        exp.circular   = False
        exp.t          = space.jd_to_numpy(2453867.79375)
        # fmt: on
        self.assertEqual(elements, exp)

    def test_file(self) -> None:
        lines = read_text_file(get_data_dir() / "gps-ops_2021_07_28.txt").split("\n")
        for i in range(len(lines) // 3):  # TODO: vectorize this function so this loop can be eliminated
            this_name = lines[3 * i]
            line1 = lines[3 * i + 1]
            line2 = lines[3 * i + 2]
            elements = space.two_line_elements(line1, line2)
            self.assertTrue(this_name.startswith("GPS "))
            self.assertGreater(elements.e, 0.0)  # type: ignore[arg-type]
            self.assertLess(elements.e, 1.0)  # type: ignore[arg-type]


# %% aerospace.rv_2_oe
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_rv_2_oe(unittest.TestCase):
    r"""
    Tests the aerospace.rv_2_oe function with the following cases:
        Nominal
        Vectorized
        Zeros
    """

    def setUp(self) -> None:
        self.r1 = np.array([1.0, 0.0, 0.0])
        self.v1 = np.array([0.0, 1.0, 0.0])
        self.a1 = 1.0

        self.r2 = np.array([space.EARTH["a"] + 1_000_000.0, 0.0, 0.0])
        self.v2 = np.array([0.0, 7777.0, 0.0])
        self.a2 = 8379707.279086602

        self.floats = {"a", "e", "i", "W", "w", "vo", "p", "uo", "P", "lo", "T"}
        self.enums = {"type"}
        self.bools = {"equatorial", "circular"}
        self.dates = {"t"}

    def test_nominal(self) -> None:
        elements1 = space.rv_2_oe(self.r1, self.v1)
        self.assertEqual(elements1.a, self.a1)
        for key in self.floats:
            self.assertIsInstance(getattr(elements1, key)[()], np.float64, f"Bad key: {key}")
        for key in self.enums:
            self.assertEqual(getattr(elements1, key), space.OrbitType.elliptic, f"Bad key: {key}")
        for key in self.bools:
            self.assertIsInstance(getattr(elements1, key)[()], np.bool_, f"Bad key: {key}")
        for key in self.dates:
            self.assertIsInstance(getattr(elements1, key), np.datetime64, f"Bad key: {key}")

        elements2 = space.rv_2_oe(self.r2, self.v2, mu=space.MU_EARTH)
        self.assertEqual(elements2.a, self.a2)

    def test_vectorized(self) -> None:
        r = np.column_stack((self.r1, self.r2))
        v = np.column_stack((self.v1, self.v2))
        mu = np.array([1.0, space.MU_EARTH])
        elements = space.rv_2_oe(r, v, mu=mu)
        exp_a = np.array([self.a1, self.a2])
        np.testing.assert_array_equal(elements.a, exp_a)

    def test_zeros(self) -> None:
        r0 = np.zeros(3)
        v0 = np.zeros(3)
        # fmt: off
        exp1            = space.Elements(1)
        exp1.a          = 0.0
        exp1.e          = 0.0
        exp1.i          = 0.0
        exp1.W          = 0.0
        exp1.w          = 0.0
        exp1.vo         = 0.0
        exp1.p          = 0.0
        exp1.uo         = 0.0
        exp1.P          = 0.0
        exp1.lo         = 0.0
        exp1.T          = 0.0
        exp1.type       = space.OrbitType.elliptic
        exp1.equatorial = True
        exp1.circular   = True
        exp1.t          = NP_NAT
        exp2            = space.Elements(1)
        exp2.a          = np.inf
        exp2.e          = 1.0
        exp2.i          = 0.0
        exp2.W          = 0.0
        exp2.w          = np.pi
        exp2.vo         = np.pi
        exp2.p          = 0.0
        exp2.uo         = 0.0
        exp2.P          = np.pi
        exp2.lo         = 0.0
        exp2.T          = np.inf
        exp2.type       = space.OrbitType.elliptic
        exp2.equatorial = True
        exp2.circular   = False
        exp2.t          = NP_NAT
        # fmt: on
        oe1 = space.rv_2_oe(r0, self.v1)
        self.assertEqual(oe1, exp1)
        oe2 = space.rv_2_oe(self.r1, v0)
        self.assertEqual(oe2, exp2)
        oe3 = space.rv_2_oe(r0, v0)
        self.assertEqual(oe3, exp1)
        exp4 = exp1.combine(exp2).combine(exp1)
        oe4 = space.rv_2_oe(np.column_stack((r0, self.r1, r0)), np.column_stack((self.v1, v0, v0)))
        self.assertEqual(oe4, exp4)

    def test_nans1(self) -> None:
        oe = space.rv_2_oe(np.full(3, np.nan), np.full(3, np.nan))
        self.assertTrue(np.isnan(oe.a))

    def test_nans2(self) -> None:
        r = np.column_stack((self.r1, self.r2, np.full(3, np.nan)))
        v = np.column_stack((self.v1, self.v2, np.full(3, np.nan)))
        mu = np.array([1.0, space.MU_EARTH, space.MU_EARTH])
        elements = space.rv_2_oe(r, v, mu=mu)
        exp_a = np.array([self.a1, self.a2, np.nan])
        np.testing.assert_array_equal(elements.a, exp_a)


# %% aerospace.oe_2_rv
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_aerospace_oe_2_rv(unittest.TestCase):
    r"""
    Tests the aerospace.oe_2_rv function with the following cases:
        Round-trip
    """

    def setUp(self) -> None:
        # fmt: off
        self.r             = np.array([-0.8, 0.6, 0.5])
        self.v             = np.array([-0.4, -0.8, 0.6])
        self.oe            = space.Elements(1)
        self.oe.a          = 1.590193260353663
        self.oe.e          = 0.3169963595807386
        self.oe.i          = 0.7439637316462275
        self.oe.W          = 1.923786714621807
        self.oe.w          = 0.229262256060102
        self.oe.vo         = 0.4920582123765659
        self.oe.p          = 1.4304000000000001
        self.oe.uo         = 0.7213204684366679
        self.oe.P          = 2.153048970681909
        self.oe.lo         = 2.645107183058475
        self.oe.T          = 12.599541202147304
        self.oe.type       = space.OrbitType.elliptic
        self.oe.equatorial = False
        self.oe.circular   = False
        self.oe.t          = NP_NAT
        # fmt: on

    def test_nominal(self) -> None:
        (r, v) = space.oe_2_rv(self.oe)
        np.testing.assert_array_almost_equal(r, self.r, 14)
        np.testing.assert_array_almost_equal(v, self.v, 14)

    def test_vectorized(self) -> None:
        pass  # TODO: write this

    def test_zeros(self) -> None:
        # fmt: off
        r0             = np.zeros(3)
        v0             = np.zeros(3)
        r1             = np.array([1.0, 0.0, 0.0])
        v1             = np.array([0.0, np.sqrt(2), 0.0])
        oe1            = space.Elements(1)
        oe1.a          = 0.0
        oe1.e          = 0.0
        oe1.i          = 0.0
        oe1.W          = 0.0
        oe1.w          = 0.0
        oe1.vo         = 0.0
        oe1.p          = 0.0
        oe1.uo         = 0.0
        oe1.P          = 0.0
        oe1.lo         = 0.0
        oe1.T          = 0.0
        oe1.type       = space.OrbitType.elliptic
        oe1.equatorial = True
        oe1.circular   = True
        oe1.t          = NP_NAT
        oe2            = space.Elements(1)
        oe2.a          = np.inf
        oe2.e          = 1.0
        oe2.i          = 0.0
        oe2.W          = 0.0
        oe2.w          = 0.0  # np.pi
        oe2.vo         = 0.0  # np.pi
        oe2.p          = 2.0  # 0.0
        oe2.uo         = 0.0
        oe2.P          = 0.0  # np.pi
        oe2.lo         = 0.0
        oe2.T          = np.inf
        oe2.type       = space.OrbitType.parabolic
        oe2.equatorial = True
        oe2.circular   = False
        oe2.t          = NP_NAT
        # fmt: on
        (r, v) = space.oe_2_rv(oe1)
        np.testing.assert_array_equal(r, r0)
        np.testing.assert_array_equal(v, v0)
        (r, v) = space.oe_2_rv(oe2)
        np.testing.assert_array_equal(r, r1)
        np.testing.assert_array_equal(v, v1)

    @unittest.skip("Skip this slow test. Run for extra credit.")
    def test_round_trip(self) -> None:  # pragma: no cover
        oe = space.Elements(1)
        for a in [0.1, 0.9, 1.0, 2.0, 10.0]:
            for e in [0.01, 0.1, 0.5, 0.999]:  # 1.0, 2.0, 5.0, 1000., np.inf
                for i in [0.001, 1 / 360, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                    for w in [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi - 0.01]:
                        for W in [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi - 0.01]:
                            for nu in [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi - 0.01]:
                                oe.a = a
                                oe.e = e
                                oe.i = i
                                oe.W = W
                                oe.w = w
                                oe.vo = nu
                                oe.type = space.OrbitType.elliptic  # if E < 0 else space.OrbitType.parabolic if E == 0 else
                                oe.equatorial = True if i == 0 else False
                                oe.circular = True if e == 0 else False
                                (r, v) = space.oe_2_rv(oe)
                                oe2 = space.rv_2_oe(r, v)
                                np.testing.assert_array_almost_equal(oe.a, oe2.a, err_msg="a is different")
                                np.testing.assert_array_almost_equal(oe.e, oe2.e, err_msg="e is different")
                                np.testing.assert_array_almost_equal(oe.i, oe2.i, err_msg="i is different")
                                np.testing.assert_array_almost_equal(oe.w, oe2.w, err_msg="w is different")
                                np.testing.assert_array_almost_equal(oe.W, oe2.W, err_msg="W is different")
                                np.testing.assert_array_almost_equal(oe.vo, oe2.vo, err_msg="nu is different")


# %% aerospace.advance_true_anomaly
class Test_aerospace_advance_true_anomaly(unittest.TestCase):
    r"""
    Tests the aerospace.advance_true_anomaly function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% aerospace.advance_elements
class Test_aerospace_advance_elements(unittest.TestCase):
    r"""
    Tests the aerospace.advance_elements function with the following cases:
        TBD
    """

    pass  # TODO: write this


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
