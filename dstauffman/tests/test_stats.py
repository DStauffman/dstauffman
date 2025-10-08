r"""
Test file for the `stats` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in December 2015.

"""

# %% Imports
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np


# %% convert_annual_to_monthly_probability
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_convert_annual_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the convert_annual_to_monthly_probability function with the following cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
        check alias to other name
    """

    def setUp(self) -> None:
        self.monthly = np.arange(10) / 1000.0
        self.annuals = self.monthly.copy()
        for _ in range(1, 12):
            self.annuals[:] = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self) -> None:
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

    def test_scalar(self) -> None:
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals[5])
        self.assertAlmostEqual(monthly, self.monthly[5])

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0.0, 0.5, -1.0]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0.0, 0.5, 1.5]))

    def test_circular(self) -> None:
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)
        annual = dcs.convert_monthly_to_annual_probability(monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

    def test_alias(self) -> None:
        monthly = dcs.ca2mp(self.annuals)
        np.testing.assert_array_almost_equal(monthly, self.monthly)


# %% convert_monthly_to_annual_probability
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_convert_monthly_to_annual_probability(unittest.TestCase):
    r"""
    Tests the convert_monthly_to_annual_probability function with the following cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
        check alias to other name
    """

    def setUp(self) -> None:
        self.monthly = np.arange(10) / 1000.0
        self.annuals = self.monthly.copy()
        for _ in range(1, 12):
            self.annuals[:] = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self) -> None:
        annual = dcs.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)

    def test_scalar(self) -> None:
        annual = dcs.convert_monthly_to_annual_probability(self.monthly[5])
        self.assertAlmostEqual(annual, self.annuals[5])

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            dcs.convert_monthly_to_annual_probability(np.array([0.0, 0.5, -1.0]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            dcs.convert_monthly_to_annual_probability(np.array([0.0, 0.5, 1.5]))

    def test_circular(self) -> None:
        annual = dcs.convert_monthly_to_annual_probability(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)
        monthly = dcs.convert_annual_to_monthly_probability(annual)
        np.testing.assert_array_almost_equal(monthly, self.monthly)

    def test_alias(self) -> None:
        annual = dcs.cm2ap(self.monthly)
        np.testing.assert_array_almost_equal(annual, self.annuals)


# %% prob_to_rate
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_prob_to_rate(unittest.TestCase):
    r"""
    Tests the prob_to_rate function with the following cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
    """

    def setUp(self) -> None:
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        self.rate = np.hstack((0.0, -np.log(1 - self.prob[1:-1]) / self.time, np.inf))

    def test_conversion(self) -> None:
        rate = dcs.prob_to_rate(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self) -> None:
        rate = dcs.prob_to_rate(self.prob[15], time=self.time)
        self.assertAlmostEqual(rate, self.rate[15])
        rate = dcs.prob_to_rate(float(self.prob[15]), time=self.time)
        self.assertAlmostEqual(rate, float(self.rate[15]))
        rate = dcs.prob_to_rate(1)
        self.assertEqual(rate, np.inf)
        rate = dcs.prob_to_rate(0)
        self.assertEqual(rate, 0.0)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            dcs.prob_to_rate(np.array([0.0, 0.5, -1.0]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            dcs.prob_to_rate(np.array([0.0, 0.5, 1.5]))

    def test_circular(self) -> None:
        rate = dcs.prob_to_rate(self.prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = dcs.rate_to_prob(rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)


# %% rate_to_prob
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_rate_to_prob(unittest.TestCase):
    r"""
    Tests the rate_to_prob function with the following cases:
        convert a vector from monthly to annual
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from monthly to annual and then back
    """

    def setUp(self) -> None:
        self.prob = np.arange(0, 1.01, 0.01)
        self.time = 5
        self.rate = np.hstack((0.0, -np.log(1 - self.prob[1:-1]) / self.time, np.inf))

    def test_conversion(self) -> None:
        prob = dcs.rate_to_prob(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self) -> None:
        prob = dcs.rate_to_prob(self.rate[20], time=self.time)
        self.assertAlmostEqual(prob, self.prob[20])
        prob = dcs.rate_to_prob(float(self.rate[20]), time=self.time)
        self.assertAlmostEqual(prob, float(self.prob[20]))
        prob = dcs.rate_to_prob(0)
        self.assertEqual(prob, 0.0)
        prob = dcs.rate_to_prob(np.inf)
        self.assertEqual(prob, 1)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            dcs.rate_to_prob(np.array([0.0, 0.5, -1.0]))

    def test_infinity(self) -> None:
        prob = dcs.rate_to_prob(np.inf)
        self.assertAlmostEqual(prob, 1.0)

    def test_circular(self) -> None:
        prob = dcs.rate_to_prob(self.rate, self.time)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = dcs.prob_to_rate(prob, self.time)
        np.testing.assert_array_almost_equal(rate, self.rate)


# %% annual_rate_to_monthly_probability
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_annual_rate_to_monthly_probability(unittest.TestCase):
    r"""
    Tests the annual_rate_to_monthly_probability function with the following cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
        check alias to other name
    """

    def setUp(self) -> None:
        self.prob = np.arange(0, 0.05, 1)
        self.rate = -np.log(1 - self.prob) * 12

    def test_conversion(self) -> None:
        prob = dcs.annual_rate_to_monthly_probability(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_scalar(self) -> None:
        prob = dcs.annual_rate_to_monthly_probability(0)
        self.assertIn(prob, self.prob)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            dcs.annual_rate_to_monthly_probability(np.array([0.0, 0.5, -1.0]))

    def test_infinity(self) -> None:
        prob = dcs.annual_rate_to_monthly_probability(np.inf)
        self.assertAlmostEqual(prob, 1.0)

    def test_circular(self) -> None:
        prob = dcs.annual_rate_to_monthly_probability(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)
        rate = dcs.monthly_probability_to_annual_rate(prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_alias(self) -> None:
        prob = dcs.ar2mp(self.rate)
        np.testing.assert_array_almost_equal(prob, self.prob)


# %% monthly_probability_to_annual_rate
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_monthly_probability_to_annual_rate(unittest.TestCase):
    r"""
    Tests the monthly_probability_to_annual_rate function with the following cases:
        convert a vector from annual to monthly
        convert a scalar
        convert a number less than zero (raise error)
        convert a number greater than one (raise error)
        convert a vector from annual to monthly and then back
        check alias to other name
    """

    def setUp(self) -> None:
        self.prob = np.arange(0, 0.05, 1)
        self.rate = -np.log(1 - self.prob) * 12

    def test_conversion(self) -> None:
        rate = dcs.monthly_probability_to_annual_rate(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)

    def test_scalar(self) -> None:
        rate = dcs.monthly_probability_to_annual_rate(0)
        self.assertIn(rate, self.rate)

    def test_lt_zero(self) -> None:
        with self.assertRaises(ValueError):
            dcs.monthly_probability_to_annual_rate(np.array([0.0, 0.5, -1.0]))

    def test_gt_one(self) -> None:
        with self.assertRaises(ValueError):
            dcs.monthly_probability_to_annual_rate(np.array([0.0, 0.5, 1.5]))

    def test_circular(self) -> None:
        rate = dcs.monthly_probability_to_annual_rate(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)
        prob = dcs.rate_to_prob(rate)
        np.testing.assert_array_almost_equal(prob, self.prob)

    def test_alias(self) -> None:
        rate = dcs.ar2mp(self.prob)
        np.testing.assert_array_almost_equal(rate, self.rate)


# %% rand_draw
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_rand_draw(unittest.TestCase):
    r"""
    Tests the rand_draw function with the following cases:
        Nominal
        Negative and zero values
        Large values
        No quality checks
    """

    def setUp(self) -> None:
        self.chances = np.array([0.1, 0.2, 0.8, 0.9])
        self.prng = np.random.default_rng()

    def test_nominal(self) -> None:
        is_set = dcs.rand_draw(self.chances, self.prng)
        self.assertTrue(is_set.dtype == bool)

    def test_zeros(self) -> None:
        is_set = dcs.rand_draw(np.array([-5, 0, 0.5]), self.prng)
        self.assertFalse(is_set[0])
        self.assertFalse(is_set[1])

    def test_large_values(self) -> None:
        is_set = dcs.rand_draw(np.array([5, 1, 0.5, 1000, np.inf]), self.prng)
        self.assertTrue(is_set[0])
        self.assertTrue(is_set[1])
        self.assertTrue(is_set[3])
        self.assertTrue(is_set[4])

    def test_without_checks(self) -> None:
        is_set = dcs.rand_draw(np.array([-5, 0.5, 5]), self.prng, check_bounds=False)
        self.assertFalse(is_set[0])
        self.assertTrue(is_set[2])


# %% apply_prob_to_mask
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_apply_prob_to_mask(unittest.TestCase):
    r"""
    Tests the apply_prob_to_mask function with the following cases:
        Nominal
        Inplace
        Zero prob
        100% prob
    """

    def setUp(self) -> None:
        self.prng = np.random.default_rng()

    def test_nominal(self) -> None:
        mask = self.prng.random(50000) < 0.5
        prob = 0.3
        num = np.count_nonzero(mask)
        out = dcs.apply_prob_to_mask(mask, prob, self.prng)
        self.assertIsNot(out, mask)
        self.assertLess(num, 30000, "Too many trues in mask.")
        self.assertLess(np.count_nonzero(out), 4 * num // 10, "Too many trues in out.")

    def test_inplace(self) -> None:
        mask = self.prng.random(50000) < 0.2
        prob = 0.8
        num = np.count_nonzero(mask)
        out = dcs.apply_prob_to_mask(mask, prob, self.prng, inplace=True)
        self.assertIs(out, mask)
        self.assertLess(num, 20000, "Too many trues in mask.")
        self.assertLess(np.count_nonzero(out), num, "Too many trues in out.")

    def test_zero_prob(self) -> None:
        mask = self.prng.random(1000) < 0.8
        prob = 0.0
        out = dcs.apply_prob_to_mask(mask, prob, self.prng)
        self.assertTrue(np.all(~out))

    def test_one_prob(self) -> None:
        mask = self.prng.random(1000) < 0.4
        prob = 1.0
        out = dcs.apply_prob_to_mask(mask, prob, self.prng)
        self.assertIsNot(out, mask)
        np.testing.assert_array_equal(out, mask)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
