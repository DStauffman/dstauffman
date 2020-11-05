r"""
Test file for the `health` module of the "dstauffman.health" library.

Notes
-----
#.  Written by David C. Stauffer in October 2017.
#.  Renamed from analysis.py to health.py by David C. Stauffer in May 2020.
"""

#%% Imports
from typing import Optional
import unittest

from dstauffman import HAVE_MPL, HAVE_NUMPY
from dstauffman.plotting import Opts, Plotter
import dstauffman.health as health

if HAVE_MPL:
    import matplotlib.pyplot as plt
    plotter = Plotter(False)
if HAVE_NUMPY:
    import numpy as np

#%% health.dist_enum_and_mons
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_dist_enum_and_mons(unittest.TestCase):
    r"""
    Tests the health.dist_enum_and_mons function with the following cases:
        Nominal usage
        All in one bin
    """
    def setUp(self) -> None:
        self.num = 100000
        self.distribution = 1./100*np.array([10, 20, 30, 40])
        self.max_months = np.array([1, 10, 50, 5])
        self.max_months = np.array([1, 1, 1, 1])
        self.prng = np.random.RandomState()
        self.per_lim = 0.01

    def test_calling(self) -> None:
        (state, mons) = health.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months)
        breakout = np.histogram(state, bins=[0.5, 1.5, 2.5, 3.5, 4.5])[0]
        breakout_per = breakout / self.num
        for ix in range(len(self.distribution)):
            self.assertTrue(np.abs(breakout_per[ix] - self.distribution[ix]) <= self.per_lim)
        self.assertTrue(np.all(mons <= self.max_months[state-1]) and np.all(mons >= 1))

    def test_all_in_one_bin(self) -> None:
        for i in range(4):
            temp = np.zeros(4)
            temp[i] = 1
            (tb_state, _) = health.dist_enum_and_mons(self.num, temp, self.prng, max_months=self.max_months)
            self.assertTrue(np.all(tb_state == i+1))

    def test_alpha_and_beta(self) -> None:
        pass #TODO: write this

    def test_different_start_num(self) -> None:
        (state1, mons1) = health.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months)
        (state2, mons2) = health.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months, start_num=101)
        np.testing.assert_array_equal(set(state1), {1, 2, 3, 4})
        np.testing.assert_array_equal(set(state2), {101, 102, 103, 104})
        np.testing.assert_array_equal(set(mons1), {1})
        np.testing.assert_array_equal(set(mons2), {1})

    def test_scalar_max_months(self) -> None:
        (state1, mons1) = health.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=1)
        (state2, mons2) = health.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=3)
        np.testing.assert_array_equal(set(state1), {1, 2, 3, 4})
        np.testing.assert_array_equal(set(state2), {1, 2, 3, 4})
        np.testing.assert_array_equal(set(mons1), {1})
        np.testing.assert_array_equal(set(mons2), {1, 2, 3})

    def test_max_months_is_none(self) -> None:
        state = health.dist_enum_and_mons(self.num, self.distribution, self.prng)
        np.testing.assert_array_equal(set(state), {1, 2, 3, 4})

    def test_single_num(self) -> None:
        self.num = 1
        (state, mons) = health.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months)
        self.assertIn(state[0], {1, 2, 3, 4})
        self.assertTrue(mons[0] <= max(self.max_months))

    def test_zero_num(self) -> None:
        self.num = 0
        (state, mons) = health.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months)
        self.assertTrue(len(state) == 0)
        self.assertTrue(len(mons) == 0)

    def test_unique_dists(self) -> None:
        num = 3
        dist = np.array([[0, 0, 0, 1], [1, 0, 0, 0],[0, 0.5, 0.5, 0]])
        state = health.dist_enum_and_mons(num, dist, self.prng, start_num=1)
        self.assertEqual(state[0], 4)
        self.assertEqual(state[1], 1)
        self.assertIn(state[2], {2, 3})

    def test_bad_distribution1(self) -> None:
        dist = np.array([0, 0.1, 0.2])
        with self.assertRaises(AssertionError) as context:
            health.dist_enum_and_mons(self.num, dist, self.prng)
        self.assertEqual(str(context.exception), "Given distribution doesn't sum to 1.")

    def test_bad_distribution2(self) -> None:
        dist = np.array([0, 1.1, 0.2])
        with self.assertRaises(AssertionError) as context:
            health.dist_enum_and_mons(self.num, dist, self.prng)
        self.assertEqual(str(context.exception), "Given distribution doesn't sum to 1.")

#%% health.icer
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_health_icer(unittest.TestCase):
    r"""
    Tests the health.icer function with the following cases:
        nominal
        no domination
        reverse order
        single scalar input
        list inputs
        bad values (should error)
        bad sizes (should error)
    """

    def setUp(self) -> None:
        self.cost     = np.array([250e3, 750e3, 2.25e6, 3.75e6])
        self.qaly     = np.array([20., 30, 40, 80])
        self.inc_cost = np.array([250e3, 500e3, 3e6])
        self.inc_qaly = np.array([20., 10, 50])
        self.icer_out = np.array([12500., 50000, 60000])
        self.order    = np.array([0., 1, np.nan, 2])
        self.fig: Optional[plt.Figure] = None

    def test_slide_example(self) -> None:
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost, self.qaly)
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_no_domination(self) -> None:
        ix = [0, 1, 3]
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost[ix], self.qaly[ix])
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order[ix], 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_reverse_order(self) -> None:
        ix = [3, 2, 1, 0]
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost[ix], self.qaly[ix])
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order[ix], 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_single_input(self) -> None:
        ix = 0
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost[ix], self.qaly[ix])
        np.testing.assert_array_equal(inc_cost, self.inc_cost[ix], 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly[ix], 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out[ix], 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order[ix], 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_list_inputs(self) -> None:
        cost = [this_cost for this_cost in self.cost]
        qaly = [this_cost for this_cost in self.qaly]
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(cost, qaly)
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_baseline1(self) -> None:
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost, self.qaly, baseline=0)
        temp = self.inc_cost
        temp[0] = 0
        np.testing.assert_array_equal(inc_cost, temp, 'Incremental cost mismatch.')
        temp = self.inc_qaly
        temp[0] = 0
        np.testing.assert_array_equal(inc_qaly, temp, 'Incremental QALY mismatch.')
        temp = self.icer_out
        temp[0] = np.nan
        np.testing.assert_array_equal(icer_out, temp, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_baseline2(self) -> None:
        # TODO: need to verify this case independently
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost, self.qaly, baseline=1)
        temp = self.inc_cost
        temp[0] = -self.inc_cost[1]
        np.testing.assert_array_equal(inc_cost, temp, 'Incremental cost mismatch.')
        temp = self.inc_qaly
        temp[0] = -self.inc_qaly[1]
        np.testing.assert_array_equal(inc_qaly, temp, 'Incremental QALY mismatch.')
        temp = self.icer_out
        temp[0] = self.icer_out[1]
        np.testing.assert_array_equal(icer_out, temp, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_names(self) -> None:
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost, self.qaly, \
            names=['Name 1', 'Name 2', 'Another name', 'Final Name'])
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_bad_values(self) -> None:
        with self.assertRaises(AssertionError):
            health.icer([1, -2, 3], [4, 5, 6])
        with self.assertRaises(AssertionError):
            health.icer([1, 2, 3], [4, -5, 6])

    def test_bad_input_sizes(self) -> None:
        with self.assertRaises(AssertionError):
            health.icer([], [])
        with self.assertRaises(AssertionError):
            health.icer([1, 2, 3], [4, 5])

    def test_all_dominated_by_last(self) -> None:
        cost = np.array([10, 20, 30, 1])
        qaly = np.array([1, 2, 3, 100])
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(cost, qaly)
        np.testing.assert_array_equal(inc_cost, 1, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, 100, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, 0.01, 'ICER mismatch.')
        np.testing.assert_array_equal(order, np.array([np.nan, np.nan, np.nan, 0]), 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_plot1(self) -> None:
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost, self.qaly, make_plot=True)
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(isinstance(self.fig, plt.Figure))

    def test_plot2(self) -> None:
        opts = Opts()
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = health.icer(self.cost, self.qaly, \
            make_plot=True, opts=opts, baseline=0)
        temp = self.inc_cost
        temp[0] = 0
        np.testing.assert_array_equal(inc_cost, temp, 'Incremental cost mismatch.')
        temp = self.inc_qaly
        temp[0] = 0
        np.testing.assert_array_equal(inc_qaly, temp, 'Incremental QALY mismatch.')
        temp = self.icer_out
        temp[0] = np.nan
        np.testing.assert_array_equal(icer_out, temp, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(isinstance(self.fig, plt.Figure))

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    if HAVE_MPL:
        plt.ioff()
    unittest.main(exit=False)
