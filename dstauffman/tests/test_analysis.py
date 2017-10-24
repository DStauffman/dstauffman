# -*- coding: utf-8 -*-
r"""
Test file for the `analysis` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in October 2017.
"""

#%% Imports
import unittest

import matplotlib.pyplot as plt
import numpy as np

import dstauffman as dcs

#%% Plotter for testing
plotter = dcs.Plotter(False)

#%% icer
class Test_icer(unittest.TestCase):
    r"""
    Tests the icer function with the following cases:
        nominal
        no domination
        reverse order
        single scalar input
        list inputs
        bad values (should error)
        bad sizes (should error)
    """

    def setUp(self):
        self.cost     = np.array([250e3, 750e3, 2.25e6, 3.75e6])
        self.qaly     = np.array([20., 30, 40, 80])
        self.inc_cost = np.array([250e3, 500e3, 3e6])
        self.inc_qaly = np.array([20., 10, 50])
        self.icer_out = np.array([12500., 50000, 60000])
        self.order    = np.array([0., 1, np.nan, 2])
        self.fig      = None

    def test_slide_example(self):
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost, self.qaly)
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_no_domination(self):
        ix = [0, 1, 3]
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost[ix], self.qaly[ix])
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order[ix], 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_reverse_order(self):
        ix = [3, 2, 1, 0]
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost[ix], self.qaly[ix])
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order[ix], 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_single_input(self):
        ix = 0
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost[ix], self.qaly[ix])
        np.testing.assert_array_equal(inc_cost, self.inc_cost[ix], 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly[ix], 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out[ix], 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order[ix], 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_list_inputs(self):
        cost = [this_cost for this_cost in self.cost]
        qaly = [this_cost for this_cost in self.qaly]
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(cost, qaly)
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_baseline1(self):
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost, self.qaly, baseline=0)
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

    def test_baseline2(self):
        # TODO: need to verify this case independently
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost, self.qaly, baseline=1)
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

    def test_names(self):
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost, self.qaly, \
            names=['Name 1', 'Name 2', 'Another name', 'Final Name'])
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_bad_values(self):
        with self.assertRaises(AssertionError):
            dcs.icer([1, -2, 3], [4, 5, 6])
        with self.assertRaises(AssertionError):
            dcs.icer([1, 2, 3], [4, -5, 6])

    def test_bad_input_sizes(self):
        with self.assertRaises(AssertionError):
            dcs.icer([], [])
        with self.assertRaises(AssertionError):
            dcs.icer([1, 2, 3], [4, 5])

    def test_all_dominated_by_last(self):
        cost = np.array([10, 20, 30, 1])
        qaly = np.array([1, 2, 3, 100])
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(cost, qaly)
        np.testing.assert_array_equal(inc_cost, 1, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, 100, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, 0.01, 'ICER mismatch.')
        np.testing.assert_array_equal(order, np.array([np.nan, np.nan, np.nan, 0]), 'Order mismatch.')
        self.assertTrue(self.fig is None)

    def test_plot1(self):
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost, self.qaly, make_plot=True)
        np.testing.assert_array_equal(inc_cost, self.inc_cost, 'Incremental cost mismatch.')
        np.testing.assert_array_equal(inc_qaly, self.inc_qaly, 'Incremental QALY mismatch.')
        np.testing.assert_array_equal(icer_out, self.icer_out, 'ICER mismatch.')
        np.testing.assert_array_equal(order, self.order, 'Order mismatch.')
        self.assertTrue(isinstance(self.fig, plt.Figure))

    def test_plot2(self):
        opts = dcs.Opts()
        (inc_cost, inc_qaly, icer_out, order, icer_data, self.fig) = dcs.icer(self.cost, self.qaly, \
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

    def tearDown(self):
        if self.fig is not None:
            plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    plt.ioff()
    unittest.main(exit=False)
