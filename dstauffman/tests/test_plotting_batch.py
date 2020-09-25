r"""
Test file for the `batch` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""

#%% Imports
import unittest

import numpy as np

from dstauffman import capture_output
from dstauffman.estimation import BpeResults
import dstauffman.plotting as plot

#%% Hard-coded values
plotter = plot.Plotter(False)

#%% plotting.plot_bpe_convergence
class Test_plotting_plot_bpe_convergence(unittest.TestCase):
    r"""
    Tests the plotting.plot_bpe_convergence function with the following cases:
        Nominal
        Only two costs
        No Opts
        No Costs
    """
    def setUp(self):
        self.costs = np.array([1, 0.1, 0.05, 0.01])
        self.opts = plot.Opts()
        self.opts.show_plot = False
        self.figs = []

    def test_nominal(self):
        self.figs.append(plot.plot_bpe_convergence(self.costs, opts=self.opts))

    def test_only_two_costs(self):
        self.figs.append(plot.plot_bpe_convergence(self.costs[np.array([0, 3])], opts=self.opts))

    def test_no_opts(self):
        self.figs.append(plot.plot_bpe_convergence(self.costs))

    def test_no_costs(self):
        self.figs.append(plot.plot_bpe_convergence([], opts=self.opts))

    def tearDown(self):
        plot.close_all(self.figs)

#%% plotting.plot_bpe_results
class Test_plotting_plot_bpe_results(unittest.TestCase):
    r"""
    Tests the plotting.plot_bpe_results function with the following cases:
        TBD
    """
    def setUp(self):
        self.figs = []
        self.bpe_results = BpeResults()
        self.opts = plot.Opts()
        self.plots = {'innovs': True, 'convergence': True, 'correlation': True, 'info_svd': True, \
        'covariance': True}

    def test_nominal(self):
        # add data
        names = ['a', 'b', 'c', 'd']
        matrix = np.random.rand(4, 4)
        self.bpe_results.param_names  = [x.encode('utf-8') for x in names]
        self.bpe_results.begin_innovs = np.array([1, 2, 3, 4], dtype=float)
        self.bpe_results.final_innovs = np.array([0.5, 0.25, 0.1, 0.05])
        self.bpe_results.costs        = np.array([1, 0.1, 0.05, 0.01])
        self.bpe_results.correlation  = matrix.copy()
        self.bpe_results.info_svd     = matrix.copy()
        self.bpe_results.covariance   = matrix.copy()
        self.figs = plot.plot_bpe_results(self.bpe_results, plots=self.plots)

    def test_nodata(self):
        with capture_output() as out:
            self.figs = plot.plot_bpe_results(self.bpe_results, plots=self.plots)
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], "Data isn't available for Innovations plot.")
        self.assertEqual(lines[1], "Data isn't available for convergence plot.")
        self.assertEqual(lines[2], "Data isn't available for correlation plot.")
        self.assertEqual(lines[3], "Data isn't available for information SVD plot.")
        self.assertEqual(lines[4], "Data isn't available for covariance plot.")

    def test_no_plots(self):
        plot.plot_bpe_results(self.bpe_results, opts=self.opts)

    def test_bad_plot(self):
        with self.assertRaises(ValueError):
            plot.plot_bpe_results(self.bpe_results, plots={'bad_key': False})

    def test_only_one_key(self):
        plot.plot_bpe_results(self.bpe_results, plots={'innovs': False})

    def tearDown(self):
        plot.close_all(self.figs)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
