r"""
Test file for the `batch` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in May 2015.

"""

# %% Imports
from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

from slog import capture_output

from dstauffman import HAVE_MPL, HAVE_NUMPY
from dstauffman.estimation import BpeResults
import dstauffman.plotting as plot

if HAVE_MPL:
    from matplotlib.figure import Figure
if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _N = NDArray[np.floating]


# %% plotting.plot_bpe_convergence
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_plot_bpe_convergence(unittest.TestCase):
    r"""
    Tests the plotting.plot_bpe_convergence function with the following cases:
        Nominal
        Only two costs
        No Opts
        No Costs
    """

    def setUp(self) -> None:
        self.costs: _N | list[float] = np.array([1, 0.1, 0.05, 0.01]) if HAVE_NUMPY else [1.0, 0.1, 0.05, 0.01]
        self.opts = plot.Opts()
        self.opts.show_plot = False
        self.figs: list[Figure] = []

    def test_nominal(self) -> None:
        self.figs.append(plot.plot_bpe_convergence(self.costs, opts=self.opts))  # type: ignore[arg-type]

    def test_only_two_costs(self) -> None:
        costs: list[float] = [self.costs[i] for i in [0, 3]]
        self.figs.append(plot.plot_bpe_convergence(costs, opts=self.opts))

    def test_no_opts(self) -> None:
        self.figs.append(plot.plot_bpe_convergence(self.costs))  # type: ignore[arg-type]

    def test_no_costs(self) -> None:
        self.figs.append(plot.plot_bpe_convergence([], opts=self.opts))

    def tearDown(self) -> None:
        plot.close_all(self.figs)


# %% plotting.plot_bpe_results
class Test_plotting_plot_bpe_results(unittest.TestCase):
    r"""
    Tests the plotting.plot_bpe_results function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.figs: list[Figure] = []
        self.bpe_results = BpeResults()
        self.opts = plot.Opts()
        self.plots = {"innovs": True, "convergence": True, "correlation": True, "info_svd": True, "covariance": True}

    @unittest.skipIf(not HAVE_MPL or not HAVE_NUMPY, "Skipping due to missing matplotlib/numpy dependency.")
    def test_nominal(self) -> None:
        # add data
        names = ["a", "b", "c", "d"]
        matrix = np.random.default_rng().random((4, 4))
        # fmt: off
        self.bpe_results.param_names  = [x.encode("utf-8") for x in names]
        self.bpe_results.begin_innovs = np.array([1.0, 2.0, 3.0, 4.0])
        self.bpe_results.final_innovs = np.array([0.5, 0.25, 0.1, 0.05])
        self.bpe_results.costs        = [1.0, 0.1, 0.05, 0.01]
        self.bpe_results.correlation  = matrix.copy()
        self.bpe_results.info_svd     = matrix.copy()
        self.bpe_results.covariance   = matrix.copy()
        # fmt: on
        self.figs = plot.plot_bpe_results(self.bpe_results, plots=self.plots)

    def test_nodata(self) -> None:
        with capture_output() as ctx:
            self.figs = plot.plot_bpe_results(self.bpe_results, plots=self.plots)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "Data isn't available for Innovations plot.")
        self.assertEqual(lines[1], "Data isn't available for convergence plot.")
        self.assertEqual(lines[2], "Data isn't available for correlation plot.")
        self.assertEqual(lines[3], "Data isn't available for information SVD plot.")
        self.assertEqual(lines[4], "Data isn't available for covariance plot.")

    def test_no_plots(self) -> None:
        plot.plot_bpe_results(self.bpe_results, opts=self.opts)

    def test_bad_plot(self) -> None:
        with self.assertRaises(ValueError):
            plot.plot_bpe_results(self.bpe_results, plots={"bad_key": False})

    @unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
    def test_only_one_key(self) -> None:
        plot.plot_bpe_results(self.bpe_results, plots={"innovs": False})

    def tearDown(self) -> None:
        if HAVE_MPL:
            plot.close_all(self.figs)


# %% Unit test execution
if __name__ == "__main__":
    plot.suppress_plots()
    unittest.main(exit=False)
