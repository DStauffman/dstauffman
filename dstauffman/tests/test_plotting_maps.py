r"""
Test file for the `maps` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in December 2024.
"""

# %% Imports
from __future__ import annotations

import unittest

from dstauffman import HAVE_MPL, HAVE_NUMPY
import dstauffman.plotting as plot

if HAVE_MPL:
    from matplotlib.figure import Figure


# %% plotting.get_map_data
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_plotting_get_map_data(unittest.TestCase):
    r"""
    Tests the plotting.get_map_data function with the following cases:
        Nominal
    """

    def test_nominal(self) -> None:
        map_data, map_labels, map_colors = plot.get_map_data()
        exp = ["Afghanistan", "Albania", "Algeria", "Angola", "Antarctica"]
        self.assertEqual(sorted(map_data.keys())[:5], exp)


# %% plotting.plot_map
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_plot_map(unittest.TestCase):
    r"""
    Tests the plotting.plot_map function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.fig: Figure | None = None

    def test_nominal(self) -> None:
        # simple Earth plot
        self.fig = plot.plot_map()

    # TODO: write more of these (steal from example script)

    def tearDown(self) -> None:
        if HAVE_MPL:
            plot.close_all(self.fig)


# %% Unit test execution
if __name__ == "__main__":
    plot.suppress_plots()
    unittest.main(exit=False)
