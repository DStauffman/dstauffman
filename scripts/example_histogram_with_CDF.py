"""Example to plot a PDF histogram with a CDF line overlaid."""

# %% Imports
import numpy as np

from dstauffman.plotting import close_all, plot_histogram

# %% Script
if __name__ == "__main__":
    close_all()
    description = "Histogram"
    data = np.abs(np.random.default_rng().normal(size=1000) / 3.0)
    bins = np.arange(0.0, 2.1, 0.1)
    if np.max(data) > 2.0:
        bins = np.hstack([bins, np.max(data) + 0.1])
    cdf_x = 0.75
    cdf_y = 0.5

    # bins2 = np.array([0.0, 0.1, 0.5, 1.0, 2.0, np.inf])

    plot_histogram(
        description,
        data,
        bins,
        skip_setup_plots=True,
        show_cdf=True,
        cdf_x=cdf_x,
        cdf_y=cdf_y,
        cdf_round_to_bin=False,
        normalize_spacing=False,
    )
