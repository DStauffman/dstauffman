"""Script to demonstrate the different interpolation options."""

import numpy as np

import dstauffman as dcs
import dstauffman.plotting as plot

if __name__ == "__main__":
    xp = np.array([0.0, 111.0, 2000.0, 5000.0])
    yp = np.array([0.0, 1.0, -2.0, 3.0])
    x  = np.arange(0, 6001, dtype=float)  # fmt: skip
    y1 = dcs.zero_order_hold(x, xp, yp)
    y2 = dcs.linear_interp(x, xp, yp, extrapolate=True)
    y3 = dcs.linear_lowpass_interp(x, xp, yp, filt_order=2, filt_freq=0.04, filt_samp=1.0, extrapolate=True)

    plot.plot_time_history("Interpolations", x, [y1, y2, y3])
