r"""
Example script for plotting data from many similar runs, such as in a Monte Carlo simulation.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""  # pylint: disable=redefined-outer-name

# %% Imports
import datetime

import numpy as np

import dstauffman as dcs
import dstauffman.plotting as plot


# %% Functions - build_months
def build_months(date_zero: datetime.datetime, num_months: int = 12) -> list[datetime.datetime]:
    """Build a list of all the months starting on day one for the given month."""
    out = [date_zero]
    temp = date_zero
    for _ in range(1, num_months):
        temp += datetime.timedelta(days=32)
        temp = temp.replace(day=1)
        out.append(temp)
    return out


# %% Main function
if __name__ == "__main__":
    # %% Create some fake data
    break1 = 5.0  # years
    coeff1 = np.array([0.1, 0.0])
    coeff2 = np.array([0.15, -0.25])
    dt = 1.0 / 12.0

    num_months = 12 * 12 + 1
    num_cycles = 100
    date_zero = 2023.0

    time = np.arange(0, num_months) / 12.0
    dates = build_months(datetime.datetime(np.round(date_zero).astype(int), 1, 1), 12 * 12 + 1)
    np_dates = dcs.convert_datetime_to_np(dates)
    trend1 = np.polyval(coeff1, time[time <= break1])
    trend2 = np.polyval(coeff2, time[time > break1])
    num1 = trend1.size
    num2 = trend2.size
    data = np.hstack([trend1, trend2])

    all_data = np.empty((num_months, num_cycles))
    for i in range(num_cycles):
        noise1 = 0.02 * np.cumsum(np.linspace(0, 1, num1) * (np.random.rand(num1) - 0.5))
        noise2 = noise1[-1] + 0.1 * np.cumsum(np.linspace(0, 1, num2) * (np.random.rand(num2) - 0.5))
        all_data[:, i] = data + np.hstack([noise1, noise2])

    # %% Create opts
    opts = plot.Opts().convert_dates("numpy")

    # %% Create the plots
    fig1 = plot.plot_time_history("Results vs Time", np_dates, all_data.T, opts=opts)

    fig2 = plot.plot_health_monte_carlo(time, all_data, "Results vs Time", units="", opts=opts)
