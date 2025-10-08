r"""
Contains utilities to load Space Weather data.

Notes
-----
#.  Written by David C. Stauffer in October 2023.

"""

# %% Imports
from __future__ import annotations

import datetime
import doctest
from pathlib import Path
import unittest

from dstauffman import HAVE_NUMPY, HAVE_PANDAS, NP_NAT

if HAVE_NUMPY:
    import numpy as np
if HAVE_PANDAS:
    import pandas as pd


# %% Constants
AP_TO_KP: dict[int, float] = {}
AP_TO_KP[0] = 0.000  # "0o"
AP_TO_KP[2] = 0.333  # "0+"
AP_TO_KP[3] = 0.667  # "1-"
AP_TO_KP[4] = 1.000  # "1o"
AP_TO_KP[5] = 1.333  # "1+"
AP_TO_KP[6] = 1.667  # "2-"
AP_TO_KP[7] = 2.000  # "2o"
AP_TO_KP[9] = 2.333  # "2+"
AP_TO_KP[12] = 2.667  # "3-"
AP_TO_KP[15] = 3.000  # "3o"
AP_TO_KP[18] = 3.333  # "3+"
AP_TO_KP[22] = 3.667  # "4-"
AP_TO_KP[27] = 4.000  # "4o"
AP_TO_KP[32] = 4.333  # "4+"
AP_TO_KP[39] = 4.667  # "5-"
AP_TO_KP[48] = 5.000  # "5o"
AP_TO_KP[56] = 5.333  # "5+"
AP_TO_KP[67] = 5.667  # "6-"
AP_TO_KP[80] = 6.000  # "6o"
AP_TO_KP[94] = 6.333  # "6+"
AP_TO_KP[111] = 6.667  # "7-"
AP_TO_KP[132] = 7.000  # "7o"
AP_TO_KP[154] = 7.333  # "7+"
AP_TO_KP[179] = 7.667  # "8-"
AP_TO_KP[207] = 8.000  # "8o"
AP_TO_KP[236] = 8.333  # "8+"
AP_TO_KP[300] = 8.667  # "9-"
AP_TO_KP[400] = 9.000  # "9o"


# %% Functions - read_tci_data
def read_tci_data(filename: Path) -> pd.DataFrame:
    r"""Reads the TCI data from files provided online."""

    def _convert_m_d_y(text: str) -> datetime.datetime:
        """Convert a m-d-y string when each are floats (yet still whole integers)."""
        (m, d, y) = text.split("-")
        return datetime.datetime(int(float(y)), int(float(m)), int(float(d)))

    names = ["Date", "TCI"]
    converters = {"Date": _convert_m_d_y}
    df = pd.read_table(filename, sep=r"\s+", names=names, converters=converters)
    return df


# %% Functions - read_kp_ap_etc_data
def read_kp_ap_etc_data(filename: Path) -> pd.DataFrame:
    """Read the Kp data and all the rest of it. Everything is a simple float or int with spaces inbetween."""
    # fmt: off
    names = [
        "YYY", "MM", "DD", "days", "days_m", "Bsr", "dB", "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
        "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8", "Ap", "SN", "F10.7obs", "F10.7adj", "D",
    ]
    # fmt: on
    df = pd.read_table(filename, names=names, sep=r"\s+", comment="#")
    # convert year-month-day to a GMT value
    df.rename(columns={"YYY": "year", "MM": "month", "DD": "day"}, inplace=True)  # noqa: PD002
    df.insert(0, "GMT", pd.to_datetime(df[["year", "month", "day"]]))
    return df


# %% Functions - read_kp_ap_nowcast
def read_kp_ap_nowcast(filename: Path) -> pd.DataFrame:
    """Read the every 3 hour Kp and Ap data."""
    names = ["YYY", "MM", "DD", "hour_start", "hour_middle", "days", "days_m", "Kp", "ap", "D"]
    df = pd.read_table(filename, names=names, sep=r"\s+", comment="#")
    # convert year-moth-day to a GMT value (including hour)
    df.rename(columns={"YYY": "year", "MM": "month", "DD": "day"}, inplace=True)  # noqa: PD002
    df.insert(0, "GMT", pd.to_datetime(df[["year", "month", "day"]]) + pd.to_timedelta(df.hour_start, "hour"))
    return df


# %% Functions - load_solar_cycles
def read_solar_cycles(filename: Path) -> pd.DataFrame:
    """Read the solar cycle history."""

    def _get_duration(text: str) -> int | float:
        if not text:
            return np.nan
        (y, m) = text.split("-")
        return 12 * int(y) + int(m)

    def _get_solar_cycle(text: str) -> int:
        return int(text.split("Solar cycle ")[-1])

    def _get_date(text: str) -> datetime.datetime | np.datetime64:
        if not text:
            return NP_NAT
        if text.startswith("("):
            return NP_NAT
        return datetime.datetime.strptime(text.strip(), "%Y-%m")  # noqa: DTZ007

    def _remove_parens(text: str) -> int | float:
        if not text:
            return np.nan
        if text.startswith("Progr"):
            return np.nan
        return int(text.replace("(", "").replace(")", ""))

    def _parse_ongoing(text: str) -> int | float:
        if not text:
            return np.nan
        try:
            return int(text)
        except ValueError:
            pass
        if text.startswith("Progr: "):
            return int(text.split(" ")[1].replace("*", ""))
        return int(text.split(" ")[0])

    names = [
        "Solar_Cycle",
        "Start",
        "Smoothed_min_SSN",
        "Maximum",
        "Smoothed_max_SSN",
        "Average_spots_per_day",
        "Time_of_Rise_mons",
        "Duration_mons",
    ]
    converters = {
        "Solar_Cycle": _get_solar_cycle,
        "Start": _get_date,
        "Maximum": _get_date,
        "Time_of_Rise_mons": _get_duration,
        "Duration_mons": _get_duration,
        "Smoothed_max_SSN": _parse_ongoing,
        "Average_spots_per_day": _remove_parens,
    }
    df = pd.read_table(filename, sep="\t", skiprows=1, names=names, converters=converters)
    return df


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_weather", exit=False)
    doctest.testmod(verbose=False)
