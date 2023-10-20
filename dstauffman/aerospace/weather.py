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
from typing import Union
import unittest

from dstauffman import HAVE_NUMPY, HAVE_PANDAS, NP_NAT

if HAVE_NUMPY:
    import numpy as np
if HAVE_PANDAS:
    import pandas as pd


# %% Functions - read_tci_data
def read_tci_data(filename: Path) -> pd.DataFrame:
    r"""Reads the TCI data from files provided online."""

    def _convert_m_d_y(text: str) -> datetime.datetime:
        """Convert a m-d-y string when each are floats (yet still whole integers)."""
        (m, d, y) = text.split("-")
        return datetime.datetime(int(float(y)), int(float(m)), int(float(d)))

    names = ["Date", "TCI"]
    converters = {"Date": _convert_m_d_y}
    df = pd.read_table(filename, delim_whitespace=True, names=names, converters=converters)
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
    df = pd.read_table(filename, names=names, delim_whitespace=True, comment="#")
    return df


# %% Functions - load_solar_cycles
def read_solar_cycles(filename: Path) -> pd.DataFrame:
    """Read the solar cycle history."""

    def _get_duration(text: str) -> Union[int, float]:
        if not text:
            return np.nan
        (y, m) = text.split("-")
        return 12 * int(y) + int(m)

    def _get_solar_cycle(text: str) -> int:
        return int(text.split("Solar cycle ")[-1])

    def _get_date(text: str) -> Union[datetime.datetime, np.datetime64]:
        if not text:
            return NP_NAT
        return datetime.datetime.strptime(text.strip(), "%Y-%m")

    def _remove_parens(text: str) -> Union[int, float]:
        if not text:
            return np.nan
        return int(text.replace("(", "").replace(")", ""))

    def _parse_ongoing(text: str) -> Union[int, float]:
        if not text:
            return np.nan
        try:
            return int(text)
        except ValueError:
            pass
        if text.startswith("Progressive: "):
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
        "100+ spots days",
        "150+ spots days",
        "200+ spots days",
    ]
    converters = {
        "Solar_Cycle": _get_solar_cycle,
        "Start": _get_date,
        "Maximum": _get_date,
        "Time_of_Rise_mons": _get_duration,
        "Duration_mons": _get_duration,
        "Smoothed_max_SSN": _parse_ongoing,
        "100+ spots days": _parse_ongoing,
        "150+ spots days": _parse_ongoing,
        "200+ spots days": _parse_ongoing,
        "Average_spots_per_day": _remove_parens,
    }
    df = pd.read_table(filename, sep="\t", skiprows=1, names=names, converters=converters)
    return df


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_weather", exit=False)
    doctest.testmod(verbose=False)
