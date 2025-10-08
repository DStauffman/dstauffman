r"""
Contains utilities and functions dealing with unit conversions.

Notes
-----
#.  Written by David C. Stauffer in February 2016.

"""

# %% Imports
from __future__ import annotations

import doctest
from math import pi
from typing import Final, TYPE_CHECKING
import unittest

if TYPE_CHECKING:
    import numpy as np

    _nf = np.floating

# %% Constants - Unit Conversions
# Time
ONE_MINUTE: Final = 60
ONE_HOUR: Final = 3600
ONE_DAY: Final = 86400
MONTHS_PER_YEAR: Final = 12

# Angle
RAD2DEG: Final = 180.0 / pi
DEG2RAD: Final = pi / 180.0

ARCSEC2RAD: Final = 1.0 / ONE_HOUR * DEG2RAD
RAD2ARCSEC: Final = ONE_HOUR * RAD2DEG

# Length
FT2M: Final = 0.3048
M2FT: Final = 1 / 0.3048
IN2CM: Final = 2.54
CM2IN: Final = 1 / 2.54

# Symbols
DEGREE_SIGN: Final = "\N{DEGREE SIGN}"  # degree sign, also u"\u00b0" ° or chr(176)
MICRO_SIGN: Final = "\N{MICRO SIGN}"  # micro sign, also u"\u00b5" μ or chr(181), note this is different than chr(956)

# Prefixes, inverses, and labels
_PREFIXES: dict[str, tuple[float, float, str]] = {
    "yotta": (1e24, 1e-24, "Y"),
    "zetta": (1e21, 1e-21, "Z"),
    "exa": (1e18, 1e-18, "E"),
    "peta": (1e15, 1e-15, "P"),
    "tera": (1e12, 1e-12, "T"),
    "giga": (1e9, 1e-9, "G"),
    "mega": (1e6, 1e-6, "M"),
    "kilo": (1e3, 1e-3, "k"),
    "hecto": (1e2, 1e-2, "h"),
    "deca": (1e1, 1e-1, "da"),
    "unity": (1.0, 1.0, ""),
    "deci": (1e-1, 1e1, "d"),
    "centi": (1e-2, 1e2, "c"),
    "milli": (1e-3, 1e3, "m"),
    "micro": (1e-6, 1e6, MICRO_SIGN),
    "nano": (1e-9, 1e9, "n"),
    "pico": (1e-12, 1e12, "p"),
    "femto": (1e-15, 1e15, "f"),
    "atto": (1e-18, 1e18, "a"),
    "zepto": (1e-21, 1e21, "z"),
    "yocto": (1e-24, 1e24, "y"),
    # Special cases
    "percentage": (0.01, 100.0, "%"),
    # below follow some stupid english units for rotation angles (try to never use them!)
    "arcminute": (1.0 / ONE_MINUTE * DEG2RAD, ONE_MINUTE / DEG2RAD, "amin"),
    "arcsecond": (ARCSEC2RAD, RAD2ARCSEC, "asec"),
    "arcsecond^2": (ARCSEC2RAD**2, RAD2ARCSEC**2, "asec^2"),
    "milliarcsecond": (1e3 * ARCSEC2RAD, 1e-3 * RAD2ARCSEC, "mas"),
    "microarcsecond": (1e6 * ARCSEC2RAD, 1e-6 * RAD2ARCSEC, MICRO_SIGN + "as"),
}


# %% get_factors
def get_factors(prefix: str, inverse: bool = False) -> tuple[float, str]:
    r"""
    Get the multiplication factor and unit label for the desired units.

    Parameters
    ----------
    prefix : str
        Unit standard metric prefix, from:
            {"yotta", "zetta", "exa", "peta", "tera", "giga", "mega",
             "kilo", "hecto", "deca", "unity", "deci", "centi", "milli",
             "micro", "nano", "pico", "femto", "atto", "zepto", "yocto",
             "arcminute", "arcsecond", "milliarcsecond", "microarcsecond",
             "percentage", "arcsecond^2"}
    inverse : bool, optional, default is False
        Whether to return the inverse version

    Returns
    -------
    mult : float
        Multiplication factor
    label : str
        Abbreviation for the prefix

    References
    ----------
    #.  http://en.wikipedia.org/wiki/Metric_prefix

    Notes
    -----
    #.  Updated by David C. Stauffer in March 2021 to provide an explicit inverse form to avoid
        needing to do a divide operation afterwards (which can have large round-off errors at the
        extreme conversion values).

    Examples
    --------
    >>> from dstauffman import get_factors
    >>> (mult, label) = get_factors("micro")
    >>> print(mult)
    1e-06

    >>> print(label)
    µ

    """
    # fmt: off
    if prefix not in _PREFIXES:
        raise ValueError("Unexpected value for units prefix.")
    (forward, backward, label) = _PREFIXES[prefix]
    mult = forward if not inverse else backward
    return (mult, label)


# %% Functions - get_time_factor
def get_time_factor(unit: str) -> int:
    r"""
    Gets the time factor for the given unit relative to the base SI unit of "sec".

    Parameters
    ----------
    unit : str
        Units to get the multiplying factor for

    Returns
    -------
    mult : int
        Multiplication factor

    Notes
    -----
    #.  Written by David C. Stauffer in June 2020.

    Examples
    --------
    >>> from dstauffman import get_time_factor
    >>> mult = get_time_factor("hr")
    >>> print(mult)
    3600

    """
    if unit == "sec":
        mult = 1
    elif unit == "min":
        mult = ONE_MINUTE
    elif unit == "hr":
        mult = ONE_HOUR
    elif unit == "day":
        mult = ONE_DAY
    else:
        raise ValueError(f'Unexpected value for "{unit}".')
    return mult


# %% Functions - get_unit_conversion
def get_unit_conversion(
    conversion: str | float | _nf | tuple[str, float] | tuple[str, _nf] | None, units: str = ""
) -> tuple[str, float]:
    r"""
    Acts as a wrapper to unit conversions for legends in plots and for scaling second axes.

    Parameters
    ----------
    conversion : str
        Unit standard metric prefix or some special cases, from:
            {"yotta", "zetta", "exa", "peta", "tera", "giga", "mega",
             "kilo", "hecto", "deca", "unity", "deci", "centi", "milli",
             "micro", "nano", "pico", "femto", "atto", "zepto", "yocto",
             "arcminute", "arcsecond", "milliarcsecond", "microarcsecond",
             "percentage", "arcsecond^2"}
    units : str
        label to apply the prefix to (sometimes replaced when dealing with radians and english units)

    Returns
    -------
    new_units : str
        Units with the correctly prepended abbreviation (or substitution)
    unit_mult : float
        Multiplication factor

    Notes
    -----
    #.  Written by David C. Stauffer in February 2021 when he had to deal with arcseconds and other
        special cases.
    #.  Special cases include dimensionless/radians to arcseconds or parts per million, or
        appropriately scaling radians squared.

    Examples
    --------
    >>> from dstauffman import get_unit_conversion
    >>> conversion = "micro"
    >>> units = "rad"
    >>> (new_units, unit_mult) = get_unit_conversion(conversion, units)
    >>> print(unit_mult)
    1000000.0

    >>> print(new_units)
    µrad

    """
    if conversion is None:
        return ("", 1)
    if isinstance(conversion, (int, float)):
        return ("", conversion)
    if not isinstance(conversion, str):
        assert isinstance(conversion, tuple) and len(conversion) == 2, "Expect a tuple with exactly two elements."  # noqa: PT018  # fmt: skip
        return (conversion[0], float(conversion[1]))
    if conversion == "percentage":
        return ("%", 100)
    (unit_mult, label) = get_factors(conversion, inverse=True)
    if units in {"", "rad", "rad^2"} and "arc" in conversion:
        new_units = label
    elif units == "rad^2":
        new_units = "(" + label + "rad)^2"
        unit_mult **= 2
    elif not units or units == "unitless":
        # special empty cases
        if conversion == "milli":
            new_units = "ppk"
        elif conversion == "micro":
            new_units = "ppm"
        elif conversion == "nano":
            new_units = "ppb"
        elif conversion == "pico":
            new_units = "ppt"
        elif label:
            raise ValueError("The unit conversion given doesn't work for empty units.")
        else:
            new_units = label + units
    else:
        new_units = label + units
    return (new_units, unit_mult)


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_units", exit=False)
    doctest.testmod(verbose=False)
