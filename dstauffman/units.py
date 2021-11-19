r"""
Contains utilities and functions dealing with unit conversions.

Notes
-----
#.  Written by David C. Stauffer in February 2016.
"""

#%% Imports
import doctest
from typing import Tuple, Union
import unittest

from math import pi

#%% Constants - Unit Conversions
# Time
ONE_MINUTE: int = 60
ONE_HOUR: int = 3600
ONE_DAY: int = 86400
MONTHS_PER_YEAR: int = 12

# Angle
RAD2DEG: float = 180.0 / pi
DEG2RAD: float = pi / 180.0

ARCSEC2RAD: float = 1.0 / ONE_HOUR * DEG2RAD
RAD2ARCSEC: float = ONE_HOUR * RAD2DEG

# Length
FT2M: float = 0.3048
M2FT: float = 1 / 0.3048
IN2CM: float = 2.54
CM2IN: float = 1 / 2.54

# Symbols
DEGREE_SIGN: str = u'\N{DEGREE SIGN}'  # degree sign, also u'\u00b0' ° or chr(176)
MICRO_SIGN: str = u'\N{MICRO SIGN}'  # micro sign, also u'\u00b5' μ or chr(181), note this is different than chr(956)

#%% get_factors
def get_factors(prefix: Union[str, int, float], inverse: bool = False) -> Tuple[float, str]:
    r"""
    Get the multiplication factor and unit label for the desired units.

    Parameters
    ----------
    prefix : str
        Unit standard metric prefix, from:
            {'yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega',
             'kilo', 'hecto', 'deca', 'unity', 'deci', 'centi', 'milli',
             'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto',
             'arcminute', 'arcsecond', 'milliarcsecond', 'microarcsecond',
             'percentage', 'arcsecond^2'}
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
    >>> (mult, label) = get_factors('micro')
    >>> print(mult)
    1e-06

    >>> print(label)
    µ

    """
    # find the desired units and label prefix
    if prefix == 'yotta':
        mult  = 1e24 if not inverse else 1e-24
        label = 'Y'
    elif prefix == 'zetta':
        mult  = 1e21 if not inverse else 1e-21
        label = 'Z'
    elif prefix == 'exa':
        mult  = 1e18 if not inverse else 1e-18
        label = 'E'
    elif prefix == 'peta':
        mult  = 1e15 if not inverse else 1e-15
        label = 'P'
    elif prefix == 'tera':
        mult  = 1e12 if not inverse else 1e-12
        label = 'T'
    elif prefix == 'giga':
        mult  = 1e9 if not inverse else 1e-9
        label = 'G'
    elif prefix == 'mega':
        mult  = 1e6 if not inverse else 1e-6
        label = 'M'
    elif prefix == 'kilo':
        mult  = 1e3 if not inverse else 1e-3
        label = 'k'
    elif prefix == 'hecto':
        mult  = 1e2 if not inverse else 1e-2
        label = 'h'
    elif prefix == 'deca':
        mult  = 1e1 if not inverse else 1e-1
        label = 'da'
    elif prefix == 'unity':
        mult  = 1.0
        label = ''
    elif prefix == 'deci':
        mult  = 1e-1 if not inverse else 1e1
        label = 'd'
    elif prefix == 'centi':
        mult  = 1e-2 if not inverse else 1e2
        label = 'c'
    elif prefix == 'milli':
        mult  = 1e-3 if not inverse else 1e3
        label = 'm'
    elif prefix == 'micro':
        mult  = 1e-6 if not inverse else 1e6
        label = MICRO_SIGN
    elif prefix == 'nano':
        mult  = 1e-9 if not inverse else 1e9
        label = 'n'
    elif prefix == 'pico':
        mult  = 1e-12 if not inverse else 1e12
        label = 'p'
    elif prefix == 'femto':
        mult  = 1e-15 if not inverse else 1e15
        label = 'f'
    elif prefix == 'atto':
        mult  = 1e-18 if not inverse else 1e18
        label = 'a'
    elif prefix == 'zepto':
        mult  = 1e-21 if not inverse else 1e21
        label = 'z'
    elif prefix == 'yocto':
        mult  = 1e-24 if not inverse else 1e24
        label = 'y'
    # Special cases
    elif prefix == 'percentage':
        mult  = 0.01 if not inverse else 100.0
        label = '%'
    # below follow some stupid english units for rotation angles (try to never use them!)
    elif prefix == 'arcminute':
        mult  = 1.0 / ONE_MINUTE * DEG2RAD if not inverse else ONE_MINUTE / DEG2RAD
        label = 'amin'
    elif prefix == 'arcsecond':
        mult  = ARCSEC2RAD if not inverse else RAD2ARCSEC
        label = 'asec'
    elif prefix == 'arcsecond^2':
        mult  = ARCSEC2RAD ** 2 if not inverse else RAD2ARCSEC ** 2
        label = 'asec^2'
    elif prefix == 'milliarcsecond':
        mult  = 1e3 * ARCSEC2RAD if not inverse else 1e-3 * RAD2ARCSEC
        label = 'mas'
    elif prefix == 'microarcsecond':
        mult  = 1e6 * ARCSEC2RAD if not inverse else 1e-6 * RAD2ARCSEC
        label = MICRO_SIGN + 'as'
    else:
        raise ValueError('Unexpected value for units prefix.')
    return (mult, label)


#%% Functions - get_time_factor
def get_time_factor(unit: str) -> int:
    r"""
    Gets the time factor for the given unit relative to the base SI unit of 'sec'.

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
    >>> mult = get_time_factor('hr')
    >>> print(mult)
    3600

    """
    if unit == 'sec':
        mult = 1
    elif unit == 'min':
        mult = ONE_MINUTE
    elif unit == 'hr':
        mult = ONE_HOUR
    elif unit == 'day':
        mult = ONE_DAY
    else:
        raise ValueError(f'Unexpected value for "{unit}".')
    return mult


#%% Functions - get_unit_conversion
def get_unit_conversion(conversion: Union[None, str, int, float, Tuple[str, float]], units: str = '') -> Tuple[str, float]:
    r"""
    Acts as a wrapper to unit conversions for legends in plots and for scaling second axes.

    Parameters
    ----------
    conversion : str
        Unit standard metric prefix or some special cases, from:
            {'yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega',
             'kilo', 'hecto', 'deca', 'unity', 'deci', 'centi', 'milli',
             'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto',
             'arcminute', 'arcsecond', 'milliarcsecond', 'microarcsecond',
             'percentage', 'arcsecond^2'}
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
    >>> conversion = 'micro'
    >>> units = 'rad'
    >>> (new_units, unit_mult) = get_unit_conversion(conversion, units)
    >>> print(unit_mult)
    1000000.0

    >>> print(new_units)
    µrad

    """
    if conversion is None:
        return ('', 1)
    if isinstance(conversion, (int, float)):
        return ('', conversion)
    if not isinstance(conversion, str):
        return (conversion[0], conversion[1])
    if conversion == 'percentage':
        return ('%', 100)
    (unit_mult, label) = get_factors(conversion, inverse=True)
    if units in {'', 'rad', 'rad^2'} and 'arc' in conversion:
        new_units = label
    elif units == 'rad^2':
        new_units = '(' + label + 'rad)^2'
        unit_mult **= 2
    elif not units or units == 'unitless':
        # special empty cases
        if conversion == 'milli':
            new_units = 'ppk'
        elif conversion == 'micro':
            new_units = 'ppm'
        elif conversion == 'nano':
            new_units = 'ppb'
        elif conversion == 'pico':
            new_units = 'ppt'
        elif label:
            raise ValueError("The unit conversion given doesn't work for empty units.")
        else:
            new_units = label + units
    else:
        new_units = label + units
    return (new_units, unit_mult)


#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_units', exit=False)
    doctest.testmod(verbose=False)
