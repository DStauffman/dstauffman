# -*- coding: utf-8 -*-
r"""
Contains utilities and functions dealing with unit conversions.

Notes
-----
#.  By design, this module does not reference any other piece of the dstauffman code base except
        constants or enums to avoid circular references.
#.  Written by David C. Stauffer in February 2016.

"""

#%% Imports
import doctest
import unittest
from dstauffman.constants import MONTHS_PER_YEAR
from dstauffman.enums     import IntEnumPlus
from dstauffman.stats     import annual_rate_to_monthly_probability, convert_annual_to_monthly_probability, \
                                 prob_to_rate, rate_to_prob

#%% Units
class Units(IntEnumPlus):
    r"""Units class that can be used to keep track of the related units and convert when asked."""
    AR    = 1 # annual rate
    AP    = 2 # annual probability
    MP    = 3 # monthly probability
    ND    = 4 # nondimensional
    PER   = 5 # percentage
    P100K = 6 # per 100,000

    @classmethod
    def convert(cls, value, base_unit, new_unit):
        r"""Convert a given value from a base_unit to a new_unit."""
        # check for no conversion case
        if base_unit == new_unit:
            return value
        # hard-coded dictionary of factors
        factors = {cls.ND: 1, cls.PER: 100, cls.P100K: 100000}
        # check for rates and probabilities
        rate_and_probs = {cls.AR, cls.AP, cls.MP}
        if base_unit in rate_and_probs:
            if new_unit not in rate_and_probs:
                raise ValueError('Cannot convert a rate or probability into an inconsistent unit.')
            if base_unit == cls.AR:
                if new_unit == cls.MP:
                    new_value = annual_rate_to_monthly_probability(value)
                elif new_unit == cls.AP:
                    new_value = rate_to_prob(value)
                else:
                    raise NotImplementedError('Shouldn''t be able to get to this line.')
            elif base_unit == cls.AP:
                if new_unit == cls.AR:
                    new_value = prob_to_rate(value)
                elif new_unit == cls.MP:
                    new_value = convert_annual_to_monthly_probability(value)
                else:
                    raise NotImplementedError('Shouldn''t be able to get to this line.')
            elif base_unit == cls.MP:
                AR = MONTHS_PER_YEAR * prob_to_rate(value)
                if new_unit == cls.AR:
                    new_value = AR
                elif new_unit == cls.AP:
                    new_value = rate_to_prob(AR)
                else:
                    raise NotImplementedError('Shouldn''t be able to get to this line.')
            else:
                raise NotImplementedError('Shouldn''t be able to get to this line.')
        else:
            mult1 = factors[base_unit]
            mult2 = factors[new_unit]
            new_value = mult2 / mult1
        return new_value

#%% get_factors
def get_factors(prefix):
    r"""
    Get the multiplication factor and unit label for the desired units.

    Parameters
    ----------
    prefix : str
        Unit standard metric prefix, from:
            {'yotta','zetta','exa','peta','tera','giga','mega',
             'kilo','hecto','deca','unity','deci','centi','milli',
             'micro','nano','pico','femto','atto','zepto','yocto'}

    Returns
    -------
    mult : float
        Multiplication factor
    label : str
        Abbreviation for the prefix

    References
    ----------
    #.  http://en.wikipedia.org/wiki/Metric_prefix

    Examples
    --------
    >>> from dstauffman import get_factors
    >>> (mult, label) = get_factors('micro')
    >>> print(mult)
    1e-06

    >>> print(label)
    μ

    """
    # find the desired units and label prefix
    if prefix == 'yotta':
        mult  = 1e24
        label = 'Y'
    elif prefix == 'zetta':
        mult  = 1e21
        label = 'Z'
    elif prefix == 'exa':
        mult  = 1e18
        label = 'E'
    elif prefix == 'peta':
        mult  = 1e15
        label = 'P'
    elif prefix == 'tera':
        mult  = 1e12
        label = 'T'
    elif prefix == 'giga':
        mult  = 1e9
        label = 'G'
    elif prefix == 'mega':
        mult  = 1e6
        label = 'M'
    elif prefix == 'kilo':
        mult  = 1e3
        label = 'k'
    elif prefix == 'hecto':
        mult  = 1e2
        label = 'h'
    elif prefix == 'deca':
        mult  = 1e1
        label = 'da'
    elif prefix == 'unity':
        mult  = 1.
        label = ''
    elif prefix == 'deci':
        mult  = 1e-1
        label = 'd'
    elif prefix == 'centi':
        mult  = 1e-2
        label = 'c'
    elif prefix == 'milli':
        mult  = 1e-3
        label = 'm'
    elif prefix == 'micro':
        mult  = 1e-6
        label = '\u03bc' # μ # chr(956)
    elif prefix == 'nano':
        mult  = 1e-9
        label = 'n'
    elif prefix == 'pico':
        mult  = 1e-12
        label = 'p'
    elif prefix == 'femto':
        mult  = 1e-15
        label = 'f'
    elif prefix == 'atto':
        mult  = 1e-18
        label = 'a'
    elif prefix == 'zepto':
        mult  = 1e-21
        label = 'z'
    elif prefix == 'yocto':
        mult  = 1e-24
        label = 'y'
    else:
        raise ValueError('Unexpected value for units prefix.')
    return (mult, label)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_units', exit=False)
    doctest.testmod(verbose=False)
