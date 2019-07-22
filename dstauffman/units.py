# -*- coding: utf-8 -*-
r"""
Contains utilities and functions dealing with unit conversions.

Notes
-----
#.  Written by David C. Stauffer in February 2016.

"""

#%% Imports
import doctest
import unittest

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
    if isinstance(prefix, (int,float)):
        mult = prefix
        label = '';
        return (mult, label)
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
    unittest.main(module='dstauffman.tests.test_units', exit=False)
    doctest.testmod(verbose=False)
