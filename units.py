# -*- coding: utf-8 -*-
r"""
Units module file for the "dstauffman" library.  It contains generic utilities that can be
independently defined and used by other modules.

Notes
-----
#.  By design, this module does not reference any other piece of the dstauffman code base except
        constants or enums to avoid circular references.
#.  Written by David C. Stauffer in February 2016.
"""
# pylint: disable=E1101, C0301, C0103

#%% Imports
import doctest
import unittest
from dstauffman.classes import Frozen
from dstauffman.enums   import IntEnumPlus

#%% Units
class Units(IntEnumPlus):
    AR    = 1 # annual rate
    AP    = 2 # annual probability
    MP    = 3 # monthly probability
    ND    = 4 # nondimensional
    PER   = 5 # percentage
    P100K = 6 # per 100,000

    @staticmethod
    def convert(base_unit, new_unit):
        rate_and_probs = {Units.AR, Units.AP, Units.MP}
        if base_unit in rate_and_probs:
            if new_unit not in rate_and_probs:
                raise ValueError('Cannot convert a rate or probability into an inconsistent unit.')
            if base_unit == Units.AR:
                pass
            elif base_unit == Units.AP:
                pass
            elif base_unit == Units.MP:
                pass
            else:
                raise NotImplemented('This unit conversion hasn''t been implemented.')
        return 1

#%% Param
class Param(Frozen):
    def __init__(self, value, name='', reference='', *, base_unit=None, disp_unit=None):
        self.value     = value
        self.name      = name
        self.reference = reference
        self.base_unit = base_unit
        self.disp_unit = disp_unit

    def __call__(self):
        return self.value

    def pretty_print(self):
        return '{} {}'.format(self.get_value(), self.get_units())

    def get_value(self):
        pass

    def get_units(self):
        pass

#%% get_factors
def get_factors(prefix):
    r"""
    Gets the multiplication factor and unit label for the desired units.

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