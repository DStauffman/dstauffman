# -*- coding: utf-8 -*-
r"""
Enums module file for the dstauffman library.  It defines enumerator related functions for the rest
of the code..

Notes
-----
#.  Written by David C. Stauffer in July 2015.
"""

#%% Imports
import doctest
from enum import Enum, EnumMeta, _is_dunder
import numpy as np
import unittest

#%% Classes - _EnumMetaPlus
class _EnumMetaPlus(EnumMeta):
    r"""
    Overrides the repr/str methods of the EnumMeta class to display all possible values.
    Also makes the __getattr__ attribute error more explicit.
    """
    def __repr__(cls):
        return '\n'.join((repr(field) for field in cls))
    def __str__(cls):
        return '\n'.join((str(field) for field in cls))
    def __getattr__(cls, name):
        r"""
        Return the enum member matching `name`.
        """
        if _is_dunder(name):
            raise AttributeError(name)
        try:
            return cls._member_map_[name]
        except KeyError:
            text = '"{}" does not have an attribute of "{}"'.format(cls.__name__,name)
            raise AttributeError(text) from None
    def list_of_names(self):
        r"""
        Returns a list of all the names within the enumerator.
        """
        # look for class.name: pattern, ignore class, return names only
        names = list(self.__members__.keys())
        return names
    def list_of_values(self):
        r"""
        Returns a list of all the values within the enumerator.
        """
        values = list(self.__members__.values())
        return values
    @property
    def num_values(self):
        r"""
        Returns the number of values within the enumerator.
        """
        return len(self)
    @property
    def min_value(self):
        r"""
        Returns the minimum value of the enumerator.
        """
        return min(self.__members__.values())
    @property
    def max_value(self):
        r"""
        Returns the maximum value of the enumerator.
        """
        return max(self.__members__.values())

#%% Classes - IntEnumPlus
class IntEnumPlus(int, Enum, metaclass=_EnumMetaPlus):
    r"""
    Custom IntEnum class based on _EnumMetaPlus metaclass to get more details from repr/str.
    """
    def __str__(self):
        return '{}.{}: {}'.format(self.__class__.__name__, self.name, self.value)

#%% Decorators - contiguous
def consecutive(enumeration):
    r"""
    Class decorator for enumerations ensuring unique and consecutive member values.
    """
    duplicates = []
    non_consecutive = []
    last_value = min(enumeration.__members__.values()) - 1
    if last_value != -1:
        raise ValueError('Bad starting value (should be zero): {}'.format(last_value+1))
    for name, member in enumeration.__members__.items():
        if name != member.name:
            duplicates.append((name, member.name))
        if member != last_value + 1:
            non_consecutive.append((name, member))
        last_value = member
    if duplicates:
        alias_details = ', '.join(
                ["%s -> %s" % (alias, name) for (alias, name) in duplicates])
        raise ValueError('duplicate values found in %r: %s' %
                (enumeration, alias_details))
    if non_consecutive:
        alias_details = ', '.join('{}:{}'.format(name, member) for (name, member) in non_consecutive)
        raise ValueError('non-consecutive values found in {}: {}'.format(enumeration.__name__, alias_details))
    return enumeration

#%% Functions - dist_enum_and_mons
def dist_enum_and_mons(num, distribution, prng, *, max_months=None, start_num=1, alpha=1, beta=1):
    r"""
    Creates a distribution for an enumerated state with a duration (such as a disease status).

    Parameters
    ----------
    num : int
        Number of people in the population
    distribution : array_like
        Likelihood of being in each state (should cumsum to 100%)
    prng : class numpy.random.RandomState
        Pseudo-random number generator
    max_months : scalar or array_like, optional
        Maximum number of months for being in each state
    start_num : int, optional
        Number to start counting from, default is 1
    alpha : int, optional
        The alpha parameter for the beta distribution
    beta : int, optional
        The beta parameter for the beta distribution

    Returns
    -------
    state : ndarray
        Enumerated status for this month for everyone in the population
    mons : ndarray
        Number of months in this state for anyone with an infection

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.
    #.  Updated by David C. Stauffer in June 2015 to use a beta curve to distribute the number of
        months spent in each state.
    #.  Made into a generic function for the dstauffman library by David C. Stauffer in July 2015.
    #.  Updated by David C. Stauffer in November 2015 to change the inputs to allow max_months and
        mons output to be optional.

    Examples
    --------

    >>> from dstauffman import dist_enum_and_mons
    >>> import numpy as np
    >>> num = 10000
    >>> distribution = 1./100*np.array([9.5, 90, 0.25, 0.25])
    >>> max_months = np.array([60, 120, 36, 6])
    >>> prng = np.random.RandomState()
    >>> (state, mons) = dist_enum_and_mons(num, distribution, prng, max_months=max_months)

    """
    # do a random draw based on the cumulative distribution
    state = np.sum(prng.rand(num) >= np.expand_dims(np.cumsum(distribution), axis=1), \
        axis=0, dtype=int) + start_num
    # set the number of months in this state based on a beta distribution with the given
    # maximum number of months in each state
    if max_months is None:
        mons = None
    else:
        if np.isscalar(max_months):
            max_months = max_months *np.ones(len(state))
        mons = np.ceil(max_months[state-start_num] * prng.beta(alpha, beta, num)).astype(int)
    return (state, mons)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_enums', exit=False)
    doctest.testmod(verbose=False)
