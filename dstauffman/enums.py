r"""
Define enumerator related functions for the rest of the library.

Notes
-----
#.  Written by David C. Stauffer in July 2015.
#.  Modified by David C. Stauffer in March 2020 to add ReturnCode class for use in commands, and as
    a great example use case.
"""

#%% Imports
import doctest
import unittest
from enum import Enum, EnumMeta

#%% Support functions
def _is_dunder(name):
    """Returns True if a __dunder__ name, False otherwise."""
    # Note that this is copied from the enum library, as it is not part of their public API.
    return len(name) > 4 and name[:2] == name[-2:] == '__' and name[2] != '_' and name[-3] != '_'

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
        r"""Return the enum member matching `name`."""
        if _is_dunder(name):
            raise AttributeError(name)
        try:
            return cls._member_map_[name]
        except KeyError:
            text = '"{}" does not have an attribute of "{}"'.format(cls.__name__, name)
            raise AttributeError(text) from None
    def list_of_names(cls):
        r"""Return a list of all the names within the enumerator."""
        # look for class.name: pattern, ignore class, return names only
        names = list(cls.__members__)
        return names
    def list_of_values(cls):
        r"""Return a list of all the values within the enumerator."""
        values = list(cls.__members__.values())
        return values
    @property
    def num_values(cls):
        r"""Return the number of values within the enumerator."""
        return len(cls)
    @property
    def min_value(cls):
        r"""Return the minimum value of the enumerator."""
        return min(cls.__members__.values())
    @property
    def max_value(cls):
        r"""Return the maximum value of the enumerator."""
        return max(cls.__members__.values())

#%% Classes - IntEnumPlus
class IntEnumPlus(int, Enum, metaclass=_EnumMetaPlus):
    r"""
    Custom IntEnum class based on _EnumMetaPlus metaclass to get more details from repr/str.
    Plus it includes additional methods for convenient retrieval of number of values, their names,
    mins and maxes.
    """
    def __str__(self):
        r"""Return string representation."""
        return '{}.{}: {}'.format(self.__class__.__name__, self.name, self.value)

#%% Decorators - consecutive
def consecutive(enumeration):
    r"""Class decorator for enumerations ensuring unique and consecutive member values that start from zero."""
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
        alias_details = ', '.join(['{} -> {}'.format(alias, name) for (alias, name) in duplicates])
        raise ValueError('Duplicate values found in {}: {}'.format(enumeration.__name__, alias_details))
    if non_consecutive:
        alias_details = ', '.join('{}:{}'.format(name, member) for (name, member) in non_consecutive)
        raise ValueError('Non-consecutive values found in {}: {}'.format(enumeration.__name__, alias_details))
    return enumeration

#%% Enums - ReturnCodes
@consecutive
class ReturnCodes(IntEnumPlus):
    r"""
    Return codes for use as outputs in the command line API.

    Examples
    --------
    >>> from dstauffman import ReturnCodes
    >>> rc = ReturnCodes.clean
    >>> print(rc)
    ReturnCodes.clean: 0

    """
    clean: int            = 0 # Clean exit
    bad_command: int      = 1 # Unexpected command
    bad_folder: int       = 2 # Folder to execute a command in doesn't exist
    bad_help_file: int    = 3 # help file doesn't exist
    test_failures: int    = 4 # A test ran to completion, but failed its criteria
    no_coverage_tool: int = 5 # Coverage tool is not installed

#%% Enums - LogLevel
class LogLevel(IntEnumPlus):
    r"""
    Add 10-ish custom levels that give more degradation beween WARNING, INFO and DEBUG.
        50 (CRITICAL, FATAL)
        40 (ERROR)
    L0  35
    L1  30 (WARNING, WARN)
    L2  28
    L3  26
    L4  24
    L5  20 (INFO)
    L6  18
    L7  16
    L8  14
    L9  12
    L10 10 (DEBUG)
    L11  9
    L12  8
    L20  0  (NOTSET)

    Examples
    --------
    >>> from dstauffman import LogLevel
    >>> print(LogLevel.L5)
    LogLevel.L5: 20

    """
    L0: int  = 35
    L1: int  = 30
    L2: int  = 28
    L3: int  = 26
    L4: int  = 24
    L5: int  = 20
    L6: int  = 18
    L7: int  = 16
    L8: int  = 14
    L9: int  = 12
    L10: int = 10
    L11: int =  9
    L12: int =  8
    L20: int =  0

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_enums', exit=False)
    doctest.testmod(verbose=False)
