# -*- coding: utf-8 -*-
r"""
Generic time based utilities.

Notes
-----
#.  Written by David C. Stauffer in May 2020.

"""

#%% Imports
import datetime
import doctest
import re
import unittest

import numpy as np

#%% Functions - get_np_time_units
def get_np_time_units(date):
    r"""
    Gets the units for a given datetime64 or timedelta64.

    Parameters
    ----------
    date : ndarray
        Date to get the units for

    Returns
    -------
    str
        Name of the units

    See Also
    --------
    numpy.datetime64, numpy.timedelta64

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020.

    Examples
    --------
    >>> from dstauffman import get_np_time_units
    >>> import numpy as np
    >>> x = np.datetime64('now')
    >>> units = get_np_time_units(x)
    >>> print(units)
    s

    """
    # convert the type to a string
    unit_str = str(date.dtype)
    # parse for a name and units in brackets
    matches = re.split(r'\[(.*)\]$', unit_str)
    form    = matches[0]
    # do a sanity check and return the result
    assert form in {'datetime64', 'timedelta64'}, f'Only expecting datetime64 or timedelta64, not "{form}".'
    return matches[1]

#%% Functions - round_datetime
def round_datetime(dt=None, round_to_sec=60, floor=False):
    r"""
    Round a datetime object to any time lapse in seconds.

    Parameters
    ----------
    dt : datetime.datetime
        time to round, default now.
    round_to_sec : int
        Closest number of seconds to round to, default 60 seconds (i.e. rounds to nearest minute)

    Returns
    -------
    datetime.datetime
        Time rounded as specified

    See Also
    --------
    datetime.datetime

    Notes
    -----
    #. Originally written by Thierry Husson 2012.  Freely distributed.
    #. Adapted by David C. Stauffer in Feb 2015.

    Examples
    --------
    >>> from dstauffman import round_datetime
    >>> from datetime import datetime
    >>> dt = datetime(2015, 3, 13, 8, 4, 10)
    >>> rounded_time = round_datetime(dt)
    >>> print(rounded_time)
    2015-03-13 08:04:00

    """
    # set default for dt
    if dt is None:
        dt = datetime.datetime.now()
    # get the current elasped time in seconds
    seconds = (dt - dt.min).seconds
    # round to the nearest whole second
    if floor:
        rounding = seconds // round_to_sec * round_to_sec
    else:
        rounding = (seconds+round_to_sec/2) // round_to_sec * round_to_sec
    # return the rounded result
    return dt + datetime.timedelta(0, rounding-seconds, -dt.microsecond)

#%% Functions - round_np_datetime
def round_np_datetime(date_in, time_delta, floor=False):
    r"""
    Rounds a numpy datetime64 time to the specified delta.

    Parameters
    ----------
    date_in : numpy.datetime64
        Date to round
    time_delta : numpy.timedelta64
        Delta time to round the date to
    floor : bool
        Whether to round or floor the result, default is False, meaning round

    Returns
    -------
    date_out : numpy.datetime64
        Rounded date

    Notes
    -----
    #.  The date_in and time_delta values need to be in the same integer time unit basis
    #.  Written by David C. Stauffer in May 2020.

    Examples
    --------
    >>> from dstauffman import round_np_datetime
    >>> import datetime
    >>> import numpy as np
    >>> date_zero  = np.datetime64(datetime.date(2020, 1, 1))
    >>> dt_sec     = np.array([0, 0.2, 0.35, 0.45, 0.59, 0.61])
    >>> date_in    = date_zero + np.round(1000*dt_sec).astype('timedelta64[ms]')
    >>> time_delta = np.timedelta64(200, 'ms')
    >>> date_out   = round_np_datetime(date_in, time_delta)
    >>> expected   = date_zero + np.array([0, 200, 400, 400, 600, 600]).astype('timedelta64[ms]')
    >>> print(all(date_out == expected))
    True

    """
    # check for consistent types
    # For v3.8 in the future:
    #assert t1 := get_np_time_units(date_in) == t2 := get_np_time_units(time_delta), \
    assert get_np_time_units(date_in) == get_np_time_units(time_delta), 'The time refernce types ' + \
    'must be the same, not {} and {}.'.format(str(date_in.dtype), str(time_delta.dtype)) # t1, t2
    # check the 64 bit integer representations
    date_in_int = date_in.astype(np.int64)
    dt_int      = time_delta.astype(np.int64)
    # quantize to the desired unit
    if floor:
        quants  = date_in_int // dt_int
    else:
        quants  = date_in_int // dt_int + ((date_in_int % dt_int) // (dt_int // 2))
    # scale and convert back to datetime outputs
    date_out    = (dt_int*quants).astype(date_in.dtype)
    return date_out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_time', exit=False)
    doctest.testmod(verbose=False)
