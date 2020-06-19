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

import matplotlib.dates as dates
import numpy as np

from dstauffman.units import ONE_DAY

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

#%% Functions - convert_date
def convert_date(date, form, date_zero=None, *, old_form='sec', numpy_form='datetime64[ns]'):
    r"""
    Converts the given date between different forms.

    Parameters
    ----------
    date : as `olf_form`
        Date
    form : str
        Desired date output form, from {'datetime', 'numpy', 'matplotlib', 'sec', 'min', 'hr', 'day', 'month', 'year'}
    date_zero : class datetime.datetime
        Date represented by zero when using numeric forms
    old_form : str
        Form of given date input, same options as `form`
    numpy_form : str
        If using numpy datetimes, then they are in this form

    Returns
    -------
    out : as `form`
        Date in new desired form

    Notes
    -----
    #.  Can handle NaNs, NaTs, and Infs
    #.  Written by David C. Stauffer in June 2020.

    Examples
    --------
    >>> from dstauffman import convert_date
    >>> from datetime import datetime
    >>> date = 3725.5
    >>> form = 'datetime'
    >>> date_zero = datetime(2020, 6, 1, 0, 0, 0)
    >>> out = convert_date(date, form, date_zero)
    >>> print(out)
    2020-06-01 01:02:05.500000

    """
    # hard-coded values
    date_forms = {'datetime', 'numpy', 'matplotlib'}
    time_forms = {'sec', } # TODO: allow for 'min', 'hr', 'day', 'month', 'year', etc.
    all_forms = date_forms | time_forms
    # data checks
    assert form in all_forms, f'Unexpected form of "{form}".'
    assert old_form in all_forms, f'Unexpected old_form of "{old_form}".'
    # exit if not changing anything
    if form == old_form:
        return date
    if form in time_forms or (old_form in time_forms and np.isfinite(date)):
        assert date_zero is not None, 'You must specify a date_zero.'
        assert isinstance(date_zero, datetime.datetime), 'The date_zero is expected to be a datetime object.'
    # do all possible conversions
    # from seconds
    if old_form in time_forms:
        is_num = np.isfinite(date)
        if form == 'datetime':
            out = date_zero + datetime.timedelta(seconds=date) if is_num else None # TODO: or np.datetime64('nat')
        elif form == 'numpy':
            if is_num:
                out = (np.datetime64(date_zero, dtype=numpy_form) + np.timedelta64(np.round(date* 10**9).astype(np.int64), 'ns')).astype(numpy_form)
            else:
                out = np.datetime64('nat', dtype=numpy_form)
        elif form == 'matplotlib':
            out = dates.date2num(date_zero) + date / ONE_DAY if is_num else date
    # from datetime
    elif old_form == 'datetime':
        is_num = date is not None
        if form == 'numpy':
            out = np.array(date, dtype=numpy_form)
        elif form == 'matplotlib':
            out = dates.date2num(date) if is_num else np.nan
        elif form in time_forms:
            if is_num:
                dt = date - date_zero
                out = ONE_DAY * dt.days + dt.seconds + dt.microseconds / 1000000
            else:
                out = np.nan
    # from numpy
    elif old_form == 'numpy':
        is_num = ~np.isnat(date)
        if form == 'datetime':
            out = datetime.datetime.utcfromtimestamp(date.astype('datetime64[ns]').astype(np.int64) / 10**9) if is_num else None
        elif form == 'matplotlib':
            out = dates.date2num(date)
        elif form in time_forms:
            if is_num:
                out = (date - np.array(date_zero, dtype='datetime64[ns]')).astype('timedelta64[ns]').astype(np.int64) / 10**9
            else:
                out = np.nan
    # from matplotlib
    elif old_form == 'matplotlib':
        is_num = np.isfinite(date)
        if form == 'datetime':
            out = dates.num2date(date) if is_num else None
        elif form == 'numpy':
            out = np.array(dates.num2date(date), dtype=numpy_form) if is_num else np.datetime64('nat')
        elif form in time_forms:
            out = ONE_DAY * (date - dates.date2num(date_zero))
    # convert from seconds to other time forms if necessary
    if form in time_forms and form != 'sec':
        raise ValueError('Time forms other than seconds are not yet implemented.')
    return out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_time', exit=False)
    doctest.testmod(verbose=False)
