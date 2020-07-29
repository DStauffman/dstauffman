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
import warnings

try:
    import matplotlib.dates as dates
except ImportError:
    dates = None
import numpy as np

from dstauffman.units import get_time_factor, ONE_DAY

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

#%% Functions - round_num_datetime
def round_num_datetime(date_in, time_delta, floor=False):
    r"""
    Rounds a numerica datetime to the given value.

    Parameters
    ----------
    date_in : float
        Date to round
    time_delta : float
        Delta time to round the date to
    floor : bool
        Whether to round or floor the result, default is False, meaning round

    Returns
    -------
    float
        Rounded date

    Notes
    -----
    #.  The date_in and time_delta values need to be in the same time units, typically could be
        seconds or days.
    #.  Written by David C. Stauffer in June 2020.

    Examples
    --------
    >>> from dstauffman import round_num_datetime
    >>> import numpy as np
    >>> date_exact = np.arange(0, 10.1, 0.1)
    >>> date_in    = date_exact + 0.001 * np.random.rand(101)
    >>> time_delta = 0.1
    >>> date_out   = round_num_datetime(date_in, time_delta)
    >>> print(np.all(np.abs(date_out - date_exact) < 1e-12))
    True

    """
    # check if date value is too close to the tolerance floor
    max_date = np.max(np.abs(date_in), initial=0)
    if (max_date / time_delta) > (0.01/ np.finfo(float).eps):
        warnings.warn('This function may have problems if time_delta gets too small.')
    quants = date_in / time_delta
    if floor:
        rounded = np.floor(quants)
    else:
        rounded = np.round(quants)
    return rounded * time_delta

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
    # check for bad conditions
    if dates is None and (form == 'matplotlib' or old_form == 'matplotlib'):
        raise RuntimeError('You must have matplotlib installed to do this conversion.')
    if form in time_forms or (old_form in time_forms and np.any(np.isfinite(date))):
        assert date_zero is not None, 'You must specify a date_zero.'
        assert isinstance(date_zero, datetime.datetime), 'The date_zero is expected to be a datetime object.'
    # convert to array as necessary
    if form != 'datetime' and old_form != 'datetime':
        date = np.asanyarray(date)
    # do all possible conversions
    # from seconds
    if old_form in time_forms:
        is_num = np.isfinite(date)
        if form == 'datetime':
            out = date_zero + datetime.timedelta(seconds=date) if is_num else None # TODO: or np.datetime64('nat')
        elif form == 'numpy':
            out = np.full(date.shape, np.datetime64('nat', dtype=numpy_form), dtype=numpy_form)
            if np.any(is_num):
                out[is_num] = (np.datetime64(date_zero, dtype=numpy_form) + np.round(date[is_num] * 10**9).astype('timedelta64[ns]')).astype(numpy_form)
        elif form == 'matplotlib':
            out = date.copy()
            if np.any(is_num):
                out[is_num] = dates.date2num(date_zero) + date[is_num] / ONE_DAY
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
            out = np.full(date.shape, np.nan)
            if np.any(is_num):
                out[is_num] = (date[is_num] - np.array(date_zero, dtype='datetime64[ns]')).astype('timedelta64[ns]').astype(np.int64) / 10**9
    # from matplotlib
    elif old_form == 'matplotlib':
        is_num = np.isfinite(date)
        if form == 'datetime':
            out = dates.num2date(date) if is_num else None
        elif form == 'numpy':
            out = np.full(date.shape, np.datetime64('nat'), dtype=numpy_form)
            if np.any(is_num):
                out[is_num] = np.array(dates.num2date(date[is_num]), dtype=numpy_form)
        elif form in time_forms:
            out = ONE_DAY * (date - dates.date2num(date_zero))
    # convert from seconds to other time forms if necessary
    if form in time_forms and form != 'sec':
        raise ValueError('Time forms other than seconds are not yet implemented.')
    return out

#%% Functions - convert_time_units
def convert_time_units(time, old_unit, new_unit):
    r"""
    Converts the given time history from the old units to the new units.

    Parameters
    ----------
    time : array_like
        Time history in the old units
    old_unit : str
        Name of the old units
    new_unit : str
        Name of the desired new units

    Returns
    -------
    out : ndarray
        New time history in the new units

    Notes
    -----
    #.  Written by David C. Stauffer in June 2020.

    Examples
    --------
    >>> from dstauffman import convert_time_units
    >>> time = 7200.
    >>> old_unit = 'sec'
    >>> new_unit = 'hr'
    >>> out = convert_time_units(time, old_unit, new_unit)
    >>> print(out)
    2.0

    """
    if old_unit == new_unit:
        return time
    mult_old = get_time_factor(old_unit)
    mult_new = get_time_factor(new_unit)
    mult = mult_old / mult_new
    return time * mult

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_time', exit=False)
    doctest.testmod(verbose=False)
