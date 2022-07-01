r"""
Generic time based utilities.

Notes
-----
#.  Written by David C. Stauffer in May 2020.
"""

#%% Imports
from __future__ import annotations

import datetime
import doctest
import re
from time import gmtime, strftime
from typing import Optional, overload, TYPE_CHECKING, Union
import unittest
import warnings

from dstauffman.constants import HAVE_MPL, HAVE_NUMPY, NP_DATETIME_UNITS, NP_INT64_PER_SEC, NP_TIMEDELTA_FORM
from dstauffman.units import get_time_factor, ONE_DAY
from dstauffman.utils import is_datetime

if HAVE_MPL:
    import matplotlib.dates as dates
if HAVE_NUMPY:
    import numpy as np

    isfinite = np.isfinite
    nan = np.nan
else:
    from math import isfinite, nan  # type: ignore[misc]

if TYPE_CHECKING:
    _AllDates = Union[None, int, float, datetime.datetime, datetime.date, np.ndarray, np.datetime64]
    _NPDates = Union[np.datetime64, np.ndarray]

#%% Constants
# maps other names of units to the ones expected by numpy
_NP_MAP = {
    "year": "Y",
    "month": "M",
    "week": "W",
    "day": "D",
    "hour": "h",
    "hr": "h",
    "minute": "m",
    "min": "m",
    "second": "s",
    "sec": "s",
}

#%% Functions - get_np_time_units
def get_np_time_units(date: Union[np.datetime64, np.timedelta64, str]) -> Optional[str]:
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
    >>> x = np.datetime64("now")
    >>> units = get_np_time_units(x)
    >>> print(units)
    s

    """
    # convert the type to a string
    if isinstance(date, str):
        unit_str = date
    else:
        unit_str = str(date.dtype)
    # parse for a name and units in brackets
    matches = re.split(r"\[(.*)\]$", unit_str)
    form = matches[0]
    # do a sanity check and return the result
    assert form in {"datetime64", "timedelta64"}, f'Only expecting datetime64 or timedelta64, not "{form}".'
    return None if len(matches) == 1 else matches[1]


#%% Functions - round_datetime
def round_datetime(dt: datetime.datetime = None, /, round_to_sec: int = 60, floor: bool = False) -> datetime.datetime:
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
    rounding: float
    if floor:
        rounding = seconds // round_to_sec * round_to_sec
    else:
        rounding = (seconds + round_to_sec / 2) // round_to_sec * round_to_sec
    # return the rounded result
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


#%% Functions - round_np_datetime
def round_np_datetime(date_in: np.datetime64, /, time_delta: np.timedelta64, floor: bool = False) -> np.datetime64:
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
    >>> date_in    = date_zero + np.round(1000*dt_sec).astype("timedelta64[ms]")
    >>> time_delta = np.timedelta64(200, "ms")
    >>> date_out   = round_np_datetime(date_in, time_delta)
    >>> expected   = date_zero + np.array([0, 200, 400, 400, 600, 600]).astype("timedelta64[ms]")
    >>> print(all(date_out == expected))
    True

    """
    # check for consistent types
    assert (t1 := get_np_time_units(date_in)) == (
        t2 := get_np_time_units(time_delta)
    ), f'The time refernce types must be the same, not "{t1}" and "{t2}".'
    # check the 64 bit integer representations
    date_in_int: np.ndarray = date_in.astype(np.int64)  # type: ignore[assignment]
    dt_int: np.ndarray = time_delta.astype(np.int64)  # type: ignore[assignment]
    # quantize to the desired unit
    if floor:
        quants = date_in_int // dt_int
    else:
        quants = date_in_int // dt_int + ((date_in_int % dt_int) // (dt_int // 2))
    # scale and convert back to datetime outputs
    date_out: np.datetime64 = (dt_int * quants).astype(date_in.dtype)
    return date_out


#%% Functions - round_num_datetime
def round_num_datetime(date_in: np.ndarray, /, time_delta: float, floor: bool = False) -> np.ndarray:
    r"""
    Rounds a numerical datetime to the given value.

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
    date_out: float
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
    if (max_date / time_delta) > (0.01 / np.finfo(float).eps):
        warnings.warn("This function may have problems if time_delta gets too small.")
    quants = date_in / time_delta
    if floor:
        rounded = np.floor(quants)
    else:
        rounded = np.round(quants)
    date_out: np.ndarray = rounded * time_delta
    return date_out


#%% Functions - round_time
@overload
def round_time(x: np.datetime64, /, t_round: np.timedelta64) -> np.datetime64:
    ...


@overload
def round_time(x: np.ndarray, /, t_round: np.timedelta64) -> np.ndarray:
    ...


def round_time(x: _NPDates, /, t_round: np.timedelta64) -> _NPDates:
    r"""
    Rounding function that handles either numpy datetimes or doubles (seconds).

    Parameters
    ----------
    x : (N, ) ndarray of float or datetime64
        Date to round
    t_round : nd.timedelta64
        Time to round to

    Returns
    -------
    (N, ) ndarray of float or datetime64
        Rounded time

    Notes
    -----
    #.  Written by David C. Stauffer in October 2020.

    Examples
    --------
    >>> from dstauffman import NP_DATETIME_FORM, NP_INT64_PER_SEC, NP_TIMEDELTA_FORM, round_time
    >>> import datetime
    >>> import numpy as np
    >>> date_zero = np.datetime64(datetime.date(2020, 1, 1)).astype(NP_DATETIME_FORM)
    >>> x_sec     = np.array([0, 0.2, 0.35, 0.45, 0.59, 0.61])
    >>> x_np      = date_zero + np.round(NP_INT64_PER_SEC * x_sec).astype(NP_TIMEDELTA_FORM)
    >>> t_round   = np.timedelta64(200, "ms").astype(NP_TIMEDELTA_FORM)

    >>> date_out1 = round_time(x_sec, t_round)
    >>> expected1 = np.array([0., 0.2, 0.4, 0.4, 0.6, 0.6])
    >>> print(all(abs(date_out1 - expected1) < 1e-14))
    True

    >>> date_out2 = round_time(x_np, t_round)
    >>> expected2 = date_zero + np.array([0, 200, 400, 400, 600, 600]).astype("timedelta64[ms]" \
    ...     ).astype(NP_TIMEDELTA_FORM)
    >>> print(all(date_out2 == expected2))
    True

    """
    if is_datetime(x):
        assert np.issubdtype(x.dtype, np.datetime64)  # for typing
        return round_np_datetime(x, t_round)  # type: ignore[arg-type]
    assert isinstance(x, np.ndarray)  # for typing
    return round_num_datetime(x, t_round.astype(np.int64) / NP_INT64_PER_SEC)  # type: ignore[arg-type]


#%% Functions - convert_date
def convert_date(date, form, date_zero=None, *, old_form="sec", numpy_form="datetime64[ns]"):
    r"""
    Converts the given date between different forms.

    Parameters
    ----------
    date : as `old_form`
        Date
    form : str
        Desired date output form, from: {"datetime", "numpy", "matplotlib", "sec", "min", "hr",
            "day", "month", "year"}
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
    #.  This function fails the do one thing principle and is impossible to statically type well.
        Consider splitting into several smaller subfunctions with a wrapper.

    Examples
    --------
    >>> from dstauffman import convert_date
    >>> from datetime import datetime
    >>> date = 3725.5
    >>> form = "datetime"
    >>> date_zero = datetime(2020, 6, 1, 0, 0, 0)
    >>> out = convert_date(date, form, date_zero)
    >>> print(out)
    2020-06-01 01:02:05.500000

    """
    # hard-coded values
    date_forms = {"datetime", "numpy", "matplotlib"}
    # TODO: allow time forms for "min", "hr", "day", "month", "year", etc.
    time_forms = {"sec",}  # fmt: skip
    all_forms = date_forms | time_forms
    # data checks
    assert form in all_forms, f'Unexpected form of "{form}".'
    assert old_form in all_forms, f'Unexpected old_form of "{old_form}".'
    # exit if not changing anything
    if form == old_form:
        return date
    # check MPL conditions
    if not HAVE_MPL and (form == "matplotlib" or old_form == "matplotlib"):
        raise RuntimeError("You must have matplotlib installed to do this conversion.")
    # convert to array as necessary
    is_actual_date: bool
    if form != "datetime" and old_form != "datetime":
        date = np.asanyarray(date)
        is_actual_date = np.any(isfinite(date))
    else:
        # determine if you have real dates to process
        if date is None:
            is_actual_date = False
        elif isinstance(date, (datetime.datetime, datetime.date)):
            is_actual_date = True
        elif isinstance(date, list):
            # Note: this assumes all entries of the list are the same
            is_actual_date = len(date) > 0 and isinstance(date[0], (datetime.datetime, datetime.date))
        elif isinstance(date, (int, float)):
            is_actual_date = isfinite(date)
        else:
            is_actual_date = np.any(isfinite(date))
    # check for bad date_zero
    if form in time_forms or (old_form in time_forms and is_actual_date):
        assert date_zero is not None, "You must specify a date_zero."
        assert isinstance(date_zero, datetime.datetime), "The date_zero is expected to be a datetime object."
    # do all possible conversions
    # from seconds
    if old_form in time_forms:
        is_num = isfinite(date) if is_actual_date else False
        if form == "datetime":
            out = date_zero + datetime.timedelta(seconds=date) if is_num else None
        elif form == "numpy":
            out = np.full(date.shape, np.datetime64("nat"), dtype=numpy_form)
            if np.any(is_num):
                datetime_units = get_np_time_units(numpy_form)
                date_zero_np = np.datetime64(date_zero) if datetime_units is None else np.datetime64(date_zero, datetime_units)
                if np.issubdtype(date.dtype, np.signedinteger):
                    out[is_num] = (date_zero_np + (date[is_num].astype(np.int64) * 10**9).astype("timedelta64[ns]")).astype(numpy_form)  # fmt: skip
                else:
                    out[is_num] = (date_zero_np + np.round(date[is_num] * 1e9).astype("timedelta64[ns]")).astype(numpy_form)
        elif form == "matplotlib":  # pragma: no branch
            out = date.copy()
            if np.any(is_num):
                out[is_num] = dates.date2num(date_zero) + date[is_num] / ONE_DAY
    # from datetime
    elif old_form == "datetime":
        is_num = date is not None
        if form == "numpy":
            out = np.array(date, dtype=numpy_form)
        elif form == "matplotlib":
            out = dates.date2num(date) if is_num else np.nan
        elif form in time_forms:  # pragma: no branch
            if is_num:
                dt = date - date_zero
                out = ONE_DAY * dt.days + dt.seconds + dt.microseconds / 1000000
            else:
                out = nan
    # from numpy
    elif old_form == "numpy":
        is_num = ~np.isnat(date)
        if form == "datetime":
            out = datetime.datetime.utcfromtimestamp(date.astype("datetime64[ns]").astype(np.int64) / 10**9) if is_num else None  # fmt: skip
        elif form == "matplotlib":
            out = dates.date2num(date)
        elif form in time_forms:  # pragma: no branch
            out = np.full(date.shape, np.nan)
            if np.any(is_num):
                out[is_num] = (date[is_num] - np.array(date_zero, dtype="datetime64[ns]")).astype("timedelta64[ns]").astype(np.int64) / 10**9  # fmt: skip
    # from matplotlib
    elif old_form == "matplotlib":  # pragma: no branch
        is_num = np.isfinite(date)
        if form == "datetime":
            out = dates.num2date(date) if is_num else None
        elif form == "numpy":
            out = np.full(date.shape, np.datetime64("nat"), dtype=numpy_form)
            if np.any(is_num):
                # TODO: I don't like this method, but the dates.num2date always returns a timezone aware datetime
                out[is_num] = np.array([x.replace(tzinfo=None) for x in dates.num2date(date[is_num])], dtype=numpy_form)
        elif form in time_forms:  # pragma: no branch
            out = ONE_DAY * (date - dates.date2num(date_zero))
    # convert from seconds to other time forms if necessary
    if form in time_forms and form != "sec":
        raise ValueError("Time forms other than seconds are not yet implemented.")  # pragma: no cover
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
    >>> old_unit = "sec"
    >>> new_unit = "hr"
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


#%% Functions - convert_datetime_to_np
def convert_datetime_to_np(time, /, units=NP_DATETIME_UNITS):
    r"""
    Convenience wrapper to convert a datetime.datetime to a numpy.datetime64 with the desired units.

    Parameters
    ----------
    time : datetime.datetime
        Date/time
    units : str, optional
        Units to use within numpy.datetime64

    Returns
    -------
    np.datetime64
        Date/time as np.datetime64 equivalent

    Notes
    -----
    #.  Written by David C. Stauffer in October 2020.

    Examples
    --------
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> time = datetime.datetime(2020, 10, 1, 12, 34, 56, 789)
    >>> print(convert_datetime_to_np(time))
    2020-10-01T12:34:56.000789000

    """
    if isinstance(time, list):
        out = np.empty(len(time), dtype="datetime64[" + units + "]")
        for (ix, t) in enumerate(time):
            out[ix] = np.datetime64(t, units)
        return out
    return np.datetime64(time, units)


#%% Functions - convert_duration_to_np
def convert_duration_to_np(dt, /, units=NP_DATETIME_UNITS):
    r"""Convenience wrapper to convert a datetime.timedelta to a numpy.timedelta64 with the desired units.

    Parameters
    ----------
    time : datetime.timedelta
        time duration
    units : str, optional
        Units to use within numpy.datetime64

    Returns
    -------
    np.timedelta64
        Time duration as np.timedelta64 equivalent

    Notes
    -----
    #.  Written by David C. Stauffer in October 2020.

    Examples
    --------
    >>> from dstauffman import convert_duration_to_np
    >>> import datetime
    >>> dt = datetime.timedelta(minutes=90)
    >>> print(convert_duration_to_np(dt))
    5400000000000 nanoseconds

    """
    return np.timedelta64(dt, units)


#%% Functions - convert_num_dt_to_np
def convert_num_dt_to_np(dt, /, units="sec", np_units=NP_TIMEDELTA_FORM):
    r"""Convenience wrapper to convert a number of seconds to a numpy.timedelta64 with the desired units.

    Parameters
    ----------
    dt: np.ndarray of float
        time duration [sec]
    units : str, optional
        Units to use within numpy.datetime64

    Returns
    -------
    np.timedelta64
        Time duration as np.timedelta64 equivalent

    Notes
    -----
    #.  Written by David C. Stauffer in October 2020.

    Examples
    --------
    >>> from dstauffman import convert_num_dt_to_np, ONE_MINUTE
    >>> import datetime
    >>> dt = 90 * ONE_MINUTE
    >>> print(convert_num_dt_to_np(dt))
    5400000000000 nanoseconds

    """
    units = _NP_MAP.get(units, units)
    return np.timedelta64(dt, units).astype(np_units)


#%% Functions - get_delta_time_str
def get_delta_time_str(
    start_time: Union[datetime.datetime, datetime.timedelta], final_time: datetime.datetime = None, *, format_: str = "%H:%M:%S"
) -> str:
    r"""
    Determine the elapsed time in a form useful for logging.

    Parameters
    ----------
    start_time : class datetime.datetime or datetime.timedelta
        Initial time, or a duration, in which case you don't need a final time
    final_time : class datetime.datetime
        Final time to be differenced with the start_time
    format_ : str
        Format used to display the results

    Results
    -------
    str
        Duration as a string

    Notes
    -----
    #.  Written by David C. Stauffer in May 2021.
    #.  This function would be completely superfluous if datetime.timedelta had a strftime method.

    Examples
    --------
    >>> from dstauffman import get_delta_time_str
    >>> import datetime
    >>> start_time = datetime.datetime.now()
    >>> final_time = start_time + datetime.timedelta(seconds=5, microseconds=10000)
    >>> print(get_delta_time_str(start_time, final_time))
    00:00:05

    """
    if isinstance(start_time, datetime.timedelta):
        delta_time = start_time
        assert final_time is None
    elif final_time is None:
        delta_time = datetime.datetime.now() - start_time
    else:
        delta_time = final_time - start_time
    return strftime(format_, gmtime(delta_time.total_seconds()))


#%% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_time", exit=False)
    doctest.testmod(verbose=False)
