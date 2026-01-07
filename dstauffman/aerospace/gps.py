r"""
Classes related to GPS processing.

Notes
-----
#.  Written by David C. Stauffer in July 2021.

"""

# %% Imports
from __future__ import annotations

import datetime
import doctest
from typing import Literal, NoReturn, overload, TYPE_CHECKING
import unittest

from dstauffman import HAVE_NUMPY, NP_DATETIME_UNITS, np_digitize, NP_ONE_DAY, NP_ONE_SECOND

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _D = NDArray[np.datetime64]
    _I = NDArray[np.int_]
    _N = NDArray[np.floating]

# %% Constants
GPS_DATE_ZERO = datetime.datetime(1980, 1, 6, 0, 0, 0)
ONE_DAY       = 86400  # fmt: skip
DAYS_PER_WEEK = 7
WEEK_ROLLOVER = 1024
NP_GPS_DATE_ZERO = np.datetime64(GPS_DATE_ZERO, NP_DATETIME_UNITS) if HAVE_NUMPY else None


# %% Mypy support - _assert_never
def _assert_never(value: NoReturn) -> NoReturn:
    r"""Static and Runtime checker for possible options."""
    raise ValueError(f"This code should never be reached, got: {value}")


# %% Functions - bsl
def bsl(bits: _I, shift: int = 1, *, inplace: bool = False) -> _I:
    r"""
    Bit shifts left.

    Parameters
    ----------
    bits : (N, ) ndarray
        input bit stream as a matrix
    shift : int
        number of bits to shift
    inplace : bool, optional, default is False
        Whether to do the bitshift inplace

    Returns
    -------
    out : (N, ) ndarray
        output bit stream as a matrix

    See Also
    --------
    dstauffman.aerospace.bsr

    Notes
    -----
    #.  Written by David C. Stauffer in Jan 2009 in MATLAB.
    #.  Translated by David C. Stauffer in July 2021 into Python.

    Examples
    --------
    >>> from dstauffman.aerospace import bsl
    >>> import numpy as np
    >>> bits = np.array([0, 0, 1, 1, 1])
    >>> out = bsl(bits)
    >>> print(out)
    [0 1 1 1 0]

    """
    out = bits if inplace else np.empty_like(bits)
    out[:] = np.roll(bits, shift=-shift)
    return out


# %% Functions - bsr
def bsr(bits: _I, shift: int = 1, *, inplace: bool = False) -> _I:
    r"""
    Bit shifts right.

    Parameters
    ----------
    bits : (N, ) ndarray
        input bit stream as a matrix
    shift : int
        number of bits to shift
    inplace : bool, optional, default is False
        Whether to do the bitshift inplace

    Returns
    -------
    out : (N, ) ndarray
        output bit stream as a matrix

    See Also
    --------
    dstauffman.aerospace.bsl

    Notes
    -----
    #.  Written by David C. Stauffer in Jan 2009 in MATLAB.
    #.  Translated by David C. Stauffer in July 2021 into Python.

    Examples
    --------
    >>> from dstauffman.aerospace import bsr
    >>> import numpy as np
    >>> bits = np.array([0, 0, 1, 1, 1])
    >>> out = bsr(bits)
    >>> print(out)
    [1 0 0 1 1]

    """
    # TODO: just alias to bsl(bits, shift=-shift, inplace=inplace)
    out = bits if inplace else np.empty_like(bits)
    out[:] = np.roll(bits, shift=shift)
    return out


# %% Functions - prn_01_to_m11
def prn_01_to_m11(bits: _I, *, inplace: bool = False) -> _I:
    r"""
    Shifts bits from (0, 1) to (1, -1).

    Parameters
    ----------
    bits : (N, ) ndarray
        PRN sequence of 0's and 1's
    inplace : bool, optional, default is False
        Whether to do the bitshift inplace

    Returns
    -------
    out : (N, ) ndarray
        PRN sequence of 1's and -1's

    Notes
    -----
    #.  Written by David C. Stauffer in Jan 2009 in MATLAB.
    #.  Translated by David C. Stauffer in July 2021 into Python.

    Examples
    --------
    >>> from dstauffman.aerospace import prn_01_to_m11
    >>> import numpy as np
    >>> bits = np.array([1, 1, 1, 0, 0, 1, 1])
    >>> out = prn_01_to_m11(bits)
    >>> print(out)
    [-1 -1 -1  1  1 -1 -1]

    """
    out = bits if inplace else np.empty_like(bits)
    out[:] = np.where(bits == 0, 1, -1)
    return out


# %% Functions - get_prn_bits
def get_prn_bits(sat: int) -> tuple[int, int]:
    r"""
    Gets the bit numbers to generate the desired prn sequence.

    Parameters
    ----------
    sat : int
        Satellite number (from 1-37)

    Returns
    -------
    b1 : int
        bit 1 for xor step in PRN sequence generator
    b2 : int
        bit 2 for xor step in PRN sequence generator

    See Also
    --------
    dstauffman.aerospace.correlate_prn, dstauffman.aerospace.generate_prn

    Notes
    -----
    #.  Written by David C. Stauffer for AA272C in Jan 2009 in MATLAB.
    #.  Translated by David C. Stauffer in July 2021 into Python.

    Examples
    --------
    >>> from dstauffman.aerospace import get_prn_bits
    >>> (b1, b2) = get_prn_bits(19)
    >>> print(b1)
    3

    >>> print(b2)
    6

    """
    prn_bits: dict[int, tuple[int, int]] = {}
    prn_bits[1] = (2, 6)
    prn_bits[2] = (3, 7)
    prn_bits[3] = (4, 8)
    prn_bits[4] = (5, 9)
    prn_bits[5] = (1, 9)
    prn_bits[6] = (2, 10)
    prn_bits[7] = (1, 8)
    prn_bits[8] = (2, 9)
    prn_bits[9] = (3, 10)
    prn_bits[10] = (2, 3)
    prn_bits[11] = (3, 4)
    prn_bits[12] = (5, 6)
    prn_bits[13] = (6, 7)
    prn_bits[14] = (7, 8)
    prn_bits[15] = (8, 9)
    prn_bits[16] = (9, 10)
    prn_bits[17] = (1, 4)
    prn_bits[18] = (2, 5)
    prn_bits[19] = (3, 6)
    prn_bits[20] = (4, 7)
    prn_bits[21] = (5, 8)
    prn_bits[22] = (6, 9)
    prn_bits[23] = (1, 3)
    prn_bits[24] = (4, 6)
    prn_bits[25] = (5, 7)
    prn_bits[26] = (6, 8)
    prn_bits[27] = (7, 9)
    prn_bits[28] = (8, 10)
    prn_bits[29] = (1, 6)
    prn_bits[30] = (2, 7)
    prn_bits[31] = (3, 8)
    prn_bits[32] = (4, 9)
    prn_bits[33] = (5, 10)
    prn_bits[34] = (4, 10)
    prn_bits[35] = (1, 7)
    prn_bits[36] = (2, 8)
    prn_bits[37] = (4, 10)
    if sat not in prn_bits:
        raise ValueError(f'Unexpected satellite number: "{sat}"')
    return prn_bits[sat]


# %% Functions - correlate_prn
def correlate_prn(prn1: _I, prn2: _I, shift: int | _I, form: Literal["zero-one", "one-one"]) -> _N:
    r"""
    Correlates two PRN codes with an optional shift.

    Parameters
    ----------
    prn1  : class numpy.random.Generator
        psuedo-random number stream 1
    prn2  : class numpy.random.Generator
        psuedo-random number stream 2
    shift : int
        bit shift between codes
    form  : str, from {"zero-one", "one-one"}
        Whether using values of 0 to 1, or 1 to -1

    Returns
    -------
    cor   : float
        correlation between two given PRNs

    See Also
    --------
    dstauffman.aerospace.generate_prn, dstauffman.aerospace.get_prn_bits

    Notes
    -----
    #.  Calling with the same PRN twice will give the auto-correlation.
    #.  Written by David C. Stauffer in Jan 2009.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import generate_prn, correlate_prn
    >>> import numpy as np
    >>> prn   = generate_prn(1)
    >>> shift = np.arange(1023)
    >>> form  = "zero-one"
    >>> cor   = correlate_prn(prn, prn, shift, form)
    >>> assert(cor[0] == 1)
    >>> assert(np.max(np.abs(cor[1:])) < 0.1)

    """
    # process inputs based on form
    if form == "zero-one":
        # change PRNs from (0, 1) to (1, -1)
        prn1 = 1 * (prn1 == 0) + -1 * (prn1 == 1)
        prn2 = 1 * (prn2 == 0) + -1 * (prn2 == 1)
    elif form == "one-one":
        pass
    else:
        _assert_never(form)
    shift = np.asanyarray(shift)

    cor = np.zeros(shift.shape)
    # loop through different shift values
    for i in range(shift.size):
        # shift prn2
        prn2s = bsr(prn2, shift[i])

        # initialize output
        temp = np.sum(prn1 * prn2s)

        # scale correlation by number of samples
        cor[i] = temp / 1023
    return cor


# %% Functions - generate_prn
def generate_prn(sat: int, length: int = 1023) -> _I:
    r"""
    Generates the prn bit stream.

    Parameters
    ----------
    sat : int
        satellite number (from 1-37)
    length : int, optional, default is 1023
        length specification

    Returns
    -------
    prn : (N, ) ndarray
        psuedo-random number for specified satellite

    See Also
    --------
    dstauffman.aerospace.correlate_prn, dstauffman.aerospace.get_prn_bits

    Notes
    -----
    #.  Written by David C. Stauffer in Jan 2009.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import generate_prn
    >>> prn = generate_prn(1)
    >>> assert(np.all((prn == 0) | (prn == 1)))

    """

    def bplus(x: _I) -> _I:
        r"""Does modulo 2 addition (exclusive or) on vector input."""
        return np.mod(np.sum(x), 2)  # type: ignore[no-any-return]

    # find which bits to mod based on the satellite number
    (bit1, bit2) = get_prn_bits(sat)

    # initialize generators
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)

    # initialize output
    prn = np.zeros(length, dtype=int)

    # loop through bits
    for i in range(length):
        # calculate new values for generators
        g1n = bplus(g1[np.array([2, 9])])
        g2n = bplus(g2[np.array([1, 2, 5, 7, 8, 9])])
        g2i = bplus(g2[np.array([bit1 - 1, bit2 - 1])])

        # calculate output bit and append to PRN
        xgi = bplus(np.array([g1[9] - 1, g2i - 1]))
        prn[i] = xgi

        # shift generators
        g1 = np.hstack((g1n, g1[:-1]))
        g2 = np.hstack((g2n, g2[:-1]))

    return prn


# %% Functions - gps_to_datetime
@overload
def gps_to_datetime(week: int | _I, time: float | _I | _N) -> datetime.datetime | list[datetime.datetime]: ...
@overload
def gps_to_datetime(
    week: int | _I, time: float | _I | _N, form: Literal["datetime"] = ...
) -> datetime.datetime | list[datetime.datetime]: ...
@overload
def gps_to_datetime(week: int | _I, time: float | _I | _N, form: Literal["numpy"]) -> np.datetime64 | _D: ...
def gps_to_datetime(
    week: int | _I, time: float | _I | _N, form: Literal["datetime", "numpy"] = "datetime"
) -> datetime.datetime | list[datetime.datetime] | np.datetime64 | _D:
    r"""
    Converts a GPS week and time to a Python datetime.

    Calculates the gps date based on the elasped number of weeks using the
    built-in MATLAB datenum abilities, then applies the time in seconds.

    Parameters
    ----------
    week : (N, ) ndarray
        GPS week [week]
    time : (N, ) ndarray
        GPS time of week [sec]
    form : str, optional, default is "datetime", from {"datetime", "numpy"}
        form of the date to return

    Returns
    -------
    date_out : class datetime.datetime
        UTC date

    See Also
    --------
    datetime.datetime

    Notes
    -----
    #.  GPS week zero = Jan 06, 1980 at midnight.
    #.  Written by David C. Stauffer in Apr 2011.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import gps_to_datetime
    >>> import datetime
    >>> import numpy as np
    >>> week     = np.array([1782, 1783])
    >>> time     = np.array([425916, 4132])
    >>> date_gps = gps_to_datetime(week, time)
    >>> assert(date_gps[0] == datetime.datetime(2014, 3, 6, 22, 18, 36))
    >>> assert(date_gps[1] == datetime.datetime(2014, 3, 9,  1,  8, 52))

    """
    week = np.asanyarray(week)
    time = np.asanyarray(time)
    # if week is less than 1024, then assume it has rollovers that put it in the correct 20 year period
    # based on the date returned by the 'now' command.
    ix = week < WEEK_ROLLOVER
    if np.any(ix):
        num_rollovers = np.floor((datetime.datetime.now() - GPS_DATE_ZERO).days / (DAYS_PER_WEEK * WEEK_ROLLOVER)).astype(int)
        week[ix] += num_rollovers * WEEK_ROLLOVER

    # GPS start week
    date_gps: datetime.datetime | list[datetime.datetime] | np.datetime64
    if form == "datetime":
        # fmt: off
        if np.size(week) == 1:
            start_week = GPS_DATE_ZERO + datetime.timedelta(days=int(DAYS_PER_WEEK * week))
            whole_sec  = int(time)
            micros     = round(1e6 * (time - whole_sec))  # type: ignore[call-overload]
            date_gps   = start_week + datetime.timedelta(seconds=whole_sec, microseconds=micros)
        else:
            date_gps = []
            for (w, t) in zip(week, time):
                start_week = GPS_DATE_ZERO + datetime.timedelta(days=int(DAYS_PER_WEEK * w))
                whole_sec  = int(t)
                micros     = round(1e6 * (t - whole_sec))
                date_gps.append(start_week + datetime.timedelta(seconds=whole_sec, microseconds=micros))
        # fmt: on
    elif form == "numpy":
        assert NP_GPS_DATE_ZERO is not None
        start_week = NP_GPS_DATE_ZERO + DAYS_PER_WEEK * week * NP_ONE_DAY  # type: ignore[assignment]
        date_gps = start_week + time * NP_ONE_SECOND  # type: ignore[operator]
    else:
        _assert_never(form)
    return date_gps


# %% Functions - get_gps_to_utc_offset
def get_gps_to_utc_offset(days_since_gps_date_zero: float | _I | _N) -> _N:
    """
    Calculate the GPS to UTC offset given the (fractional) number of days since GPS origin.

    Parameters
    ----------
    days_since_gps_date_zero: (N, ) ndarray
        Days (in continuous GPS reference, since GPS Date Zero (1980-Jan-06))

    Returns
    -------
    gps_to_utc_offset

    Change Log
    ----------
    #.  Functionalize separately by David C. Stauffer in August 2024 and expanded to include
        all leap from the start of GPS epoch 0. Also fixed bug where later results where
        wiping earlier ones.

    References
    ----------
    Recent Leap Seconds
    | Date        |     JD    |   MJD   |DAYS_GPS0| TAI-UTC|GPS-UTC|
    | 1980 JAN 01 | 2444239.5 | 44239.0 |      0  |  19.0  |   0  |
    | 1981 JUL 01 | 2444786.5 | 44786.0 |    542  |  20.0  |   1  |
    | 1982 JUL 01 | 2445151.5 | 45151.0 |    907  |  21.0  |   2  |
    | 1983 JUL 01 | 2445516.5 | 45516.0 |   1272  |  22.0  |   3  |
    | 1985 JUL 01 | 2446247.5 | 46247.0 |   2003  |  23.0  |   4  |
    | 1988 JAN 01 | 2447161.5 | 47161.0 |   2917  |  24.0  |   5  |
    | 1990 JAN 01 | 2447892.5 | 47892.0 |   3648  |  25.0  |   6  |
    | 1991 JAN 01 | 2448257.5 | 48257.0 |   4013  |  26.0  |   7  |
    | 1992 JUL 01 | 2448804.5 | 48804.0 |   4560  |  27.0  |   8  |
    | 1993 JUL 01 | 2449169.5 | 49169.0 |   4925  |  28.0  |   9  |
    | 1994 JUL 01 | 2449534.5 | 49534.0 |   5290  |  29.0  |  10  |
    | 1996 JAN 01 | 2450083.5 | 50083.0 |   5839  |  30.0  |  11  |
    | 1997 JUL 01 | 2450630.5 | 50630.0 |   6386  |  31.0  |  12  |
    | 1999 JAN 01 | 2451179.5 | 51179.0 |   6935  |  32.0  |  13  |
    | 2006 JAN 01 | 2453736.5 | 53736.0 |   9492  |  33.0  |  14  |
    | 2009 JAN 01 | 2454832.5 | 54832.0 |  10588  |  34.0  |  15  |
    | 2012 JUL 01 | 2456109.5 | 56109.0 |  11865  |  35.0  |  16  |
    | 2015 JUL 01 | 2457204.5 | 57204.0 |  12960  |  36.0  |  17  |
    | 2017 JAN 01 | 2457754.5 | 57754.0 |  13510  |  37.0  |  18  |

    Examples
    --------
    >>> from dstauffman.aerospace import get_gps_to_utc_offset
    >>> gps_to_utc_offset = get_gps_to_utc_offset(11111)
    >>> assert gps_to_utc_offset == -15

    """
    assert np.all(days_since_gps_date_zero >= 0), "Days since origin must be positive."
    # get the days since GPS origin that leap seconds are added
    day_bounds = [
        0, 542, 907, 1272, 2003, 2917, 3648, 4013, 4560, 4925, 5290, 5839, 6386, 6935, 9492, 10588,
        11865, 12960, 13510, 2**31,
    ]  # fmt: skip
    partial_days = np.arange(len(day_bounds)) / ONE_DAY
    leap_seconds = np.arange(0, -19, -1, dtype=int)
    gps_day_bounds = day_bounds + partial_days
    this_bin = np_digitize(days_since_gps_date_zero, gps_day_bounds)
    gps_to_utc_offset = leap_seconds[this_bin]
    return gps_to_utc_offset  # type: ignore[return-value]


# %% Functions - gps_to_utc_datetime
@overload
def gps_to_utc_datetime(week: int | _I, time: float | _I | _N) -> datetime.datetime | list[datetime.datetime]: ...
@overload
def gps_to_utc_datetime(
    week: int | _I,
    time: float | _I | _N,
    gps_to_utc_offset: int | _I | None,
    form: Literal["datetime"] = ...,
) -> datetime.datetime | list[datetime.datetime]: ...
@overload
def gps_to_utc_datetime(
    week: int | _I, time: float | _I | _N, gps_to_utc_offset: int | _I | None, form: Literal["numpy"]
) -> np.datetime64 | _D: ...
def gps_to_utc_datetime(
    week: int | _I,
    time: float | _I | _N,
    gps_to_utc_offset: int | _I | None = None,
    form: Literal["datetime", "numpy"] = "datetime",
) -> datetime.datetime | list[datetime.datetime] | np.datetime64 | _D:
    r"""
    Converts a GPS week and time to UTC time as a datetime.

    Calculates the gps date based on the elasped number of weeks using the
    built-in datetime abilities, then applies the time in seconds,
    then applies the GPS offset and returns the answer as a datetime or numpy.datetime64.

    Parameters
    ----------
    week : (N, ) ndarray
        GPS week [week]
    time : (N, ) ndarray
        GPS time of week [sec]
    gps_to_utc_offset : int, optional
        gps to UTC leap second correction [sec]
    form : str, optional, default is "datetime", from {"datetime", "numpy"}
        form of the date to return

    Returns
    -------
    date_out : class datetime.datetime
        UTC date

    See Also
    --------
    datetime.datetime, dstauffman.aerospace.gps_to_datetime

    Notes
    -----
    #.  GPS week zero = Jan 06, 1980 at midnight.
    #.  Written by David C. Stauffer in March 2011.
    #.  Updated by David C. Stauffer in Apr 2011 to include leap seconds since J2000.
    #.  Updated by David C. Stauffer to be current through 2017, and incorporated into matspace
        library.
    #.  Updated by David C. Stauffer in July 2018, based on bug found by Chinh Tran to add leap
        second at GPS midnight rather than UTC midnight.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import gps_to_utc_datetime
    >>> import datetime
    >>> import numpy as np
    >>> week     = np.array([1782, 1783])
    >>> time     = np.array([425916, 4132])
    >>> date_utc = gps_to_utc_datetime(week, time)
    >>> assert(date_utc[0] == datetime.datetime(2014, 3, 6, 22, 18, 20))
    >>> assert(date_utc[1] == datetime.datetime(2014, 3, 9,  1,  8, 36))

    """
    week = np.asanyarray(week)
    time = np.asanyarray(time)
    # if week is less than 1024, then assume it has rollovers that put it in the correct 20 year period
    # based on the date returned by the 'now' command.
    ix = week < WEEK_ROLLOVER
    if np.any(ix):
        num_rollovers = np.floor((datetime.datetime.now() - GPS_DATE_ZERO).days / (DAYS_PER_WEEK * WEEK_ROLLOVER)).astype(int)
        week[ix] += num_rollovers * WEEK_ROLLOVER

    # check for optional inputs
    if gps_to_utc_offset is None:
        days_since_date_zero = week * DAYS_PER_WEEK + time / ONE_DAY
        gps_to_utc_offset = get_gps_to_utc_offset(days_since_date_zero)  # type: ignore[assignment]
    gps_to_utc_offset = np.asanyarray(gps_to_utc_offset)

    # GPS start week
    date_utc: datetime.datetime | list[datetime.datetime] | np.datetime64 | _D
    if form == "datetime":
        # fmt: off
        if np.size(week) == 1:
            start_week = GPS_DATE_ZERO + datetime.timedelta(days=int(DAYS_PER_WEEK * week))
            frac_sec   = time + gps_to_utc_offset
            whole_sec  = int(frac_sec)
            micros     = round(1e6 * (frac_sec - whole_sec))  # type: ignore[call-overload]
            date_utc   = start_week + datetime.timedelta(seconds=whole_sec, microseconds=micros)
        else:
            date_utc = []
            for (w, t, off) in zip(week, time, gps_to_utc_offset):
                start_week = GPS_DATE_ZERO + datetime.timedelta(days=int(DAYS_PER_WEEK * w))
                frac_sec   = t + off
                whole_sec  = int(frac_sec)
                micros     = round(1e6 * (frac_sec - whole_sec))
                date_utc.append(start_week + datetime.timedelta(seconds=whole_sec, microseconds=micros))
        # fmt: on
    elif form == "numpy":
        assert NP_GPS_DATE_ZERO is not None
        start_week_np = NP_GPS_DATE_ZERO + DAYS_PER_WEEK * week * NP_ONE_DAY
        date_utc = start_week_np + (time + gps_to_utc_offset) * NP_ONE_SECOND
    else:
        raise ValueError(f'Unexpected value for form: "{form}".')
    return date_utc


# %% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_gps", exit=False)
    doctest.testmod(verbose=False)
