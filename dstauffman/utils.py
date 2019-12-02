# -*- coding: utf-8 -*-
r"""
Generic utilities that can be independently defined and used by other modules.

Notes
-----
#.  By design, this module does not reference any other piece of the dstauffman code base except
        constants to avoid circular references.
#.  Written by David C. Stauffer in March 2015.

"""

#%% Imports
import doctest
import inspect
import logging
import os
import sys
import unittest
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import StringIO

import numpy as np

from dstauffman.constants import MONTHS_PER_YEAR

#%% Globals
logger = logging.getLogger(__name__)

#%% Functions - _nan_equal
def _nan_equal(a, b):
    r"""
    Test ndarrays for equality, but ignore NaNs.

    Parameters
    ----------
    a : ndarray
        Array one
    b : ndarray
        Array two

    Returns
    -------
    is_same : bool
        Flag for whether the inputs are the same or not

    Examples
    --------
    >>> from dstauffman.utils import _nan_equal
    >>> import numpy as np
    >>> a = np.array([1, 2, np.nan])
    >>> b = np.array([1, 2, np.nan])
    >>> print(_nan_equal(a, b))
    True

    >>> a = np.array([1, 2, np.nan])
    >>> b = np.array([3, 2, np.nan])
    >>> print(_nan_equal(a, b))
    False

    """
    # preallocate to True
    is_same = True
    try:
        # use numpy testing module to assert that they are equal (ignores NaNs)
        np.testing.assert_array_equal(a, b)
    except AssertionError:
        # if assertion fails, then they are not equal
        is_same = False
    return is_same

#%% Functions - rms
def rms(data, axis=None, keepdims=False, ignore_nans=False):
    r"""
    Calculate the root mean square of a number series.

    Parameters
    ----------
    data : array_like
        input data
    axis : int, optional
        Axis along which RMS is computed. The default is to compute the RMS of the flattened array.
    keepdims : bool, optional
        If true, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original `data`.

    Returns
    -------
    out : ndarray
        RMS results

    See Also
    --------
    numpy.mean, numpy.nanmean, numpy.conj, numpy.sqrt

    Notes
    -----
    #.  Written by David C. Stauffer in Mar 2015.

    Examples
    --------
    >>> from dstauffman import rms
    >>> rms([0, 1, 0., -1])
    0.7071067811865476

    """
    # check for empty data
    if not np.isscalar(data) and len(data) == 0:
        return np.nan
    # do the root-mean-square, but use x * conj(x) instead of square(x) to handle complex numbers correctly
    if not ignore_nans:
        out = np.sqrt(np.mean(data * np.conj(data), axis=axis, keepdims=keepdims))
    else:
        # check for all NaNs case
        if np.all(np.isnan(data)):
            out = np.nan
        else:
            out = np.sqrt(np.nanmean(data * np.conj(data), axis=axis, keepdims=keepdims))
    # return the result
    return out

#%% Functions - rss
def rss(data, axis=None, keepdims=False, ignore_nans=False):
    r"""
    Calculate the root sum square of a number series.

    Parameters
    ----------
    data : array_like
        input data
    axis : int, optional
        Axis along which RMS is computed. The default is to compute the RMS of the flattened array.
    keepdims : bool, optional
        If true, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original `data`.

    Returns
    -------
    out : ndarray
        RSS results

    See Also
    --------
    numpy.sum, numpy.nansum, numpy.conj

    Notes
    -----
    #.  Written by David C. Stauffer in April 2016.

    Examples
    --------
    >>> from dstauffman import rss
    >>> rss([0, 1, 0., -1])
    2.0

    """
    # check for empty data
    if not np.isscalar(data) and len(data) == 0:
        return np.nan
    # do the root-mean-square, but use x * conj(x) instead of square(x) to handle complex numbers correctly
    if not ignore_nans:
        out = np.sum(data * np.conj(data), axis=axis, keepdims=keepdims)
    else:
        # check for all NaNs case
        if np.all(np.isnan(data)):
            out = np.nan
        else:
            out = np.nansum(data * np.conj(data), axis=axis, keepdims=keepdims)
    # return the result
    return out

#%% Functions - setup_dir
def setup_dir(folder, rec=False):
    r"""
    Clear the contents for existing folders or instantiates the directory if it doesn't exist.

    Parameters
    ----------
    folder : str
        giving the fullpath location of the folder to empty or instantiate.
    rec : {False, True}, optional
        whether to recursively delete contents.

    See Also
    --------
    os.makedirs, os.rmdir, os.remove

    Raises
    ------
    RuntimeError
        Problems creating or deleting a file or folder, likely due to permission issues.

    Notes
    -----
    #.  Written by David C. Stauffer in Feb 2015.

    Examples
    --------
    >>> from dstauffman import setup_dir
    >>> setup_dir(r'C:\Temp\test_folder') # doctest: +SKIP

    """
    # check for an empty string and exit
    if not folder:
        return
    if os.path.isdir(folder):
        # Loop through the contained files/folders
        for this_elem in os.listdir(folder):
            # alias the fullpath of this file element
            this_full_elem = os.path.join(folder, this_elem)
            # check if a folder or file
            if os.path.isdir(this_full_elem):
                # if a folder, then delete recursively if rec is True
                if rec:
                    setup_dir(this_full_elem)
                    os.rmdir(this_full_elem)
            elif os.path.isfile(this_full_elem):
                # if a file, then remove it
                os.remove(this_full_elem)
            else:
                raise RuntimeError('Unexpected file type, neither file nor folder: "{}".'\
                    .format(this_full_elem)) # pragma: no cover
        logger.info('Files/Sub-folders were removed from: "' + folder + '"')
    else:
        # create directory if it does not exist
        try:
            os.makedirs(folder)
            logger.info('Created directory: "' + folder + '"')
        except: # pragma: no cover
            # re-raise last exception, could try to handle differently in the future
            raise # pragma: no cover

#%% Functions - compare_two_classes
def compare_two_classes(c1, c2, suppress_output=False, names=None, ignore_callables=True, compare_recursively=True):
    r"""
    Compare two classes by going through all their public attributes and showing that they are equal.

    Parameters
    ----------
    c1 : class object
        Any class object
    c2 : class object
        Any other class object
    suppress_output : bool, optional
        If True, suppress the information printed to the screen, defaults to False.
    names : list of str, optional
        List of the names to be printed to the screen for the two input classes.
    ignore_callables : bool, optional
        If True, ignore differences in callable attributes (i.e. methods), defaults to True.

    Returns
    -------
    is_same : bool
        True/False flag for whether the two class are the same.

    Examples
    --------
    >>> from dstauffman import compare_two_classes
    >>> c1 = type('Class1', (object, ), {'a': 0, 'b' : '[1, 2, 3]', 'c': 'text'})
    >>> c2 = type('Class2', (object, ), {'a': 0, 'b' : '[1, 2, 4]', 'd': 'text'})
    >>> is_same = compare_two_classes(c1, c2)
    b is different from c1 to c2.
    c is only in c1.
    d is only in c2.
    "c1" and "c2" are not the same.

    """
    def _not_true_print():
        r"""Set is_same to False and optionally prints information to the screen."""
        is_same = False
        if not suppress_output:
            print('{} is different from {} to {}.'.format(this_attr, name1, name2))
        return is_same

    def _is_function(obj):
        r"""Determine whether the object is a function or not."""
        # need second part for Python compatibility for v2.7, which distinguishes unbound methods from functions.
        return inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj)

    def _is_class_instance(obj):
        r"""Determine whether the object is an instance of a class or not."""
        return hasattr(obj, '__dict__') and not _is_function(obj) # and hasattr(obj, '__call__')

    # preallocate answer to True until proven otherwise
    is_same = True
    # get names if specified
    if names is not None:
        name1 = names[0]
        name2 = names[1]
    else:
        name1 = 'c1'
        name2 = 'c2'
    # simple test
    if c1 is not c2:
        # get the list of public attributes
        attrs1 = set((name for name in dir(c1) if not name.startswith('_')))
        attrs2 = set((name for name in dir(c2) if not name.startswith('_')))
        # compare the attributes that are in both
        same = attrs1 & attrs2
        for this_attr in sorted(same):
            # alias the attributes
            attr1 = getattr(c1, this_attr)
            attr2 = getattr(c2, this_attr)
            # determine if this is a subclass
            if _is_class_instance(attr1):
                if _is_class_instance(attr2):
                    if compare_recursively:
                        names = [name1 + '.' + this_attr, name2 + '.' + this_attr]
                        # Note: don't want the 'and' to short-circuit, so do the 'and is_same' last
                        if isinstance(attr1, dict) and isinstance(attr2, dict):
                            is_same = compare_two_dicts(attr1, attr2, suppress_output=suppress_output, names=names) and is_same
                        else:
                            is_same = compare_two_classes(attr1, attr2, suppress_output=suppress_output, \
                                names=names, ignore_callables=ignore_callables, \
                                compare_recursively=compare_recursively) and is_same
                        continue
                    else:
                        continue # pragma: no cover (actually covered, optimization issue)
                else:
                    is_same = _not_true_print()
                    continue
            else:
                if _is_class_instance(attr2):
                    is_same = _not_true_print()
            if _is_function(attr1) or _is_function(attr2):
                if ignore_callables:
                    continue # pragma: no cover (actually covered, optimization issue)
                else:
                    is_same = _not_true_print()
                    continue
            # if any differences, then this test fails
            if isinstance(attr1, Mapping) and isinstance(attr2, Mapping):
                is_same = compare_two_dicts(attr1, attr2, suppress_output=True) and is_same
            elif np.logical_not(_nan_equal(attr1, attr2)):
                is_same = _not_true_print()
        # find the attributes in one but not the other, if any, then this test fails
        diff = attrs1 ^ attrs2
        for this_attr in sorted(diff):
            is_same = False
            if not suppress_output:
                if this_attr in attrs1:
                    print(this_attr + ' is only in ' + name1 + '.')
                else:
                    print(this_attr + ' is only in ' + name2 + '.')
    # display results
    if not suppress_output:
        if is_same:
            print('"' + name1 + '" and "' + name2 + '" are the same.')
        else:
            print('"' + name1 + '" and "' + name2 + '" are not the same.')
    return is_same

#%% Functions - compare_two_dicts
def compare_two_dicts(d1, d2, suppress_output=False, names=None):
    r"""
    Compare two dictionaries for the same keys, and the same value of those keys.

    Parameters
    ----------
    d1 : class object
        Any class object
    d2 : class object
        Any other class object
    suppress_output : bool, optional
        If True, suppress the information printed to the screen, defaults to False.
    names : list of str, optional
        List of the names to be printed to the screen for the two input classes.

    Returns
    -------
    is_same : bool
        True/False flag for whether the two class are the same.

    Examples
    --------
    >>> from dstauffman import compare_two_dicts
    >>> d1 = {'a': 1, 'b': 2, 'c': 3}
    >>> d2 = {'a': 1, 'b': 5, 'd': 6}
    >>> is_same = compare_two_dicts(d1, d2)
    b is different.
    c is only in d1.
    d is only in d2.
    "d1" and "d2" are not the same.

    """
    # preallocate answer to True until proven otherwise
    is_same = True
    # get names if specified
    if names is not None:
        name1 = names[0]
        name2 = names[1]
    else:
        name1 = 'd1'
        name2 = 'd2'
    # simple test
    if d1 is not d2:
        # compare the keys that are in both
        same = set(d1) & set(d2)
        for key in sorted(same):
            # if any differences, then this test fails
            if np.logical_not(_nan_equal(d1[key], d2[key])):
                is_same = False
                if not suppress_output:
                    print(key + ' is different.')
        # find keys in one but not the other, if any, then this test fails
        diff = set(d1) ^ set(d2)
        for key in sorted(diff):
            is_same = False
            if not suppress_output:
                if key in d1:
                    print(key + ' is only in ' + name1 + '.')
                else:
                    print(key + ' is only in ' + name2 + '.')
    # display results
    if not suppress_output:
        if is_same:
            print('"' + name1 + '" and "' + name2 + '" are the same.')
        else:
            print('"' + name1 + '" and "' + name2 + '" are not the same.')
    return is_same

#%% Functions - round_time
def round_time(dt=None, round_to_sec=60):
    r"""
    Round a datetime object to any time lapse in seconds.

    Parameters
    ----------
    dt           : datetime.datetime
        time to round, default now.
    round_to_sec : int
        Closest number of seconds to round to, default 60 seconds (i.e. rounds to nearest minute)

    See Also
    --------
    datetime.datetime

    Notes
    -----
    #. Originally written by Thierry Husson 2012.  Freely distributed.
    #. Adapted by David C. Stauffer in Feb 2015.

    Examples
    --------
    >>> from dstauffman import round_time
    >>> from datetime import datetime
    >>> dt = datetime(2015, 3, 13, 8, 4, 10)
    >>> rounded_time = round_time(dt)
    >>> print(rounded_time)
    2015-03-13 08:04:00

    """
    # set default for dt
    if dt is None:
        dt = datetime.now()
    # get the current elasped time in seconds
    seconds = (dt - dt.min).seconds
    # round to the nearest whole second
    rounding = (seconds+round_to_sec/2) // round_to_sec * round_to_sec
    # return the rounded result
    return dt + timedelta(0, rounding-seconds, -dt.microsecond)

#%% Functions - read_text_file
def read_text_file(filename):
    r"""
    Open and read a complete text file.

    Parameters
    ----------
    filename : str
        fullpath name of the file to read

    Returns
    -------
    text : str
        text of the desired file

    Raises
    ------
    RuntimeError
        If unable to open, or unable to read file.

    See Also
    --------
    write_text_file, open

    Examples
    --------
    >>> from dstauffman import read_text_file, write_text_file, get_tests_dir
    >>> import os
    >>> text = 'Hello, World\n'
    >>> filename = os.path.join(get_tests_dir(), 'temp_file.txt')
    >>> write_text_file(filename, text)
    >>> text2 = read_text_file(os.path.join(get_tests_dir(), 'temp_file.txt'))
    >>> print(text2)
    Hello, World
    <BLANKLINE>

    >>> os.remove(filename)

    """
    try:
        # open file for reading
        with open(filename, 'rt') as file:
            # read file
            text = file.read() # pragma: no branch
        # return results
        return text
    except:
        # on any exceptions, print a message and re-raise the error
        print('Unable to open file "{}" for reading.'.format(filename))
        raise

#%% Functions - write_text_file
def write_text_file(filename, text):
    r"""
    Open and write the specified text to a file.

    Parameters
    ----------
    filename : str
        fullpath name of the file to read
    text : str
        text to be written to the file

    Raises
    ------
    RuntimeError
        If unable to open, or unable to write file.

    See Also
    --------
    open_text_file, open

    Examples
    --------
    >>> from dstauffman import write_text_file, get_tests_dir
    >>> import os
    >>> text = 'Hello, World\n'
    >>> filename = os.path.join(get_tests_dir(), 'temp_file.txt')
    >>> write_text_file(filename, text)

    >>> os.remove(filename)

    """
    try:
        # open file for writing
        with open(filename, 'wt') as file:
            # write file
            file.write(text) # pragma: no branch
    except:
        # on any exceptions, print a message and re-raise the error
        print('Unable to open file "{}" for writing.'.format(filename))
        raise

#%% Functions - capture_output
@contextmanager
def capture_output(mode='out'):
    r"""
    Capture the stdout and stderr streams instead of displaying to the screen.

    Parameters
    ----------
    mode : str
        Mode to use when capturing output
            'out' captures just sys.stdout
            'err' captures just sys.stderr
            'all' captures both sys.stdout and sys.stderr

    Returns
    -------
    out : class StringIO
        stdout stream output
    err : class StringIO
        stderr stream output

    Examples
    --------
    >>> from dstauffman import capture_output
    >>> with capture_output() as out:
    ...     print('Hello, World!')
    >>> output = out.getvalue().strip()
    >>> out.close()
    >>> print(output)
    Hello, World!

    """
    # alias modes
    capture_out = True if mode == 'out' or mode == 'all' else False
    capture_err = True if mode == 'err' or mode == 'all' else False
    # create new string buffers
    new_out, new_err = StringIO(), StringIO()
    # alias the old string buffers for restoration afterwards
    old_out, old_err = sys.stdout, sys.stderr
    try:
        # override the system buffers with the new ones
        if capture_out:
            sys.stdout = new_out
        if capture_err:
            sys.stderr = new_err
        # yield results as desired
        if mode == 'out':
            yield sys.stdout
        elif mode == 'err':
            yield sys.stderr
        elif mode == 'all':
            yield sys.stdout, sys.stderr
    finally:
        # restore the original buffers once all results are read
        sys.stdout, sys.stderr = old_out, old_err

#%% Functions - unit
def unit(data, axis=None):
    r"""
    Normalize a matrix into unit vectors along a specified dimension.

    Default to column normalization.

    Parameters
    ----------
    data : ndarray
        Data
    axis : int, optional
        Axis upon which to normalize, defaults to last axis

    Returns
    -------
    norm_data : ndarray
        Normalized data

    See Also
    --------
    sklearn.preprocessing.normalize

    Notes
    -----
    #.  Written by David C. Stauffer in May 2015.

    Examples
    --------
    >>> from dstauffman import unit
    >>> import numpy as np
    >>> data = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 1]])
    >>> norm_data = unit(data, axis=0)
    >>> print(norm_data) # doctest: +NORMALIZE_WHITESPACE
    [[ 1. 0. -0.70710678]
     [ 0. 0.  0.        ]
     [ 0. 0.  0.70710678]]

    """
    if axis is None:
        axis = data.ndim - 1
    if axis >= data.ndim:
        raise ValueError('axis {} is out of bounds for array of dimension {}'.format(axis, data.ndim))
    # calculate the magnitude of each vector
    mag = np.atleast_1d(np.sqrt(np.sum(data * np.conj(data), axis=axis)))
    # check for zero vectors, and replace magnitude with 1 to make them unchanged
    mag[mag == 0] = 1
    # calculate the new normalized data
    norm_data = data / mag
    return norm_data

#%% modd
def modd(x1, x2, out=None):
    r"""
    Return element-wise remainder of division, except that instead of zero it gives the divisor instead.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and it must be of the right
        shape to hold the output. See doc.ufuncs.

    Returns
    -------
    y : ndarray
        The remainder of the quotient x1/x2, element-wise. Returns a scalar if both x1 and x2 are
        scalars.  Replaces what would be zeros in the normal modulo command with the divisor instead.

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

    Examples
    --------
    >>> from dstauffman import modd
    >>> import numpy as np
    >>> x1 = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    >>> x2 = 4
    >>> y = modd(x1, x2)
    >>> print(y)
    [4 1 2 3 4 1 2 3 4]

    """
    x1 = np.asanyarray(x1)
    if out is None:
        y = np.mod(x1 - 1, x2) + 1
        return y
    else:
        np.mod(x1 - 1, x2, out)
        np.add(out, 1, out) # needed to force add to be inplace operation

#%% np_digitize
def np_digitize(x, bins, right=False):
    r"""
    Act as a wrapper to the numpy.digitize function with customizations.

    The customizations include additional error checks, and bins starting from 0 instead of 1.

    Parameters
    ----------
    x : array_like
        Input array to be binned.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is open in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    out : ndarray of ints
        Output array of indices, of same shape as `x`.

    Raises
    ------
    ValueError
        If `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.digitize

    Notes
    -----
    #.  This function is equilavent to the MATLAB `discretize` function.

    Examples
    --------
    >>> from dstauffman import np_digitize
    >>> import numpy as np
    >>> x    = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> out  = np_digitize(x, bins)
    >>> print(out)
    [0 3 2 1]

    """
    # hard-coded values
    precision = 0 * 1e-13 # TODO: do I need a precision here (or only if bins are not ints?)

    # allow an empty x to pass through just fine
    if x.size == 0:
        return np.array([], dtype=int)

    # check for NaNs
    if np.any(np.isnan(x)):
        raise ValueError('Some values were NaN.')

    # check the bounds
    if right:
        if np.any(x < bins[0]-precision) or np.any(x >= bins[-1]+precision):
            raise ValueError('Some values of x are outside the given bins.')
    else:
        if np.any(x <= bins[0]-precision) or np.any(x > bins[-1]+precision):
            raise ValueError('Some values of x are outside the given bins.')

    # do the calculations by calling the numpy command and shift results by one
    out = np.digitize(x, bins, right) - 1
    return out

#%% histcounts
def histcounts(x, bins, right=False):
    r"""
    Count the number of points in each of the given bins.

    Parameters
    ----------
    x : array_like
        Input array to be binned.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is open in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    hist : ndarray of ints
        Output array the number of points in each bin

    See Also
    --------
    numpy.digitize, np_digitize

    Notes
    -----
    #.  This function is equilavent to the MATLAB `histcounts` function.

    Examples
    --------
    >>> from dstauffman import histcounts
    >>> import numpy as np
    >>> x    = np.array([0.2, 6.4, 3.0, 1.6, 0.5])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> hist = histcounts(x, bins)
    >>> print(hist)
    [2 1 1 1]

    """
    # get the bin number that each point is in
    ix_bin = np_digitize(x, bins, right=right)
    # count the number in each bin
    hist = np.bincount(ix_bin, minlength=len(bins)-1)
    return hist

#%% full_print
@contextmanager
def full_print():
    r"""
    Context manager for printing full numpy arrays.

    Notes
    -----
    #.  Adapted by David C. Stauffer in January 2017 from a stackover flow answer by Paul Price,
        given here: http://stackoverflow.com/questions/1987694/print-the-full-numpy-array

    Examples
    --------
    >>> from dstauffman import full_print
    >>> import numpy as np
    >>> temp_options = np.get_printoptions()
    >>> np.set_printoptions(threshold=10)
    >>> a = np.zeros((10, 5))
    >>> a[3, :] = 1.23
    >>> print(a) # doctest: +NORMALIZE_WHITESPACE
    [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     ...
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]

    >>> with full_print():
    ...     print(a)
    [[0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [1.23 1.23 1.23 1.23 1.23]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]]

    >>> np.set_printoptions(**temp_options)

    """
    # get current options
    opt = np.get_printoptions()
    # update to print all digits
    np.set_printoptions(threshold=np.nan)
    # yield options for the context manager to do it's thing
    yield
    # reset the options back to what they were originally
    np.set_printoptions(**opt)

#%% pprint_dict
def pprint_dict(dct, *, name='', indent=1, align=True, disp=True):
    r"""
    Print all the fields and their values.

    Parameters
    ----------
    dct : dict
        Dictionary to print
    name : str, optional, default is empty string
        Name title to print first
    indent : int, optional, default is 1
        Number of characters to indent before all the fields
    align : bool, optional, default is True
        Whether to align all the equal signs

    Notes
    -----
    #.  Written by David C. Stauffer in February 2017.

    Examples
    --------
    >>> from dstauffman import pprint_dict
    >>> dct = {'a': 1, 'bb': 2, 'ccc': 3}
    >>> name = 'Example'
    >>> text = pprint_dict(dct, name=name)
    Example
     a   = 1
     bb  = 2
     ccc = 3

    """
    # print the name of the class/dictionary
    text = []
    if name:
        text.append(name)
    # build indentation padding
    this_indent = ' ' * indent
    # find the length of the longest field name
    pad_len = max(len(x) for x in dct)
    # loop through fields
    for (this_key, this_value) in dct.items():
        this_pad = ' ' * (pad_len - len(this_key)) if align else ''
        this_line = '{}{}{} = {}'.format(this_indent, this_key, this_pad, this_value)
        text.append(this_line)
    text = '\n'.join(text)
    if disp:
        print(text)
    return text

#%% line_wrap
def line_wrap(text, wrap=80, min_wrap=0, indent=4, line_cont='\\'):
    r"""
    Wrap lines of text to the specified length, breaking at any whitespace characters.

    Parameters
    ----------
    text : str or list of str
        Text to be wrapped
    wrap : int, optional
        Number of characters to wrap text at, default is 80
    min_wrap : int, optional
        Minimum number of characters to wrap at, default is 0
    indent : int, optional
        Number of characters to indent the next line with, default is 4

    Returns
    -------
    out : str or list of str
        wrapped form of text

    Examples
    --------
    >>> from dstauffman import line_wrap
    >>> text = ('lots of repeated words ' * 4).strip()
    >>> wrap = 40
    >>> out = line_wrap(text, wrap)
    >>> print(out)
    lots of repeated words lots of \
        repeated words lots of repeated \
        words lots of repeated words

    """
    # check if single str
    is_single = isinstance(text, str)
    if is_single:
        text = [text]
    # create the pad for any newline
    pad = ' ' * indent
    # initialize output
    out = []
    # loop through text lines
    for this_line in text:
        # determine if too long
        while len(this_line) > wrap:
            # find the last whitespace to break on, possibly with a minimum start
            space_break = this_line.rfind(' ', min_wrap, wrap-1)
            if space_break == -1 or space_break <= indent:
                raise ValueError('The specified min_wrap:wrap of "{}:{}" was too small.'.format(min_wrap, wrap))
            # add the shorter line
            out.append(this_line[:space_break] + ' ' + line_cont)
            # reduce and repeat
            this_line = pad + this_line[space_break+1:]
        # add the final shorter line
        out.append(this_line)
    if is_single:
        out = '\n'.join(out)
    return out

#%% combine_per_year
def combine_per_year(data, func=None):
    r"""
    Combine the time varying values over one year increments using a supplied function.

    Parameters
    ----------
    data : ndarray, 1D or 2D
        Data array

    Returns
    -------
    data2 : ndarray, 1D or 2D
        Data array combined as mean over 12 month periods

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.
    #.  Made more generic by David C. Stauffer in August 2017.
    #.  This function was designed with np.nanmean and np.nansum in mind.

    Examples
    --------
    >>> from dstauffman import combine_per_year
    >>> import numpy as np
    >>> time = np.arange(120)
    >>> data = np.sin(time)
    >>> data2 = combine_per_year(data, func=np.mean)

    """
    # check that a function was provided
    assert func is not None and callable(func), 'A callable function must be provided.'
    # check for null case and exit
    if data is None:
        return None
    # check dimensionality
    is_1d = True if data.ndim == 1 else False
    # get original sizes
    if is_1d:
        data = data[:, np.newaxis]
    (num_time, num_chan) = data.shape
    num_year = num_time // MONTHS_PER_YEAR
    # check for case with all NaNs
    if np.all(np.isnan(data)):
        data2 = np.full((num_year, num_chan), np.nan, dtype=float)
    else:
        # disables warnings for time points that are all NaNs for nansum or nanmean
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            warnings.filterwarnings('ignore', message='Sum of empty slice')
            # calculate sum or mean (or whatever)
            data2 = func(np.reshape(data[:num_year*MONTHS_PER_YEAR, :], (num_year, MONTHS_PER_YEAR, num_chan)), axis=1)
    # optionally squeeze the vector case back to 1D
    if is_1d:
        data2 = data2.squeeze(axis=1)
    return data2

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_utils', exit=False)
    doctest.testmod(verbose=False)
