# -*- coding: utf-8 -*-
r"""
Utils module file for the "dstauffman" library.  It contains generic utilities that can be
independently defined and used by other modules.

Notes
-----
#.  By design, this module does not reference any other piece of the dstauffman code base except
        constants or enums to avoid circular references.
#.  Written by David C. Stauffer in March 2015.
"""

# pylint: disable=E1101, C0301, C0103

#%% Imports
from __future__ import print_function
from __future__ import division
from contextlib import contextmanager
import doctest
import os
import numpy as np
import sys
import unittest
from datetime import datetime, timedelta
from dstauffman.constants import MONTHS_PER_YEAR
# compatibility issues
ver = sys.version_info
if ver[0] == 2:
    from io import BytesIO as StringIO
elif ver[0] == 3:
    from io import StringIO
else:
    raise('Unexpected Python version: "{}'.format(ver[0]))

#%% Functions - rms
def rms(data, axis=None, keepdims=False):
    r"""
    Calculates the root mean square of a number series

    Parameters
    ----------
    data : array_like
        input data
    axis : int, optional
        Axis along which the RMS is computed. The default is to compute the RMS of the flattened array.
    keepdims : bool, optional
        If true, the axes which are reduced are left in the result as dimensions ith size one.
        With this option, the result will broadcast correctly against the original `data`.

    Returns
    -------
    out : ndarray
        RMS results

    See Also
    --------
    numpy.mean, numpy.conj, numpy.sqrt

    Notes
    -----
    #. Written by David C. Stauffer in Mar 2015.

    Examples
    --------

    >>> from dstauffman import rms
    >>> rms([0, 1, 0., -1])
    0.70710678118654757

    """
    # do the root-mean-square, but use x * conj(x) instead of square(x) to handle complex numbers correctly
    out = np.sqrt(np.mean(data * np.conj(data), axis=axis, keepdims=keepdims))
    # return the result
    return out

#%% Functions - setup_dir
def setup_dir(folder, rec=False):
    r"""
    Clears the contents for existing folders or instantiates the directory if it doesn't exist.

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
    #. Written by David C. Stauffer in Feb 2015.

    Examples
    --------

    >>> from dstauffman import setup_dir
    >>> setup_dir(r'C:\Temp\test_folder') #doctest: +SKIP

    """
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
                    .format(this_full_elem)) #pragma: no cover
        print('Files/Sub-folders were removed from: "' + folder + '"')
    else:
        # create directory if it does not exist
        try:
            os.makedirs(folder)
            print('Created directory: "' + folder + '"')
        except: #pragma: no cover
            # re-raise last exception, could try to handle differently in the future
            raise #pragma: no cover

#%% Functions - compare_two_classes
def compare_two_classes(c1, c2, suppress_output=False, names=None):
    r"""
    Compares two classes by going through all their public attributes and showing that they are equal.

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
    b is different.
    c is only in c1.
    d is only in c2.
    "c1" and "c2" are not the same.

    """
    # preallocate answer to True until proven otherwise
    is_same = True
    # get names if specified
    if names is not None:
        name1 = names[0]
        name2 = names[1]
    else:
        # TODO: figure out Matlab inputname equivalent
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
            # if any differences, then this test fails
            if np.logical_not(_nan_equal(getattr(c1, this_attr), getattr(c2, this_attr))):
                is_same = False
                if not suppress_output:
                    print(this_attr + ' is different.')
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
    Compares two dictionaries for the same keys, and the same value of those keys.

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
        # TODO: figure out Matlab inputname equivalent
        name1 = 'd1'
        name2 = 'd2'
    # simple test
    if d1 is not d2:
        # compare the keys that are in both
        same = set(d1.keys()) & set(d2.keys())
        for key in sorted(same):
            # if any differences, then this test fails
            if np.any(d1[key] != d2[key]):
                is_same = False
                if not suppress_output:
                    print(key + ' is different.')
        # find keys in one but not the other, if any, then this test fails
        diff = set(d1.keys()) ^ set(d2.keys())
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
    ----------
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
    if dt == None:
        dt = datetime.now()
    # get the current elasped time in seconds
    seconds = (dt - dt.min).seconds
    # round to the nearest whole second
    rounding = (seconds+round_to_sec/2) // round_to_sec * round_to_sec
    # return the rounded result
    return dt + timedelta(0, rounding-seconds, -dt.microsecond)

#%% Functions - make_python_init
def make_python_init(folder):
    r"""
    Makes the Python __init__.py file based on the files/definitions found within the specified folder.
    """
    # exclusions
    exclusions = ['__init__.py']
    # initialize output
    results = {}
    # Loop through the contained files/folders
    for this_elem in os.listdir(folder):
        # alias the fullpath of this file element
        this_full_elem = os.path.join(folder, this_elem)
        # check if a folder or file
        if not os.path.isdir(this_full_elem):
            # get the file extension
            fileext = this_full_elem.split('.')
            # only process source *.py files
            if fileext[-1] == 'py':
                # exclude any existing '__init__.py' file
                if any([this_elem.startswith(exc) for exc in exclusions]):
                    continue
                # read the contents of the file
                text = read_text_file(this_full_elem)
                # get a list of definitions from the text file
                funcs = get_python_definitions(text)
                # append these results (if not empty)
                if len(funcs) > 0:
                    results[this_elem[:-3]] = funcs
    # check for duplicates
    all_funcs = [func for k in results for func in results[k]]
    if len(all_funcs) != len(set(all_funcs)):
        print('Uniqueness Problem: {funs} functions, but only {uni_funcs} unique functions'.format( \
            funs=len(all_funcs), uni_funcs=len(set(all_funcs))))
    dups = set([x for x in all_funcs if all_funcs.count(x) > 1])
    if dups:
        print('Duplicated functions:')
        print(dups)
    # start building text output
    text = []
    # loop through results and build text output
    for k in sorted(results):
        temp = ', '.join(results[k])
        text.append('from .' + k + ' import ' + temp)
    return '\n'.join(text)

#%% Functions - get_python_definitions
def get_python_definitions(text):
    r"""
    Gets all public class and def names from the text of the file.
    """
    funcs = []
    for line in text.split('\n'):
        if line.startswith('class ') and not line.startswith('class _'):
            temp = line[len('class '):].split('(')
            funcs.append(temp[0])
        if line.startswith('def ') and not line.startswith('def _'):
            temp = line[len('def '):].split('(')
            funcs.append(temp[0])
    return funcs

#%% Functions - read_text_file
def read_text_file(filename):
    r"""
    Opens and reads a complete text file.

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
    >>> write_text_file(os.path.join(get_tests_dir(), 'temp_file.txt'), text)
    >>> text2 = read_text_file(os.path.join(get_tests_dir(), 'temp_file.txt'))
    >>> print(text2)
    Hello, World
    <BLANKLINE>

    """
    try:
        with open(filename, 'rt') as file: # pylint: disable=W1501
            text = file.read()
        return text
    except:
        print('Unable to open file "{}" for reading.'.format(filename))
        raise

#%% Functions - write_text_file
def write_text_file(filename, text):
    r"""
    Opens and writes the specified text to a file.

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
    >>> write_text_file(os.path.join(get_tests_dir(), 'temp_file.txt'), text)

    """
    try:
        with open(filename, 'wt') as file: # pylint: disable=W1501
            file.write(text)
    except:
        print('Unable to open file "{}" for writing.'.format(filename))
        raise

#%% Functions - disp
def disp(struct, level=0, padding=None, suppress_output=False):
    r"""
    Matlab like 'disp' or display function.

    Parameters
    ----------
    struct : class
        Structure to display
    level : int, optional
        Level to indent, used for substructures within structures.
    padding : int, optional
        Minimum number of spaces to pad the results to
    suppress_output : bool, optional
        Choose whether to display results to the command window, default False, so show output

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman import disp
    >>> a = type('a', (object, ), {'b': 0, 'c' : '[1, 2, 3]', 'd': 'text'})
    >>> txt = disp(a)
    b : 0
    c : [1, 2, 3]
    d : text

    """
    # padding per additional level
    pad_per_level = 4
    # get the variables within the structure
    d = vars(struct)
    # determine padding level
    if padding is None:
        padding = max([len(name) for name in d if not name.startswith('_')]) + 1
    # initialize output
    x = ''
    # loop through dict of vars
    for name in sorted(d):
        if name.startswith('_'):
            continue
        # find out how many characters to pad on the front
        pad_len = padding - len(name) - 2
        # find out if an extra pad around the dots (" ... ") has room
        extra_pad = ' ' if pad_len > -1 else ''
        # append this variable
        x = x + pad_per_level*level * ' ' + (name + ' ' + (pad_len * '.') + extra_pad + ': ' + str(d[name]) + '\n')
    # print the results to the screen
    if not suppress_output:
        print(x[:-1])
    # return the final results, minus the last newline character
    return x[:-1]

#%% Functions - convert_annual_to_monthly_probability
def convert_annual_to_monthly_probability(annual):
    r"""
    Converts a given annual probabily into the equivalent monthly one.

    Parameters
    ----------
    annual : numpy.nd_array
        annual probabilities, 0 <= annual <= 1

    Returns
    -------
    monthly : numpy.nd_array
        equivalent monthly probabilities, 0 <= monthly <= 1

    Raises
    ------
    ValueError
        Any probabilities outside of the [0, 1] range

    Notes
    -----
    #.  Checks for boundary cases to avoid a divide by zero warning

    Examples
    --------

    >>> from dstauffman import convert_annual_to_monthly_probability
    >>> import numpy as np
    >>> annual  = np.array([0, 0.1, 1])
    >>> monthly = convert_annual_to_monthly_probability(annual)
    >>> print(monthly) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0.00874161  1. ]

    """
    # check ranges
    if np.any(annual < 0):
        raise ValueError('annual must be >= 0')
    if np.any(annual > 1):
        raise ValueError('annual must be <= 1')
    # ignore divide by zero errors when annual == 1
    with np.errstate(divide='ignore'):
        # convert to equivalent probability and return result
        monthly = 1-np.exp(np.log(1-annual)/MONTHS_PER_YEAR)
    return monthly

#%% Functions - get_root_dir
def get_root_dir():
    r"""
    Returns the folder that contains this source file and thus the root folder for the whole code.

    Returns
    -------
    folder : str
        Location of the folder that contains all the source files for the code.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman import get_root_dir
    >>> folder = get_root_dir()

    """
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return folder

#%% Functions - get_tests_dir
def get_tests_dir():
    r"""
    Returns the default test folder location.

    Returns
    -------
    folder : str
        Location of the folder that contains all the test files for the code.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman import get_tests_dir
    >>> folder = get_tests_dir()

    """
    folder = os.path.join(get_root_dir(), 'tests')
    return folder

#%% Functions - get_data_dir
def get_data_dir():
    r"""
    Returns the default data folder location.

    Returns
    -------
    folder : str
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.

    Examples
    --------

    >>> from dstauffman import get_data_dir
    >>> folder = get_data_dir()

    """
    folder = os.path.join(get_root_dir(), 'data')
    return folder

#%% Functions - capture_output
@contextmanager
def capture_output():
    r"""
    Captures the stdout and stderr streams instead of displaying to the screen.

    Returns
    -------
    out : class StringIO
        stdout stream output
    err : class StringIO
        stderr stream output

    Examples
    --------

    >>> from dstauffman import capture_output
    >>> with capture_output() as (out, _):
    ...     print('Hello, World!')
    >>> output = out.getvalue().strip()
    >>> out.close()
    >>> print(output)
    Hello, World!

    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

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
        np.testing.assert_equal(a, b)
    except AssertionError:
        # if assertion fails, then they are not equal
        is_same = False
    return is_same

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_utils', exit=False)
    doctest.testmod(verbose=False)
