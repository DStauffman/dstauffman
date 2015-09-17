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
import inspect
import os
import numpy as np
import sys
import types
import unittest
from datetime import datetime, timedelta
from dstauffman.constants import MONTHS_PER_YEAR
# compatibility issues
ver = sys.version_info
if ver[0] == 2:
    from io import BytesIO as StringIO # pragma: no cover
elif ver[0] == 3:
    from io import StringIO # pragma: no cover
else:
    raise('Unexpected Python version: "{}'.format(ver[0])) # pragma: no cover

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

#%% Functions - rms
def rms(data, axis=None, keepdims=False):
    r"""
    Calculates the root mean square of a number series

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
    numpy.mean, numpy.conj, numpy.sqrt

    Notes
    -----
    #.  Written by David C. Stauffer in Mar 2015.

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
    #.  Written by David C. Stauffer in Feb 2015.

    Examples
    --------

    >>> from dstauffman import setup_dir
    >>> setup_dir(r'C:\Temp\test_folder') # doctest: +SKIP

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
                    .format(this_full_elem)) # pragma: no cover
        print('Files/Sub-folders were removed from: "' + folder + '"')
    else:
        # create directory if it does not exist
        try:
            os.makedirs(folder)
            print('Created directory: "' + folder + '"')
        except: # pragma: no cover
            # re-raise last exception, could try to handle differently in the future
            raise # pragma: no cover

#%% Functions - compare_two_classes
def compare_two_classes(c1, c2, suppress_output=False, names=None, ignore_callables=True, compare_recursively=True):
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
        r"""Sets is_same to False and optionally prints information to the screen."""
        is_same = False
        if not suppress_output:
            print('{} is different from {} to {}.'.format(this_attr, name1, name2))
        return is_same
    def _is_function(obj):
        r"""Determines whether the object is a function or not."""
        # need second part for Python compatibility for v2.7, which distinguishes unbound methods from functions.
        return inspect.isfunction(obj) or inspect.ismethod(obj)
    def _is_class_instance(obj):
        r"""Determines whether the object is an instance of a class or not."""
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
                        # Note: don't want the 'and' to short-circuit, so do the 'and is_same' last
                        is_same = compare_two_classes(attr1, attr2, suppress_output=suppress_output, \
                            names= [name1 + '.' + this_attr, name2 + '.' + this_attr], \
                            ignore_callables=ignore_callables, compare_recursively=compare_recursively) and is_same
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
            if np.logical_not(_nan_equal(getattr(c1, this_attr), getattr(c2, this_attr))):
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
        name1 = 'd1'
        name2 = 'd2'
    # simple test
    if d1 is not d2:
        # compare the keys that are in both
        same = set(d1.keys()) & set(d2.keys())
        for key in sorted(same):
            # if any differences, then this test fails
            if np.any((d1[key] != d2[key]) ^ (np.isnan(d1[key]) & np.isnan(d2[key]))):
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

    Parameters
    ----------
    folder : str
        Name of folder to process

    Returns
    -------
    output : str
        Resulting text for __init__.py file

    Examples
    --------

    >>> from dstauffman import make_python_init, get_root_dir
    >>> folder = get_root_dir()
    >>> text = make_python_init(folder)
    >>> print(text[0:20])
    from .classes import

    """
    # exclusions
    exclusions = ['__init__.py']
    # initialize intermediate results
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
    # pass back the combined output text
    output = '\n'.join(text)
    return output

#%% Functions - get_python_definitions
def get_python_definitions(text):
    r"""
    Gets all public class and def names from the text of the file.

    Parameters
    ----------
    text : str
        The text of the python file

    Returns
    -------
    funcs : array_like, str
        List of functions within the text of the python file

    Examples
    --------

    >>> from dstauffman import get_python_definitions
    >>> text = 'def a():\n    pass\n'
    >>> funcs = get_python_definitions(text)
    >>> print(funcs)
    ['a']

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
        # open file for reading
        with open(filename, 'rt') as file: # pylint: disable=W1501
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
        # open file for writing
        with open(filename, 'wt') as file: # pylint: disable=W1501
            # write file
            file.write(text) # pragma: no branch
    except:
        # on any exceptions, print a message and re-raise the error
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

#%% Functions - convert_monthly_to_annual_probability
def convert_monthly_to_annual_probability(monthly):
    r"""
    Converts a given monthly probability into the equivalent annual one.

    Parameters
    ----------
    monthly : numpy.nd_array
        equivalent monthly probabilities, 0 <= monthly <= 1

    Returns
    -------
    annual : numpy.nd_array
        annual probabilities, 0 <= annual <= 1

    Examples
    --------

    >>> from dstauffman import convert_monthly_to_annual_probability
    >>> import numpy as np
    >>> monthly = np.array([0, 0.1, 1])
    >>> annual = convert_monthly_to_annual_probability(monthly)
    >>> print(annual) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0.71757046 1. ]

    """
    # check ranges
    if np.any(monthly < 0):
        raise ValueError('monthly must be >= 0')
    if np.any(monthly > 1):
        raise ValueError('annual must be <= 1')
    # convert to equivalent probability and return result
    annual = 1 - (1 - monthly)**MONTHS_PER_YEAR
    return annual

#%% Functions - ca2mp & cm2ap aliases
ca2mp = convert_annual_to_monthly_probability
cm2ap = convert_monthly_to_annual_probability

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
    # this folder is the root directory based on the location of this file (utils.py)
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
    # this  folder is the 'tests' subfolder
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
    # this folder is the 'data' subfolder
    folder = os.path.join(get_root_dir(), 'data')
    return folder

#%% Functions - get_images_dir
def get_images_dir():
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

    >>> from dstauffman import get_images_dir
    >>> folder = get_images_dir()

    """
    # this folder is the 'images' subfolder
    folder = os.path.join(get_root_dir(), 'images')
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
    # create new string buffers
    new_out, new_err = StringIO(), StringIO()
    # alias the old string buffers for restoration afterwards
    old_out, old_err = sys.stdout, sys.stderr
    try:
        # override the system buffers with the new ones
        sys.stdout, sys.stderr = new_out, new_err
        # yield results as desired
        yield sys.stdout, sys.stderr
    finally:
        # restore the original buffers once all results are read
        sys.stdout, sys.stderr = old_out, old_err

#%% Functions - unit
def unit(data, axis=1):
    r"""
    Normalizes a matrix into unit vectors along a specified dimension, default to column
    normalization.

    Parameters
    ----------
    data : ndarray
        Data
    axis : int, optional
        Axis upon which to normalize

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
    >>> data = np.array([[1, 0, -1],[0, 0, 0], [0, 0, 1]])
    >>> norm_data = unit(data, axis=0)
    >>> print(norm_data) # doctest: +NORMALIZE_WHITESPACE
    [[ 1. 0. -0.70710678]
     [ 0. 0.  0.        ]
     [ 0. 0.  0.70710678]]

    """
    # calculate the magnitude of each vector
    mag = np.sqrt(np.sum(data * np.conj(data), axis=axis))
    # check for zero vectors, and replace magnitude with 1 to make them unchanged
    mag[mag == 0] = 1
    # calculate the new normalized data
    norm_data = data / mag
    return norm_data

#%% Functions - nonzero_indices
def nonzero_indices(data):
    r"""
    Returns the indices for a boolean array, similar to the Matlab "find" command.

    Parameters
    ----------
    data : array_like
        Input data

    Returns
    -------
    ix : ndarray
        Indices to elements that evaluate as true

    Notes
    -----
    #.  Written by David C. Stauffer in May 2015.

    Examples
    --------

    >>> from dstauffman import nonzero_indices
    >>> data = [True, True, False, False, True]
    >>> ix   = nonzero_indices(data)
    >>> print(ix) # doctest: +NORMALIZE_WHITESPACE
    [0 1 4]

    """
    # build and return the indices
    ix = np.array([ix for (ix, val) in enumerate(data) if val], dtype=int)
    return ix

#%% Functions - combine_sets
def combine_sets(n1, u1, s1, n2, u2, s2):
    r"""
    Combines the mean and standard deviations for two non-overlapping sets of data.

    This function combines two non-overlapping data sets, given a number of samples, mean
    and standard deviation for the two data sets.  It first calculates the total number of samples
    then calculates the total mean using a weighted average, and then calculates the combined
    standard deviation using an equation found on wikipedia.  It also checks for special cases
    where either data set is empty or if only one total point is in the combined set.

    Parameters
    ----------
    n1 : float
        number of points in data set 1
    u1 : float
        mean of data set 1
    s1 : float
        standard deviation of data set 1
    n2 : float
        number of points in data set 2
    u2 : float
        mean of data set 2
    s2 : float
        standard deviation of data set 2

    Returns
    -------
    n  : float,
        number of points in the combined data set
    u  : float,
        mean of the combined data set
    s  : float,
        standard deviation of the combined data set

    See Also
    --------
    np.mean, np.std

    References
    ----------
    #.  http://en.wikipedia.org/wiki/Standard_deviation#Sample-based_statistics, on 8/7/12

    Notes
    -----
    #.  Written in Matlab by David C. Stauffer in Jul 2012.
    #.  Ported to Python by David C. Stauffer in May 2015.
    #.  Could be expanded to broadcast and handle array inputs.

    Examples
    --------

    >>> from dstauffman import combine_sets
    >>> n1 = 5
    >>> u1 = 1
    >>> s1 = 0.5
    >>> n2 = 10
    >>> u2 = 2
    >>> s2 = 0.25
    >>> (n, u, s) = combine_sets(n1, u1, s1, n2, u2, s2)
    >>> print(n)
    15
    >>> print(u) # doctest: +ELLIPSIS
    1.666666...67
    >>> print(s)
    0.59135639081

    """
    # assertions
    assert n1 >= 0
    assert n2 >= 0
    assert s1 >= 0
    assert s2 >= 0
    # combine total number of samples
    n = n1 + n2
    # check for zero case
    if n == 0:
        u = 0
        s = 0
        return (n, u, s)
    # calculate the combined mean
    u = 1/n * (n1*u1 + n2*u2)
    # calculate the combined standard deviation
    if n != 1:
        s = np.sqrt(1/(n-1) * ( (n1-1)*s1**2 + n1*u1**2 + (n2-1)*s2**2 + n2*u2**2 - n*u**2))
    else:
        # special case where one of the data sets is empty
        if n1 == 1:
            s = s1
        elif n2 == 1:
            s = s2
        else:
            # shouldn't be able to ever reach this line with assertions on
            raise ValueError('Total samples are 1, but neither data set has only one item.') # pragma: no cover
    return (n, u, s)

#%% Functions - reload_package
def reload_package(root_module, disp_reloads=True): # pragma: no cover
    r"""
    Forces Python to reload all the items within a module.  Useful for interactive debugging in IPython.

    Parameters
    ----------
    root_module : module
        The module to force python to reload
    disp_reloads : bool
        Whether to display the modules that are reloaded

    Notes
    -----
    #.  Added to the library in June 2015 from:
        http://stackoverflow.com/questions/2918898/prevent-python-from-caching-the-imported-modules/2918951#2918951
    #.  Restarting the IPython kernel is by far a safer option, but for some debugging cases this is useful,
        however unit tests on this function will fail, because it reloads too many dependencies.

    Examples
    --------
    >>> from dstauffman import reload_package
    >>> import dstauffman as dcs
    >>> reload_package(dcs) # doctest: +ELLIPSIS
    loading dstauffman...

    """
    package_name = root_module.__name__

    # get a reference to each loaded module
    loaded_package_modules = dict([
        (key, value) for key, value in sys.modules.items()
        if key.startswith(package_name) and isinstance(value, types.ModuleType)])

    # delete references to these loaded modules from sys.modules
    for key in loaded_package_modules:
        del sys.modules[key]

    # load each of the modules again;
    # make old modules share state with new modules
    for key in loaded_package_modules:
        if disp_reloads:
            print('loading {}'.format(key))
        newmodule = __import__(key)
        oldmodule = loaded_package_modules[key]
        oldmodule.__dict__.clear()
        oldmodule.__dict__.update(newmodule.__dict__)

#%% Functions - delete_pyc
def delete_pyc(folder, recursive=True, print_progress=True):
    r"""
    Deletes all the *.pyc files (Python Byte Code) in the specified directory.

    Parameters
    ----------
    folder : str
        Name of folder to delete the files from
    recursive : bool, optional
        Whether to delete files recursively

    Examples
    --------
    >>> from dstauffman import get_root_dir, delete_pyc
    >>> folder = get_root_dir()
    >>> delete_pyc(folder)

    """
    def _remove_pyc(root, name):
        r"""Does the actual file removal"""
        # check for allowable extensions
        (_, file_ext) = os.path.splitext(name)
        if file_ext == '.pyc':
            # remove this file
            if print_progress:
                print('Removing "{}"'.format(os.path.join(root, name)))
            os.remove(os.path.join(root, name))

    if recursive:
        # walk through folder
        for (root, _, files) in os.walk(folder):
            # go through files
            for name in files:
                # remove relevant files
                _remove_pyc(root, name)
    else:
        # list files in folder
        for name in os.listdir(folder):
            # check if it's a file
            if os.path.isfile(os.path.join(folder, name)):
                # remove relevant files
                _remove_pyc(folder, name)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_utils', exit=False)
    doctest.testmod(verbose=False)
