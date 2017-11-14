# -*- coding: utf-8 -*-
r"""
Generic utilities that can be independently defined and used by other modules.

Notes
-----
#.  By design, this module does not reference any other piece of the dstauffman code base except
        constants or enums to avoid circular references.
#.  Written by David C. Stauffer in March 2015.

"""

#%% Imports
import doctest
import inspect
import logging
import os
import shutil
import sys
import types
import unittest
import warnings
from collections import Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import StringIO

import numpy as np

from dstauffman.constants import MONTHS_PER_YEAR

#%% Globals
root_logger = logging.getLogger('')
logger      = logging.getLogger(__name__)

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
    0.70710678118654757

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

#%% Functions - make_python_init
def make_python_init(folder, lineup=True, wrap=100, filename=''):
    r"""
    Make the Python __init__.py file based on the files/definitions found within the specified folder.

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
    >>> print(text[0:21])
    from .analysis import

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
    # get information about padding
    max_len   = max(len(x) for x in results)
    indent = len('from . import ') + max_len + 4
    # start building text output
    text = []
    # loop through results and build text output
    for key in sorted(results):
        pad = ' ' * (max_len - len(key)) if lineup else ''
        temp = ', '.join(results[key])
        header = 'from .' + key + pad + ' import '
        min_wrap = len(header)
        this_line = [header + temp]
        wrapped_lines = line_wrap(this_line, wrap=wrap, min_wrap=min_wrap, indent=indent)
        text += wrapped_lines
    # combined the text into a single string with newline characters
    output = '\n'.join(text)
    # optionally write the results to a file
    if filename:
        write_text_file(filename, output)
    return output

#%% Functions - get_python_definitions
def get_python_definitions(text):
    r"""
    Get all public class and def names from the text of the file.

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

#%% Functions - get_root_dir
def get_root_dir():
    r"""
    Return the folder that contains this source file and thus the root folder for the whole code.

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
    Return the default test folder location.

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
    Return the default data folder location.

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
    Return the default data folder location.

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

#%% Functions - get_output_dir
def get_output_dir():
    r"""
    Return the default output folder location.

    Returns
    -------
    folder : str
        Location of the default folder for storing the code data.

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------
    >>> from dstauffman import get_output_dir
    >>> folder = get_output_dir()

    """
    # this folder is the 'images' subfolder
    folder = os.path.join(get_root_dir(), 'results')
    return folder

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
def unit(data, axis=1):
    r"""
    Normalize a matrix into unit vectors along a specified dimension.

    Default to column normalization.

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
    >>> data = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 1]])
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

#%% Functions - reload_package
def reload_package(root_module, disp_reloads=True): # pragma: no cover
    r"""
    Force Python to reload all the items within a module.  Useful for interactive debugging in IPython.

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

    # load each of the modules again
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
    Delete all the *.pyc files (Python Byte Code) in the specified directory.

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
    >>> delete_pyc(folder, print_progress=False) # doctest: +SKIP

    """
    def _remove_pyc(root, name):
        r"""Do the actual file removal."""
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

#%% rename_module
def rename_module(folder, old_name, new_name, print_status=True):
    r"""
    Rename the given module from the old to new name.

    Parameters
    ----------
    folder : str
        Name of folder to operate within
    old_name : str
        The original name of the module
    new_name : str
        The new name of the module

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015 mostly for Matt Beck, because he doesn't
        like the name of my library.

    Examples
    --------
    >>> from dstauffman import rename_module, get_root_dir
    >>> import os
    >>> folder = os.path.split(get_root_dir())[0]
    >>> old_name = 'dstauffman'
    >>> new_name = 'dcs_tools'
    >>> rename_module(folder, old_name, new_name) # doctest: +SKIP

    """
    # hard-coded values
    folder_exclusions = {'.git'}
    file_exclusions = {'.pyc'}
    files_to_edit = {'.py', '.rst', '.bat', '.txt'}
    root_ix = len(folder)
    for (root, _, files) in os.walk(os.path.join(folder, old_name)):
        for skip in folder_exclusions:
            if root.endswith(skip) or root.find(skip + os.path.sep) >= 0:
                if print_status:
                    print('Skipping: {}'.format(root))
                break
        else:
            for name in files:
                (_, file_ext) = os.path.splitext(name)
                this_old_file = os.path.join(root, name)
                this_new_file = os.path.join(folder + root[root_ix:].replace(old_name, new_name), \
                    name.replace(old_name, new_name))
                if file_ext in file_exclusions:
                    if print_status:
                        print('Skipping: {}'.format(this_old_file))
                    continue # pragma: no cover (actually covered, optimization issue)
                # only create the new folder if not all files skipped
                new_folder = os.path.split(this_new_file)[0]
                if not os.path.isdir(new_folder):
                    setup_dir(new_folder)
                if file_ext in files_to_edit:
                    # edit files
                    if print_status:
                        print('Editing : {}'.format(this_old_file))
                        print('     To : {}'.format(this_new_file))
                    text = read_text_file(this_old_file)
                    text = text.replace(old_name, new_name)
                    write_text_file(this_new_file, text)
                else:
                    # copy file as-is
                    if print_status:
                        print('Copying : {}'.format(this_old_file))
                        print('     To : {}'.format(this_new_file))
                    shutil.copyfile(this_old_file, this_new_file)

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

#%% find_tabs
def find_tabs(folder, extensions=None, list_all=False, trailing=False):
    r"""
    Find all the tabs in source code that should be spaces instead.

    Parameters
    ----------
    folder : str
        Folder path to search
    list_all : bool, optional, default is False
        Whether to list all the files, or only those with tabs in them
    trailing : bool, optional, default is False
        Whether to consider trailing whitespace a problem, too

    Examples
    --------
    >>> from dstauffman import find_tabs, get_root_dir
    >>> folder = get_root_dir()
    >>> find_tabs(folder)

    """
    if extensions is None:
        extensions = frozenset(('m', 'py'))
    for (root, dirs, files) in os.walk(folder, topdown=True):
        dirs.sort()
        for name in sorted(files):
            fileparts = name.split('.')
            if fileparts[-1] in extensions:
                this_file = os.path.join(root, name)
                already_listed = list_all
                if already_listed:
                    print('Evaluating: "{}"'.format(this_file))
                with open(this_file, encoding='utf8') as file:
                    c = 0
                    try:
                        for line in file:
                            c += 1
                            if line.count('\t') > 0:
                                if not already_listed:
                                    print('Evaluating: "{}"'.format(this_file))
                                    already_listed = True
                                print('    Line {:03}: '.format(c) + repr(line))
                            elif trailing and len(line) >= 2 and line[-2] == ' ' and sum(1 for x in line if x not in ' \n') > 0:
                                if not already_listed:
                                    print('Evaluating: "{}"'.format(this_file))
                                    already_listed = True
                                print('    Line {:03}: '.format(c) + repr(line))
                    except UnicodeDecodeError: # pragma: no cover
                        print('File: "{}" was not a valid utf-8 file.'.format(this_file))

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
    precision = 1e-13

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
    [[ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     ...,
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]

    >>> with full_print():
    ...     print(a)
    [[ 0.    0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.  ]
     [ 1.23  1.23  1.23  1.23  1.23]
     [ 0.    0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.  ]]

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
def line_wrap(text, wrap=80, min_wrap=0, indent=4):
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
            out.append(this_line[:space_break] + ' \\')
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

#%% Functions - activate_logging
def activate_logging(log_level=logging.INFO, filename=''):
    r"""
    Set up logging based on a user specified settings file.

    Parameters
    ----------
    log_level : int
        Level of logging
    filename : str
        File to log to, if empty, use default output folder with today's date

    Notes
    -----
    #.  Written by David C. Stauffer in August 2017.

    Examples
    --------
    >>> from dstauffman import activate_logging, deactivate_logging, get_tests_dir
    >>> import logging
    >>> import os
    >>> filename = os.path.join(get_tests_dir(), 'testlog.txt')
    >>> activate_logging(log_level=logging.DEBUG, filename=filename)
    >>> logging.debug('Test message') # doctest: +SKIP
    >>> deactivate_logging()

    Remove the log file
    >>> os.remove(filename)

    """
    # update the log level
    root_logger.setLevel(log_level)

    # optionally get the default filename
    if not filename:
        filename = os.path.join(get_output_dir(), 'log_file_' + datetime.now().strftime('%Y-%m-%d') + '.txt')

    # create the log file handler
    fh = logging.FileHandler(filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(fh)

    # create the log stream handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter('Log: %(message)s'))
    root_logger.addHandler(ch)

#%% Functions - deactivate_logging
def deactivate_logging():
    r"""
    Tear down logging.

    Notes
    -----
    #.  Written by David C. Stauffer in August 2017.

    Examples
    --------
    >>> from dstauffman import deactivate_logging
    >>> deactivate_logging()

    """
    # hard-coded values
    max_handlers = 50
    # initialize a counter to avoid infinite while loop
    i = 0
    # loop through and remove all the handlers
    while root_logger.handlers and i < max_handlers:
        handler = root_logger.handlers.pop()
        handler.flush()
        handler.close()
        root_logger.removeHandler(handler)
        # increment the counter
        i += 1
    # check for bad situations
    if i == max_handlers or bool(root_logger.handlers):
        raise ValueError('Something bad happended when trying to close the logger.') # pragma: no cover

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_utils', exit=False)
    doctest.testmod(verbose=False)
