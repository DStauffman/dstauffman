# -*- coding: utf-8 -*-
r"""
Utils module file for the "dstauffman" library.  It contains generic utilities that can be
independently defined and used by other modules.

Notes
-----
#.  By design, this model does not reference any other piece of the dstauffman code base exceptx
        constans or enums to avoid circular references.
#.  Written by David C. Stauffer in March 2015.
"""

# pylint: disable=E1101, C0301, C0103

#%% Imports
from __future__ import print_function
from __future__ import division
import os
import numpy as np
import unittest
from datetime import datetime, timedelta
from dstauffman.constants import MONTHS_PER_YEAR

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
    os.mkdir, os.rmdir, os.remove

    Raises
    ------
    RuntimeError
        Problems creating or deleting a file or folder, likely due to permission issues.

    Notes
    -----
    #. Written by David C. Stauffer in Feb 2015.

    Examples
    --------

    >>> setup_dir(r'C:\Temp\test_folder')

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
                raise RuntimeError('Unexpected file type, neither file nor folder: "{}".'.format(this_full_elem))
        print('Files/Sub-folders were removed from: "' + folder + '"')
    else:
        # create directory if it does not exist
        try:
            os.mkdir(folder)
            print('Created directory: "' + folder + '"')
        except:
            # re-raise last exception, could try to handle differently in the future
            raise

#%% Functions - compare_two_structures
def compare_two_structures(c1, c2):
    r"""
    Compares to classes by going through all their public attributes and showing that they are equal.
    """
    # TODO: write this function
    return True

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

    >>> text = read_text_file(r'temp_file.txt')

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

    >>> text = 'Hello, World\n'
    >>> write_text_file(r'temp_file.txt', text)

    """
    try:
        with open(filename, 'wt') as file: # pylint: disable=W1501
            file.write(text)
    except:
        print('Unable to open file "{}" for writing.'.format(filename))
        raise

#%% Functions - disp
def disp(struct, level=0, padding=12):
    r"""
    Matlab like 'disp' or display function.
    """
    # padding per additional level
    pad_per_level = 4
    # get the variables within the structure
    d = vars(struct)
    # initialize output
    x = '\n'
    # loop through dict of vars
    for name in sorted(d):
        # find out how many characters to pad on the front
        pad_len = padding - len(name) - 2
        # find out if an extra pad around the dots (" ... ") has room
        extra_pad = ' ' if pad_len > -1 else ''
        # append this variable
        x = x + pad_per_level*level * ' ' + (name + ' ' + (pad_len * '.') + extra_pad + ': ' + str(d[name]) + '\n')
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
        Any probabilities outside of the [0,1] range

    Examples
    --------

    >>> import numpy as np
    >>> monthly = convert_annual_to_monthly_probability(np.array([0, 0.1, 1]))
    >>> print(monthly)
    [ 0.          0.00874161  1.        ]

    """
    # check ranges
    if np.any(annual < 0):
        raise ValueError('annual must be >= 0')
    if np.any(annual > 1):
        raise ValueError('annual must be <= 1')
    # convert to equivalent probability and return result
    monthly = 1-np.exp(np.log(1-annual)/MONTHS_PER_YEAR)
    return monthly

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_utils', exit=False)
