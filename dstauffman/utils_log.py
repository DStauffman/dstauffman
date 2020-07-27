r"""
Generic utilities that print or log information.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Split by logging options by David C. Stauffer in June 2020.
"""

#%% Imports
import doctest
import logging
import os
import unittest

import numpy as np

from dstauffman.enums import LogLevel

#%% Globals
logger = logging.getLogger(__name__)

#%% Functions - setup_dir
def setup_dir(folder, recursive=False):
    r"""
    Clear the contents for existing folders or instantiates the directory if it doesn't exist.

    Parameters
    ----------
    folder : str
        Location of the folder to empty or instantiate.
    recursive : bool, optional
        Whether to recursively delete contents.

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
                # if a folder, then delete recursively if recursive is True
                if recursive:
                    setup_dir(this_full_elem, recursive=recursive)
                    os.rmdir(this_full_elem)
            elif os.path.isfile(this_full_elem):
                # if a file, then remove it
                os.remove(this_full_elem)
            else:
                raise RuntimeError('Unexpected file type, neither file nor folder: "{}".'\
                    .format(this_full_elem)) # pragma: no cover
        logger.log(LogLevel.L1, 'Files/Sub-folders were removed from: "' + folder + '"')
    else:
        # create directory if it does not exist
        try:
            os.makedirs(folder)
            logger.log(LogLevel.L1, 'Created directory: "' + folder + '"')
        except: # pragma: no cover
            # re-raise last exception, could try to handle differently in the future
            raise # pragma: no cover

#%% Functions - fix_rollover
def fix_rollover(data, roll, axis=None):
    r"""
    Unrolls data that has finite ranges and rollovers.

    Parameters
    ----------
    data : (N,) or (N, M) array_like
        Raw input data with rollovers
    roll : int or float
        Range over which the data rolls over
    axis : int
        Axes to unroll over

    Returns
    -------
    out : ndarray
        Data with the rollovers removed

    Notes
    -----
    #.  Finds the points at which rollovers occur, where a rollover is considered a
        difference of more than half the rollover value, and then classifies the rollover as a top
        to bottom or bottom to top rollover.  It then rolls the data as a step function in partitions
        formed by the rollover points, so there will always be one more partition than rollover points.
    #.  Function can call itself recursively.
    #.  Adapted by David C. Stauffer from Matlab version in May 2020.

    Examples
    --------
    >>> from dstauffman import fix_rollover
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 0, 1,  3,  6,  0,  6,  5, 2])
    >>> roll = 7
    >>> out  = fix_rollover(data, roll)
    >>> print(out) # doctest: +NORMALIZE_WHITESPACE
    [ 1 2 3 4 5 6 7 8 10 13 14 13 12 9]

    """
    # call recursively when specifying an axis on 2D data
    if axis is not None:
        assert data.ndim == 2, 'Must be a 2D array when axis is specified'
        out = np.zeros_like(data)
        if axis == 0:
            for i in range(data.shape[1]):
                out[:, i] = fix_rollover(data[:, i], roll)
        elif axis == 1:
            for i in range(data.shape[0]):
                out[i, :] = fix_rollover(data[i, :], roll)
        else:
            raise ValueError(f'Unexpected axis: "{axis}".')
        return out

    # check that input is a vector and initialize compensation variables with the same dimensions
    if data.ndim == 1:
        if data.size == 0:
            return np.array([])
        # t2b means top to bottom rollovers, while b2t means bottom to top rollovers
        num_el           = data.size
        compensation_t2b = np.zeros_like(data)
        compensation_b2t = np.zeros_like(data)
    else:
        raise ValueError('Input argument "data" must be a vector.')

    # find indices for top to bottom rollovers, these indices act as partition boundaries
    roll_ix = np.flatnonzero(np.diff(data) > (roll/2))
    if roll_ix.size > 0:
        # add final field to roll_ix so that final partition can be addressed
        roll_ix = np.hstack((roll_ix, num_el-1))
        # loop only on original length of roll_ix, which is now length - 1
        for i in range(roll_ix.size - 1):
            # creates a step function to be added to the input array where each
            # step down occurs after a top to bottom roll over.
            compensation_t2b[roll_ix[i] + 1:roll_ix[i+1] + 1] = -roll * (i+1)
        # display a warning based on the log level
        logger.log(LogLevel.L6, 'corrected {} bottom to top rollover(s)'.format(roll_ix.size-1))

    # find indices for top to bottom rollover, these indices act as partition boundaries
    roll_ix = np.flatnonzero(np.diff(data) < -(roll/2))
    if roll_ix.size > 0:
        # add final field to roll_ix so that final partition can be addressed
        roll_ix = np.hstack((roll_ix, num_el-1))
        # loop only on original length of roll_ix, which is now length - 1
        for i in range(roll_ix.size-1):
            # creates a step function to be added to the input array where each
            # step up occurs after a bottom to top roll over.
            compensation_b2t[roll_ix[i] + 1:roll_ix[i+1] + 1] = roll * (i+1)
        # display a warning based on the log level
        logger.log(LogLevel.L6, 'corrected {} top to bottom rollover(s)'.format(roll_ix.size-1))

    # create output
    out = data + compensation_b2t + compensation_t2b

    # double check for remaining rollovers
    if np.any(np.diff(out) > (roll/2)) | np.any(np.diff(out) < -(roll/2)):
        logger.log(LogLevel.L6, 'A rollover was fixed recursively')
        out = fix_rollover(out, roll)
    return out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_utils_log', exit=False)
    doctest.testmod(verbose=False)
