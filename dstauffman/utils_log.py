r"""
Generic utilities that print or log information.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Split by logging options by David C. Stauffer in June 2020.
"""

#%% Imports
import contextlib
import doctest
import logging
from pathlib import Path
from typing import Union
import unittest

from slog import LogLevel

from dstauffman.constants import HAVE_NUMPY
from dstauffman.utils import find_in_range, rms

if HAVE_NUMPY:
    import numpy as np

#%% Globals
logger = logging.getLogger(__name__)

#%% Functions - setup_dir
def setup_dir(folder: Union[str, Path], recursive: bool = False) -> None:
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
    os.makedirs, os.rmdir, os.remove, pathlib.Path.mkdir, pathlib.Path.rmdir

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
    >>> setup_dir(r"C:\Temp\test_folder") # doctest: +SKIP

    """
    # convert older string API to paths
    if isinstance(folder, str):
        # check for an empty string and exit
        if not folder:
            return
        folder = Path(folder)
    if folder.is_dir():
        # Loop through the contained files/folders
        for this_elem in folder.glob("*"):
            # alias the fullpath of this file element
            this_full_elem = this_elem.resolve()
            # check if a folder or file
            if this_full_elem.is_dir():
                # if a folder, then delete recursively if recursive is True
                if recursive:
                    setup_dir(this_full_elem, recursive=recursive)
                    with contextlib.suppress(FileNotFoundError):
                        this_full_elem.rmdir()
            elif this_full_elem.is_file():
                # if a file, then remove it
                this_full_elem.unlink(missing_ok=True)
            else:
                raise RuntimeError(f'Unexpected file type, neither file nor folder: "{this_full_elem}".')  # pragma: no cover
        logger.log(LogLevel.L1, 'Files/Sub-folders were removed from: "%s"', folder)
    else:
        # create directory if it does not exist
        try:
            folder.mkdir(parents=True)
            logger.log(LogLevel.L1, 'Created directory: "%s"', folder)
        except:  # pragma: no cover
            # re-raise last exception, could try to handle differently in the future
            raise  # pragma: no cover


#%% Functions - fix_rollover
def fix_rollover(data, roll, axis=None, check_accel=False, **kwargs):
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

    def comp_roll(comp, roll_ix, mode):
        # add final field to roll_ix so that final partition can be addressed
        roll_aug = np.hstack((roll_ix, num_el - 1))
        # loop only on original length of roll_ix
        for i in range(roll_ix.size):
            # creates a step function to be added to the input array where each
            # step occurs after a roll over.
            if mode == "+":
                comp[roll_aug[i] + 1 : roll_aug[i + 1] + 1] += i + 1
            elif mode == "-":
                comp[roll_aug[i] + 1 : roll_aug[i + 1] + 1] -= i + 1
            elif mode == "++":
                comp[roll_ix[i] + 1 :] += np.arange(1, num_el - roll_ix[i])
            elif mode == "--":
                comp[roll_ix[i] + 1 :] -= np.arange(1, num_el - roll_ix[i])

    # call recursively when specifying an axis on 2D data
    if axis is not None:
        assert data.ndim == 2, "Must be a 2D array when axis is specified"
        out = np.zeros_like(data)
        if axis == 0:
            for i in range(data.shape[1]):
                out[:, i] = fix_rollover(data[:, i], roll, check_accel=check_accel, **kwargs)
        elif axis == 1:
            for i in range(data.shape[0]):
                out[i, :] = fix_rollover(data[i, :], roll, check_accel=check_accel, **kwargs)
        else:
            raise ValueError(f'Unexpected axis: "{axis}".')
        return out

    # check that input is a vector and initialize compensation variables with the same dimensions
    if data.ndim == 1:
        if data.size == 0:
            return np.array([])
        # t2b means top to bottom rollovers, while b2t means bottom to top rollovers
        num_el = data.size
        comp = np.zeros(data.shape, dtype=int)
    else:
        raise ValueError('Input argument "data" must be a vector.')

    # find indices for top to bottom rollovers, these indices act as partition boundaries
    roll_ix = np.flatnonzero(find_in_range(np.diff(data), max_=-roll / 2))
    # compensation for top to bottom rollovers
    if roll_ix.size > 0:
        comp_roll(comp, roll_ix, "+")
        logger.log(LogLevel.L6, "corrected {} top to bottom rollover(s)".format(roll_ix.size))

    # find indices for bottom to top rollover, these indices act as partition boundaries
    roll_ix = np.flatnonzero(find_in_range(np.diff(data), min_=roll / 2))
    # compensate for bottom to top rollovers
    if roll_ix.size > 0:
        comp_roll(comp, roll_ix, "-")
        logger.log(LogLevel.L6, "corrected {} bottom to top rollover(s)".format(roll_ix.size))

    # create output
    out = data + roll * comp

    # optionally check accelerations
    if check_accel:
        acc = np.diff(out, 2)
        clean_acc = remove_outliers(acc, **kwargs)
        bad_ix = np.flatnonzero(np.isnan(clean_acc) & ~np.isnan(acc))
        if bad_ix.size > 0:
            comp_roll(comp, bad_ix[acc[bad_ix] < 0] + 1, "++")
            comp_roll(comp, bad_ix[acc[bad_ix] > 0] + 1, "--")
            # recreate output
            out = data + roll * comp
            logger.log(LogLevel.L6, "corrected {} rollovers due to acceleration checks".format(bad_ix.size))

    # double check for remaining rollovers
    if np.any(find_in_range(np.diff(out), min_=roll / 2, max_=-roll / 2)):
        logger.log(LogLevel.L6, "A rollover was fixed recursively")
        out = fix_rollover(out, roll, check_accel=check_accel, **kwargs)
    return out


#%% remove_outliers
def remove_outliers(x, /, sigma=3.0, axis=None, *, num_iters=3, return_stats=False, inplace=False, hardmax=None):
    r"""
    Removes the outliers from a data set based on the RMS of the points in the set.

    Parameters
    ----------
    x : array_like
        Input data
    sigma : float, optional
        Standard deviation over which to exclude the data
    axis : int, optional
        Axis to process along
    num_iters : int, optional
        Number of successive iterations to process
    return_stats : bool, optional
        Whether to return additional statistics
    inplace : bool, optional
        Whether to modify the input inplace
    hardmax : float, optional
        A hard absolute limit to exclude any data, meant for completely corrupted points

    Returns
    -------
    y : ndarray of float
        Modified data, with NaNs in-place of outliers
    num_replaced : int
        Number of points removed from the data and replaced with NaNs
    rms_initial : float
        Initial RMS
    rms_removed : float
        RMS after bad points were removed

    Notes
    -----
    #.  Written by David C. Stauffer in March 2021 loosely based on a Matlab version from Tom Trankle.

    Examples
    --------
    >>> from dstauffman import remove_outliers
    >>> import numpy as np
    >>> x = 0.6 * np.random.rand(1000)
    >>> x[5] = 1e5
    >>> x[15] = 1e24
    >>> x[100] = np.nan
    >>> x[200] = np.nan
    >>> (y, num_replaced, rms_initial, rms_removed) = remove_outliers(x, return_stats=True)
    >>> print(y[15])
    nan

    >>> print(num_replaced)
    2

    >>> print(rms_initial > 1e22)
    True

    >>> print(rms_removed < 1)
    True

    """
    x = np.asanyarray(x)
    y = x if inplace else x.copy()
    num_nans = np.count_nonzero(np.isnan(x))
    if hardmax is None:
        num_hard = 0
    else:
        ix_hard = np.greater(np.abs(y), hardmax, where=~np.isnan(y), out=np.zeros(y.shape, dtype=bool))
        y[ix_hard] = np.nan
        num_hard = np.count_nonzero(ix_hard)
    for i in range(num_iters):
        rms_all = rms(y, axis=axis, ignore_nans=True, keepdims=True)
        if i == 0:
            rms_initial = np.squeeze(rms_all)
        ix_bad = np.greater(np.abs(y), rms_all * sigma, out=np.zeros(y.shape, dtype=bool), where=~np.isnan(y))
        y[ix_bad] = np.nan
    rms_removed = rms(y, axis=axis, ignore_nans=True)
    num_replaced = np.count_nonzero(np.isnan(y)) - num_nans
    num_removed = num_replaced - num_hard
    logger.log(LogLevel.L6, "Number of NaNs = %s", num_nans)
    logger.log(LogLevel.L6, "Number exceeding hardmax = %s", num_hard)
    logger.log(LogLevel.L6, "Number of outliers = %s", num_removed)
    if rms_initial.ndim == 0:
        logger.log(LogLevel.L6, "RMS before removal = {:.6g}, after = {:.6g}".format(rms_initial, rms_removed))
    else:
        logger.log(
            LogLevel.L6,
            "RMS before removal = {}, after = {}".format(
                np.array_str(rms_initial, precision=6), np.array_str(rms_removed, precision=6)
            ),
        )
    if return_stats:
        return (y, num_replaced, rms_initial, rms_removed)
    return y


#%% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_utils_log", exit=False)
    doctest.testmod(verbose=False)
