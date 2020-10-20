r"""
Functions that make it easier to deal with Matlab data.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
from __future__ import annotations
import contextlib
import doctest
from typing import Any, Dict, List, Optional
import unittest

with contextlib.suppress(ModuleNotFoundError):
    import h5py
with contextlib.suppress(ModuleNotFoundError):
    import numpy as np

#%% load_matlab
def load_matlab(filename: str, varlist: List[str] = None, *, squeeze: bool = True, enums: Dict[str, Any] = None) -> Dict[str, Any]:
    r"""
    Load simple arrays from a MATLAB v7.3 HDF5 based *.mat file.

    Parameters
    ----------
    filename : str
        Name of the file to load
    varlist : list of str, optional
        Name of the variables to load
    squeeze : bool, optional, default is True
        Whether to squeeze any singleton vectors down a dimension

    Returns
    -------
    out : dict
        Equivalent structure as python dictionary

    Examples
    --------
    >>> from dstauffman import load_matlab, get_tests_dir
    >>> import os
    >>> filename = os.path.join(get_tests_dir(), 'test_numbers.mat')
    >>> out = load_matlab(filename)
    >>> print(out['row_nums'][1])
    2.2

    """
    def _load(file: h5py.Group, varlist: Optional[List[str]], squeeze: bool, enums: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        r"""Wrapped subfunction so it can be called recursively."""
        # initialize output
        out: Dict[str, Any] = {}
        # loop through keys, keys are the MATLAB variable names, like TELM
        for key in file:
            # skip keys that are not in the given varlist
            if varlist is not None and key not in varlist:
                continue
            # if no varlist (thus loading every key), still skip those that start with #
            if varlist is None and key in {'#refs#', '#subsystem#'}:
                continue
            # alias this group
            grp = file[key]
            # check if this is a dataset, meaning its just an array and not a structure
            if isinstance(grp, h5py.Dataset):
                # Note: data is transposed due to how Matlab stores columnwise
                values = grp[()].T
                out[key] = np.squeeze(values) if squeeze else values
            elif 'EnumerationInstanceTag' in grp:
                # likely a MATLAB enumerator???
                class_name = grp.attrs['MATLAB_class'].decode()
                if enums is None or class_name not in enums:
                    raise ValueError(f'Tried to load a MATLAB enumeration class called "{class_name}" without a decoder ring, pass in via `enums`.')
                ix       = grp['ValueIndices'][()].T
                values   = np.array([enums[class_name][x] for x in ix.flatten()]).reshape(ix.shape)
                out[key] = np.squeeze(values) if squeeze else values
            else:
                # call recursively
                out[key] = load_matlab(grp, varlist=None, squeeze=squeeze, enums=enums)
                        #if isinstance(this_data, dict) and 'ValueNames' in this_data:
        return out

    if not isinstance(filename, h5py.Group):
        with h5py.File(filename, 'r') as file:
            # normal method
            out = _load(file=file, varlist=varlist, squeeze=squeeze, enums=enums)
    else:
        # recursive call method where the file is already opened to a given group
        out = _load(file=filename, varlist=varlist, squeeze=squeeze, enums=enums)
    return out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_matlab', exit=False)
    doctest.testmod(verbose=False)
