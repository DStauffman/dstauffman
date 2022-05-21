r"""
Functions that make it easier to deal with Matlab data.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
from __future__ import annotations

import doctest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import unittest

from dstauffman.constants import HAVE_H5PY, HAVE_NUMPY

if HAVE_H5PY:
    import h5py
if HAVE_NUMPY:
    import numpy as np

#%% load_matlab
def load_matlab(
    filename: Union[str, Path],
    varlist: Union[List[str], Set[str], Tuple[str]] = None,
    *,
    squeeze: bool = True,
    enums: Dict[str, Any] = None,
) -> Dict[str, Any]:
    r"""
    Load simple arrays from a MATLAB v7.3 HDF5 based *.mat file.

    Parameters
    ----------
    filename : class pathlib.Path
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
    >>> filename = get_tests_dir() / "test_numbers.mat"
    >>> out = load_matlab(filename)
    >>> print(out["row_nums"][1])
    2.2

    """

    def _load(
        file: h5py.Group,
        varlist: Optional[Union[List[str], Set[str], Tuple[str]]],
        squeeze: bool,
        enums: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        r"""Wrapped subfunction so it can be called recursively."""
        # initialize output
        out: Dict[str, Any] = {}
        # loop through keys, keys are the MATLAB variable names, like TELM
        for key in file:
            # skip keys that are not in the given varlist
            if varlist is not None and key not in varlist:
                continue
            # if no varlist (thus loading every key), still skip those that start with #
            if varlist is None and key in {"#refs#", "#subsystem#"}:
                continue
            # alias this group
            grp = file[key]
            # check if this is a dataset, meaning its just an array and not a structure
            if isinstance(grp, h5py.Dataset):
                # Note: data is transposed due to how Matlab stores columnwise
                values = grp[()].T
                # check for cell array references
                if isinstance(values.flat[0], h5py.Reference):
                    # TODO: for now, always collapse to 1D cell array as a list
                    temp = [file[item] for item in values.flat]
                    temp2 = []
                    for x in temp:
                        if isinstance(x, h5py.Group):
                            temp2.append(load_matlab(x, varlist=None, squeeze=squeeze, enums=enums))
                        else:
                            data = x[()].T
                            temp2.append(np.squeeze(data) if squeeze else data)
                    out[key] = temp2
                else:
                    out[key] = np.squeeze(values) if squeeze else values
            elif "EnumerationInstanceTag" in grp:
                # likely a MATLAB enumerator???
                class_name = grp.attrs["MATLAB_class"].decode()
                if enums is None or class_name not in enums:
                    raise ValueError(
                        f'Tried to load a MATLAB enumeration class called "{class_name}" without a decoder ring, pass in via `enums`.'
                    )
                ix = grp["ValueIndices"][()].T
                values = np.array([enums[class_name][x] for x in ix.flatten()]).reshape(ix.shape)
                out[key] = np.squeeze(values) if squeeze else values
            else:
                # call recursively
                out[key] = load_matlab(grp, varlist=None, squeeze=squeeze, enums=enums)
        return out

    if not isinstance(filename, h5py.Group):
        with h5py.File(filename, "r") as file:
            # normal method
            out = _load(file=file, varlist=varlist, squeeze=squeeze, enums=enums)
    else:
        # recursive call method where the file is already opened to a given group
        out = _load(file=filename, varlist=varlist, squeeze=squeeze, enums=enums)
    return out


#%% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_matlab", exit=False)
    doctest.testmod(verbose=False)
