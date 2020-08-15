r"""
Functions that make it easier to deal with Matlab data.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import doctest
import unittest

try:
    import h5py
except ModuleNotFoundError: # pragma: no cover
    pass
try:
    import numpy as np
except ModuleNotFoundError: # pragma: no cover
    pass

#%% load_matlab
def load_matlab(filename, varlist=None, squeeze=True):
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
    # initialize output
    out = {}
    with h5py.File(filename, 'r') as file:
        # loop through keys, keys are the MATLAB variable names, like TELM
        for key in file:
            # skip keys that are not in the given varlist
            if varlist is not None and key not in varlist:
                continue
            # if no varlist (thus loading every key), still skip those that start with #
            if varlist is None and key.startswith('#'):
                continue
            # alias this group
            grp = file[key]
            # check if this is a dataset, meaning its just an array and not a structure
            if isinstance(grp, h5py.Dataset):
                # Note: data is transposed due to how Matlab stores columnwise
                values = grp[()].T
                out[key] = np.squeeze(values) if squeeze else values
            else:
                # initialize the structure for output and save the name
                this_struct = {}
                # loop through fields
                # fields are the structure subfield names, like date, gyro_counts, etc.
                for field in grp:
                    # alias this dataset
                    this_data = grp[field]
                    if isinstance(this_data, h5py.Dataset):
                        # normal method, just store
                        values = this_data[()].T
                    elif isinstance(this_data, h5py.Group):
                        # likely a MATLAB enumerator???
                        enums   = this_data['ValueNames'][()]
                        indices = this_data['ValueIndices'][()]
                        values  = enums.T[indices.T]
                    this_struct[field] = np.squeeze(values) if squeeze else values
                out[key] = this_struct
    return out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_matlab', exit=False)
    doctest.testmod(verbose=False)
