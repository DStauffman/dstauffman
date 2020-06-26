# -*- coding: utf-8 -*-
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
except ImportError: # pragma: no cover
    pass

#%% load_matlab
def load_matlab(filename):
    r"""
    Load simple arrays from a MATLAB v7.3 HDF5 based *.mat file.

    Parameters
    ----------
    filename : str
        Name of the file to load

    Returns
    -------
    out : dict
        Equivalent structure as python dictionary

    Examples
    --------
    >>> filename = r'truth_data.mat'
    >>> out = load_matlab(filename)

    """
    out = {}
    with h5py.File(filename, 'r') as file:
        # loop through keys, keys are the MATLAB variable names, like omega_truth
        for key in file:
            # alias this group
            grp = file[key]
            # check if this is a dataset, meaning its just an array and not a structure
            if isinstance(grp, h5py.Dataset):
                out[key] = grp.value
            else:
                pass # placeholder for complex version of this function that handles structures
    return out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_matlab', exit=False)
    doctest.testmod(verbose=False)
