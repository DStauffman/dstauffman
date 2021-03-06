r"""
Classes related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
from __future__ import annotations
import doctest
from typing import Callable, ClassVar, TYPE_CHECKING
import unittest

if TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg

from dstauffman import Frozen, HAVE_H5PY, HAVE_NUMPY, SaveAndLoad

if HAVE_H5PY:
    import h5py
if HAVE_NUMPY:
    import numpy as np

#%% KfInnov
class KfInnov(Frozen):
    r"""
    A class for Kalman Filter innovations outputs.

    Attributes
    ----------
    name : str
        Name of the innovation structure, often specifying which sensor it comes from, like GPS
    chan : [M, ] list of str
        Names of the different axes found within the innovations
    units : str
        Units for the innovations
    time : (N, ) ndarray
        Time vector
    innov : (N, M) ndarray
        Time history of the raw innovations
    norm : (N, M) ndarray
        Time history of the normalized innovations
    status : (N,)
        Status of the innovation, such as applied, or reason for rejection

    Examples
    --------
    >>> from dstauffman.aerospace import KfInnov
    >>> innov = KfInnov()

    """
    def __init__(self, *, name='', units='', num_innovs=0, num_axes=0, time_dtype=float):
        r"""Initializes a new KfInnov instance."""
        self.name   = name
        self.chan   = ['' for i in range(num_axes)] if num_axes > 0 else None
        self.units  = units
        if num_innovs > 0:
            self.time   = np.empty(num_innovs, dtype=time_dtype)
            innov_shape = (num_axes, num_innovs) if num_axes > 1 else (num_innovs, )
            self.innov  = np.full(innov_shape, np.nan, dtype=float)
            self.norm   = np.full(innov_shape, np.nan, dtype=float)
            self.status = np.empty(num_innovs, dtype=int)
        else:
            self.time   = None
            self.innov  = None
            self.norm   = None
            self.status = None

#%% Kf
class Kf(Frozen):
    r"""
    A class for doing Kalman Filter analysis.

    Attributes
    ----------
    name : str
        Name of the structure, used when comparing multiple sources or runs
    chan : list of str
        Name of the states in the state and covar fields
    time : (N, ) ndarray
        Time vector
    att : (4, N) ndarray
        Attitude quaternion history
    pos : (3, N) ndarray
        Position history
    vel : (3, N) ndarray
        Velocity history
    innov : class KfInnov
        Innovation history for GPS measurements
    state : (N, M) ndarray
        State history
    covar : (N, M) ndarray
        Covariance history

    Examples
    --------
    >>> from dstauffman.aerospace import Kf
    >>> kf = Kf()

    """
    def __init__(self, *, name='', num_points=0, num_states=0, time_dtype=float, active_states=None, innov_class=None, **kwargs):
        r"""Initializes a new Kf instance."""
        self.name = name
        self.chan = ['' for i in range(num_states)] if num_states > 0 else None
        if num_points > 0:
            num_active  = num_states if active_states is None else len(active_states)
            state_shape = (num_active, num_points) if num_active > 1 else (num_points, )
            self.time   = np.empty(num_points, dtype=time_dtype)
            self.att    = np.empty((4, num_points), dtype=float)
            self.pos    = None # TODO: flag to enable?
            self.vel    = None # TODO: flag to enable?
            self.active = active_states if active_states is not None else np.arange(num_states)
            self.state  = np.empty(state_shape, dtype=float)
            self.istate = np.empty(num_states, dtype=float)
            self.covar  = np.empty(state_shape, dtype=float)
        else:
            self.time   = None
            self.att    = None
            self.pos    = None
            self.vel    = None
            self.active = None
            self.state  = None
            self.istate = None
            self.covar  = None
        if innov_class is None:
            self.innov  = KfInnov(time_dtype=time_dtype, **kwargs)
        else:
            self.innov  = innov_class(time_dtype=time_dtype, **kwargs)

    def save(self, filename='', subclasses=frozenset({'innov'})):
        r"""Save the object to disk as an HDF5 file."""
        # exit if no filename is given
        if not filename:
            return
        # Save data
        with h5py.File(filename, 'w') as file:
            grp = file.create_group('self')
            for key in vars(self):
                if key in subclasses:
                    # handle substructures
                    sub = getattr(self, key)
                    inner_grp = grp.create_group(key)
                    for subkey in vars(sub):
                        value = getattr(sub, subkey)
                        if value is not None:
                            if subkey in {'chan'}:
                                value = [x.encode('utf-8') for x in value]
                            inner_grp.create_dataset(subkey, data=value)
                else:
                    # normal values
                    value = getattr(self, key)
                    if value is not None:
                        # special case to handle lists of strings
                        if key in {'chan'}:
                            value = [x.encode('utf-8') for x in value]
                        grp.create_dataset(key, data=value)

    @classmethod
    def load(cls, filename='', subclasses=frozenset({'innov'})):
        r"""Load the object from disk."""
        if not filename:
            raise ValueError('No file specified to load.')
        # Load data
        out = cls()
        with h5py.File(filename, 'r') as file:
            for (key, grp) in file.items():
                for field in grp:
                    if field in subclasses:
                        inner_grp = grp[field]
                        for subfield in inner_grp:
                            value = inner_grp[subfield][()]
                            if subfield in {'chan'}:
                                value = [x.decode('utf-8') for x in value]
                            setattr(getattr(out, field), subfield, value)
                    else:
                        value = grp[field][()]
                        if field in {'chan'}:
                            value = [x.decode('utf-8') for x in value]
                        setattr(out, field, value)
        return out

#%% Classes - KfRecord
class KfRecord(Frozen, metaclass=SaveAndLoad):
    r"""
    Full records of the Kalman Filter for use in a backards information smoother.

    Attributes
    ----------
    time : ndarray
        time points for all subfields
    stm : (n_state, n_state) ndarray
        State transition matrix data
    P : (n_state, n_state) ndarray
        Filter error covariance matrix
    H : (n_meas, n_state) ndarray
        Measurement distribution matrix
    Pz : (n_meas, n_meas) ndarray
        Innovation covariance matrix
    K : (n_state, n_meas) ndarray
        Kalman gain matrix
    z : (n_meas, ) ndarray
        innovation vector

    Examples
    --------
    >>> from dstauffman.aerospace import KfRecord
    >>> kf_record = KfRecord()

    """
    load: ClassVar[Callable[[str, DefaultNamedArg(bool, 'use_hdf5')], KfRecord]]
    save: Callable[[KfRecord, str, DefaultNamedArg(bool, 'use_hdf5')], None]

    def __init__(self, num_points=0, num_states=0, num_active=0, num_axes=0, time_dtype=float):
        if num_points > 0:
            self.time = np.empty(num_points, dtype=time_dtype)
            self.P    = np.empty((num_active, num_active, num_points))
            self.stm  = np.empty((num_active, num_active, num_points))
            self.H    = np.empty((num_axes, num_states, num_points))
            self.Pz   = np.empty((num_axes, num_axes, num_points))
            self.K    = np.empty((num_active, num_axes, num_points))
            self.z    = np.empty((num_axes, num_points))
        else:
            self.time = None
            self.P    = None
            self.stm  = None
            self.H    = None
            self.Pz   = None
            self.K    = None
            self.z    = None

    def keep_subset(self, ix_keep):
        r"""Returns only the specified indices (likely only accepted measurements)."""
        self.time = self.time[ix_keep]
        self.P    = self.P[:, :, ix_keep]
        self.stm  = self.stm[:, :, ix_keep]
        self.H    = self.H[:, :, ix_keep]
        self.Pz   = self.Pz[:, :, ix_keep]
        self.K    = self.K[:, :, ix_keep]
        self.z    = self.z[:, ix_keep]

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_classes', exit=False)
    doctest.testmod(verbose=False)
