r"""
Classes related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
from __future__ import annotations
import doctest
from pathlib import Path
from typing import Any, FrozenSet, List, Optional, TYPE_CHECKING, Union
import unittest

from dstauffman import Frozen, HAVE_H5PY, HAVE_NUMPY, is_datetime, load_method, NP_DATETIME_FORM, \
    save_method

if HAVE_H5PY:
    import h5py
if HAVE_NUMPY:
    import numpy as np
if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    _Sets = Union[set, FrozenSet]

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
    innov : (M, N) ndarray
        Time history of the raw innovations
    norm : (M, N) ndarray
        Time history of the normalized innovations
    status : (N,)
        Status of the innovation, such as applied, or reason for rejection
    fploc : (2, N), optional
        Focal plane location for sightings that are on a 2D focal plane
    snr : (N, ), optional
        Information about the signal to noise ratio (SNR) or about the magnitude or brightness

    Notes
    -----
    #.  Written by David C. Stauffer in April 2019.
    #.  Uses Fortran (columnwise) ordering on matrices, as that is the way they will be sliced when
        running within the Kalman filter

    Examples
    --------
    >>> from dstauffman.aerospace import KfInnov
    >>> innov = KfInnov()

    """
    def __init__(self, *, name: str = '', units: str = '', num_innovs: int = 0, num_axes: int = 0, time_dtype: DTypeLike = float):
        r"""Initializes a new KfInnov instance."""
        self.name  = name
        self.chan: Optional[List[str]] = ['' for i in range(num_axes)] if num_axes > 0 else None
        self.units = units
        self.time: Optional[np.ndarray]
        self.innov: Optional[np.ndarray]
        self.norm: Optional[np.ndarray]
        self.status: Optional[np.ndarray]
        if num_innovs > 0:
            self.time   = np.empty(num_innovs, dtype=time_dtype)
            innov_shape = (num_axes, num_innovs) if num_axes > 1 else (num_innovs, )
            self.innov  = np.full(innov_shape, np.nan, dtype=float, order='F')
            self.norm   = np.full(innov_shape, np.nan, dtype=float, order='F')
            self.status = np.empty(num_innovs, dtype=int)
        else:
            self.time   = None
            self.innov  = None
            self.norm   = None
            self.status = None
        self.fploc: Optional[np.ndarray] = None
        self.snr: Optional[np.ndarray]   = None

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
    def __init__(self, *, name: str = '', num_points: int = 0, num_states: int = 0, time_dtype: DTypeLike = float, \
                 active_states: np.ndarray = None, innov_class: Any = None, use_pv: bool = True, **kwargs):
        r"""Initializes a new Kf instance."""
        self.name = name
        self.chan: Optional[List[str]] = ['' for i in range(num_states)] if num_states > 0 else None
        self.time: Optional[np.ndarray]
        self.att: Optional[np.ndarray]
        self.pos: Optional[np.ndarray]
        self.vel: Optional[np.ndarray]
        self.active: Optional[np.ndarray]
        self.state: Optional[np.ndarray]
        self.istate: Optional[np.ndarray]
        self.covar: Optional[np.ndarray]
        if num_points > 0:
            num_active   = num_states if active_states is None else len(active_states)
            state_shape  = (num_active, num_points) if num_active > 1 else (num_points, )
            self.time    = np.empty(num_points, dtype=time_dtype)
            self.att     = np.empty((4, num_points), order='F')
            if use_pv:
                self.pos = np.empty((3, num_points), order='F')
                self.vel = np.empty((3, num_points), order='F')
            self.active  = active_states if active_states is not None else np.arange(num_states)
            self.state   = np.empty(state_shape, order='F')
            self.istate  = np.empty(num_states)
            self.covar   = np.empty(state_shape, order='F')
        else:
            self.time    = None
            self.att     = None
            if use_pv:
                self.pos = None
                self.vel = None
            self.active  = None
            self.state   = None
            self.istate  = None
            self.covar   = None
        self.innov: Any
        if innov_class is None:
            self.innov = KfInnov(time_dtype=time_dtype, **kwargs)
            self._subclasses = frozenset({'innov', })
        elif callable(innov_class):
            self.innov = innov_class(time_dtype=time_dtype, **kwargs)
            self._subclasses = frozenset({'innov', })
        else:
            for (name, func) in innov_class.items():
                setattr(self, name, func(time_dtype=time_dtype, **kwargs))
            self._subclasses = frozenset(innov_class.keys())

    def save(self, filename: Path = None) -> None:
        r"""Save the object to disk as an HDF5 file."""
        # exit if no filename is given
        if filename is None:
            return
        # Save data
        value: Any
        with h5py.File(filename, 'w') as file:
            grp = file.create_group('self')
            for key in vars(self):
                if key == '_subclasses':
                    # TODO: update to always write this field first
                    value = [x.encode('utf-8') for x in getattr(self, key)]
                    grp.create_dataset(key, data=value)
                elif key in self._subclasses:
                    # handle substructures
                    sub = getattr(self, key)
                    inner_grp = grp.create_group(key)
                    for subkey in vars(sub):
                        value = getattr(sub, subkey)
                        if value is not None:
                            if subkey in {'chan'}:
                                value = [x.encode('utf-8') for x in value]
                            elif subkey in {'time'} and is_datetime(value):
                                value = value.copy().astype(np.int64)
                            inner_grp.create_dataset(subkey, data=value)
                else:
                    # normal values
                    value = getattr(self, key)
                    if value is not None:
                        # special case to handle lists of strings
                        if key in {'chan'}:
                            value = [x.encode('utf-8') for x in value]
                        elif key in {'time'} and is_datetime(value):
                            value = value.copy().astype(np.int64)
                        grp.create_dataset(key, data=value)

    @classmethod
    def load(cls, filename: Path = None, subclasses: _Sets = frozenset({'innov'})) -> Kf:
        r"""Load the object from disk."""
        if filename is None:
            raise ValueError('No file specified to load.')
        # Load data
        out = cls()  # TODO: dynamically determine subclass field names and pv option?
        with h5py.File(filename, 'r') as file:
            for (key, grp) in file.items():
                for field in grp:
                    if field in subclasses:
                        inner_grp = grp[field]
                        for subfield in inner_grp:
                            value = inner_grp[subfield][()]
                            if subfield in {'chan'}:
                                value = [x.decode('utf-8') for x in value]
                            elif subfield in {'time'}:
                                if value.dtype == np.int64:
                                    value.dtype = NP_DATETIME_FORM
                            setattr(getattr(out, field), subfield, value)
                    else:
                        value = grp[field][()]
                        if field in {'chan'}:
                            value = [x.decode('utf-8') for x in value]
                        elif field in {'time'}:
                            if value.dtype == np.int64:
                                value.dtype = NP_DATETIME_FORM
                        elif isinstance(value, bytes):
                            value = value.decode('utf-8')
                        setattr(out, field, value)
        return out

#%% Classes - KfRecord
class KfRecord(Frozen):
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
    def __init__(self, num_points: int = 0, num_states: int = 0, num_active: int = 0, num_axes: int = 0, \
            time_dtype: DTypeLike = float):
        self.time: Optional[np.ndarray]
        self.P: Optional[np.ndarray]
        self.stm: Optional[np.ndarray]
        self.H: Optional[np.ndarray]
        self.Pz: Optional[np.ndarray]
        self.K: Optional[np.ndarray]
        self.z: Optional[np.ndarray]
        if num_points > 0:
            self.time = np.empty(num_points, dtype=time_dtype)
            self.P    = np.empty((num_active, num_active, num_points), order='F')
            self.stm  = np.empty((num_active, num_active, num_points), order='F')
            self.H    = np.empty((num_axes, num_states, num_points), order='F')
            self.Pz   = np.empty((num_axes, num_axes, num_points), order='F')
            self.K    = np.empty((num_active, num_axes, num_points), order='F')
            self.z    = np.empty((num_axes, num_points), order='F')
        else:
            self.time = None
            self.P    = None
            self.stm  = None
            self.H    = None
            self.Pz   = None
            self.K    = None
            self.z    = None

    def keep_subset(self, ix_keep: np.ndarray) -> None:
        r"""Returns only the specified indices (likely only accepted measurements)."""
        self.time = self.time[ix_keep].copy() if self.time is not None else None
        self.P    = self.P[:, :, ix_keep].copy() if self.P is not None else None
        self.stm  = self.stm[:, :, ix_keep].copy() if self.stm is not None else None
        self.H    = self.H[:, :, ix_keep].copy() if self.H is not None else None
        self.Pz   = self.Pz[:, :, ix_keep].copy() if self.Pz is not None else None
        self.K    = self.K[:, :, ix_keep].copy() if self.K  is not None else None
        self.z    = self.z[:, ix_keep].copy() if self.z is not None else None

    def save(self, filename: Path = None, use_hdf5: bool = True) -> None:
        r"""
        Save the object to disk.

        Parameters
        ----------
        filename : classs pathlib.Path
            Name of the file to save
        use_hdf5 : bool, optional, defaults to False
            Write as *.hdf5 instead of *.pkl

        """
        convert_times = hasattr(self.time, 'dtype') and np.issubdtype(self.time.dtype, np.datetime64)  # type: ignore[union-attr]
        if convert_times:
            orig_type = self.time.dtype  # type: ignore[union-attr]
            self.time.dtype = np.int64  # type: ignore[misc, union-attr]
        save_method(self, filename=filename, use_hdf5=use_hdf5)
        if convert_times:
            self.time.dtype = orig_type  # type: ignore[misc, union-attr]

    @classmethod
    def load(cls, filename: Path = None, use_hdf5: bool = True) -> KfRecord:
        r"""
        Load the object from disk.

        Parameters
        ----------
        filename : classs pathlib.Path
            Name of the file to load
        use_hdf5 : bool, optional, defaults to False
            Write as *.hdf5 instead of *.pkl

        """
        out: KfRecord = load_method(cls, filename=filename, use_hdf5=use_hdf5)
        if hasattr(out.time, 'dtype') and out.time.dtype == np.int64:  # type: ignore[union-attr]
            out.time.dtype = NP_DATETIME_FORM  # type: ignore[misc, union-attr]
        return out

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_classes', exit=False)
    doctest.testmod(verbose=False)
