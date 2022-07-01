r"""
Classes related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
from __future__ import annotations

import copy
import doctest
from pathlib import Path
from typing import Any, FrozenSet, List, Literal, Optional, overload, Set, Tuple, TYPE_CHECKING, TypeVar, Union
import unittest

from dstauffman import chop_time, Frozen, HAVE_H5PY, HAVE_NUMPY, is_datetime, load_method, NP_DATETIME_FORM, save_method

if HAVE_H5PY:
    import h5py
if HAVE_NUMPY:
    import numpy as np
if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    _Chan = Union[List[str], Tuple[str, ...]]
    _Sets = Union[Set[str], FrozenSet[str]]
    _Time = Union[float, np.datetime64]
    _T = TypeVar("_T")

#%% Support Functions
@overload
def _chop_wrapper(
    orig: _T,
    exclude: _Sets,
    ti: _Time = ...,
    tf: _Time = ...,
    *,
    include_last: bool = ...,
    inplace: bool = ...,
    return_ends: Literal[True],
    subclasses: _Sets = ...,
) -> Tuple[_T, _T, _T]:
    ...


@overload
def _chop_wrapper(
    orig: _T,
    exclude: _Sets,
    ti: _Time = ...,
    tf: _Time = ...,
    *,
    include_last: bool = ...,
    inplace: bool = ...,
    return_ends: Literal[False] = ...,
    subclasses: _Sets = ...,
) -> _T:
    ...


def _chop_wrapper(
    orig: _T,
    exclude: _Sets,
    ti: _Time = None,
    tf: _Time = None,
    *,
    include_last: bool = True,
    inplace: bool = False,
    return_ends: bool = False,
    subclasses: _Sets = None,
) -> Union[_T, Tuple[_T, _T, _T]]:
    assert orig.time is not None, "You can't chop an uninitialized time field."  # type: ignore[attr-defined]
    use_dates = is_datetime(orig.time)  # type: ignore[attr-defined]
    if ti is None:
        ti = np.datetime64("nat") if use_dates else -np.inf
    if tf is None:
        tf = np.datetime64("nat") if use_dates else np.inf
    assert ti is not None
    assert tf is not None
    if return_ends:
        left = copy.deepcopy(orig)
        right = copy.deepcopy(orig)
        tl = np.datetime64("nat") if use_dates else -np.inf
        tr = np.datetime64("nat") if use_dates else np.inf
        chop_time(left, time_field="time", exclude=exclude, ti=tl, tf=ti, right=False)  # type: ignore[arg-type]
        chop_time(right, time_field="time", exclude=exclude, ti=tf, tf=tr, left=False)  # type: ignore[arg-type]
    out = orig if inplace else copy.deepcopy(orig)
    chop_time(out, time_field="time", exclude=exclude, ti=ti, tf=tf, right=include_last)
    if subclasses is not None:
        for sub in subclasses:
            temp = getattr(out, sub).chop(ti=ti, tf=tf, include_last=include_last, inplace=inplace, return_ends=return_ends)
            if return_ends:
                setattr(left, sub, temp[0])
                setattr(out, sub, temp[1])
                setattr(right, sub, temp[2])
            else:
                setattr(out, sub, temp)
    if return_ends:
        return (left, out, right)
    return out


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

    def __init__(
        self,
        *,
        name: str = "",
        units: str = "",
        chan: _Chan = None,
        num_innovs: int = 0,
        num_axes: int = 0,
        time_dtype: DTypeLike = float,
    ):
        r"""Initializes a new KfInnov instance."""
        self.name = name
        self.chan: Optional[_Chan]
        if chan is not None:
            self.chan = chan
        elif num_axes > 0:
            self.chan = ["" for i in range(num_axes)]
        else:
            self.chan = None
        self.units = units
        self.time: Optional[np.ndarray]
        self.innov: Optional[np.ndarray]
        self.norm: Optional[np.ndarray]
        self.status: Optional[np.ndarray]
        # fmt: off
        if num_innovs > 0:
            self.time   = np.empty(num_innovs, dtype=time_dtype)
            innov_shape = (num_axes, num_innovs) if num_axes > 1 else (num_innovs,)
            self.innov  = np.full(innov_shape, np.nan, dtype=float, order="F")
            self.norm   = np.full(innov_shape, np.nan, dtype=float, order="F")
            self.status = np.empty(num_innovs, dtype=int)
        else:
            self.time   = None
            self.innov  = None
            self.norm   = None
            self.status = None
        # fmt: on
        self.fploc: Optional[np.ndarray] = None
        self.snr: Optional[np.ndarray] = None

    def combine(self, kfinnov2: KfInnov, /, *, inplace: bool = False) -> KfInnov:
        r"""Combines two KfInnov structures together."""
        # allow an empty structure to be passed through
        if self.time is None:
            if inplace:
                for (key, value) in vars(kfinnov2).items():
                    setattr(self, key, value)
            return kfinnov2  # TODO: make a copy?
        # concatenate fields
        if inplace:
            kfinnov = self
        else:
            kfinnov = copy.deepcopy(self)
        if kfinnov2.time is None:
            return kfinnov
        assert kfinnov.time is not None
        assert kfinnov2.time is not None
        kfinnov.time = np.hstack((self.time, kfinnov2.time))
        # TODO: deal with Nones
        kfinnov.innov = np.column_stack((self.innov, kfinnov2.innov))  # type: ignore[arg-type]
        kfinnov.norm = np.column_stack((self.norm, kfinnov2.norm))  # type: ignore[arg-type]
        kfinnov.status = np.hstack((self.status, kfinnov2.status))  # type: ignore[arg-type]
        if self.fploc is not None and kfinnov2.fploc is not None:
            kfinnov.fploc = np.column_stack((self.fploc, kfinnov2.fploc))
        if self.snr is not None and kfinnov2.snr is not None:
            kfinnov.snr = np.hstack((self.snr, kfinnov2.snr))
        return kfinnov

    @overload
    def chop(
        self, ti: _Time = ..., tf: _Time = ..., *, include_last: bool = ..., inplace: bool = ..., return_ends: Literal[True]
    ) -> Tuple[KfInnov, KfInnov, KfInnov]:
        ...

    @overload
    def chop(
        self,
        ti: _Time = ...,
        tf: _Time = ...,
        *,
        include_last: bool = ...,
        inplace: bool = ...,
        return_ends: Literal[False] = ...,
    ) -> KfInnov:
        ...

    def chop(
        self, ti: _Time = None, tf: _Time = None, *, include_last: bool = True, inplace: bool = False, return_ends: bool = False
    ) -> Union[KfInnov, Tuple[KfInnov, KfInnov, KfInnov]]:
        r"""Chops the KfInnov data structure to the given time bounds."""
        exclude = frozenset({"name", "chan", "units"})
        out = _chop_wrapper(
            self, exclude=exclude, ti=ti, tf=tf, include_last=include_last, inplace=inplace, return_ends=return_ends
        )  # type: ignore[call-overload]
        return out  # type: ignore[no-any-return]


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
    active : (M, ) ndarray
        Active states
    state : (M, N) ndarray
        Active state history
    istate : (M, ) ndarray
        Initial state values
    covar : (N, M) ndarray
        Covariance history
    innov : class KfInnov
        Innovation history for GPS measurements

    Examples
    --------
    >>> from dstauffman.aerospace import Kf
    >>> kf = Kf()

    """

    def __init__(
        self,
        *,
        name: str = "",
        chan: _Chan = None,
        num_points: int = 0,
        num_states: int = 0,
        time_dtype: DTypeLike = float,
        active_states: np.ndarray = None,
        innov_class: Any = None,
        use_pv: bool = True,
        innov_chan: _Chan = None,
        **kwargs,
    ):
        r"""Initializes a new Kf instance."""
        self.name = name
        self.chan: Optional[Union[List[str], Tuple[str, ...]]]
        if chan is not None:
            self.chan = chan
        elif num_states > 0:
            self.chan = ["" for i in range(num_states)]
        else:
            self.chan = None
        self.time: Optional[np.ndarray]
        self.att: Optional[np.ndarray]
        self.pos: Optional[np.ndarray]
        self.vel: Optional[np.ndarray]
        self.active: Optional[np.ndarray]
        self.state: Optional[np.ndarray]
        self.istate: Optional[np.ndarray]
        self.covar: Optional[np.ndarray]
        # fmt: off
        if num_points > 0:
            num_active   = num_states if active_states is None else len(active_states)
            state_shape  = (num_active, num_points) if num_active > 1 else (num_points,)
            self.time    = np.empty(num_points, dtype=time_dtype)
            self.att     = np.empty((4, num_points), order="F")
            if use_pv:
                self.pos = np.empty((3, num_points), order="F")
                self.vel = np.empty((3, num_points), order="F")
            self.active  = active_states if active_states is not None else np.arange(num_states)
            self.state   = np.empty(state_shape, order="F")
            self.istate  = np.empty(num_states)
            self.covar   = np.empty(state_shape, order="F")
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
        # fmt: on
        if innov_class is None:
            self.innov = KfInnov(time_dtype=time_dtype, chan=innov_chan, **kwargs)
            self._subclasses = frozenset({"innov",})  # fmt: skip
        elif callable(innov_class):
            self.innov = innov_class(time_dtype=time_dtype, chan=innov_chan, **kwargs)
            self._subclasses = frozenset({"innov",})  # fmt: skip
        else:
            for (innov_name, func) in innov_class.items():
                setattr(self, innov_name, func(time_dtype=time_dtype, chan=innov_chan, **kwargs))
            self._subclasses = frozenset(innov_class.keys())

    def save(self, filename: Path = None) -> None:
        r"""Save the object to disk as an HDF5 file."""
        # exit if no filename is given
        if filename is None:
            return
        # Save data
        value: Any
        with h5py.File(filename, "w") as file:
            grp = file.create_group("self")
            for key in vars(self):
                if key == "_subclasses":
                    # TODO: update to always write this field first
                    value = [x.encode("utf-8") for x in getattr(self, key)]
                    grp.create_dataset(key, data=value)
                elif key in self._subclasses:
                    # handle substructures
                    sub = getattr(self, key)
                    inner_grp = grp.create_group(key)
                    for subkey in vars(sub):
                        value = getattr(sub, subkey)
                        if value is not None:
                            if subkey in {"chan"}:
                                value = [x.encode("utf-8") for x in value]
                            elif subkey in {"time"} and is_datetime(value):
                                value = value.copy().astype(np.int64)
                            inner_grp.create_dataset(subkey, data=value)
                else:
                    # normal values
                    value = getattr(self, key)
                    if value is not None:
                        # special case to handle lists of strings
                        if key in {"chan"}:
                            value = [x.encode("utf-8") for x in value]
                        elif key in {"time"} and is_datetime(value):
                            value = value.copy().astype(np.int64)
                        grp.create_dataset(key, data=value)

    @classmethod
    def load(cls, filename: Path = None, subclasses: _Sets = frozenset({"innov"})) -> Kf:
        r"""Load the object from disk."""
        if filename is None:
            raise ValueError("No file specified to load.")
        # Load data
        out = cls()  # TODO: dynamically determine subclass field names and pv option?
        with h5py.File(filename, "r") as file:
            for grp in file.values():  # pylint: disable=too-many-nested-blocks
                for field in grp:
                    if field in subclasses:
                        inner_grp = grp[field]
                        for subfield in inner_grp:
                            value = inner_grp[subfield][()]
                            if subfield in {"chan"}:
                                value = [x.decode("utf-8") for x in value]
                            elif subfield in {"time"}:
                                if value.dtype == np.int64:
                                    value.dtype = NP_DATETIME_FORM
                            elif isinstance(value, bytes):
                                value = value.decode("utf-8")
                            setattr(getattr(out, field), subfield, value)
                    else:
                        value = grp[field][()]
                        if field in {"chan"}:
                            value = [x.decode("utf-8") for x in value]
                        elif field in {"time"}:
                            if value.dtype == np.int64:
                                value.dtype = NP_DATETIME_FORM
                        elif isinstance(value, bytes):
                            value = value.decode("utf-8")
                        setattr(out, field, value)
        return out

    def combine(self, kf2: Kf, /, *, inplace: bool = False) -> Kf:
        r"""Combines two Kf structures together."""
        # allow an empty structure to be passed through
        if self.time is None:
            if inplace:
                for (key, value) in vars(kf2).items():
                    setattr(self, key, value)
            return kf2  # TODO: make a copy?
        # concatenate fields
        if inplace:
            kf = self
        else:
            kf = copy.deepcopy(self)
        if kf2.time is None:
            return kf
        assert kf.time is not None
        assert kf2.time is not None
        kf.time = np.hstack((self.time, kf2.time))
        kf.istate = self.istate.copy() if self.istate is not None else None
        kf.active = self.active.copy() if self.active is not None else None  # TODO: assert that they are the same?
        for field in frozenset({"att", "pos", "vel", "state", "covar"}):
            if (x := getattr(self, field)) is not None and (y := getattr(kf2, field)) is not None:
                setattr(kf, field, np.column_stack((x, y)))
        for sub in self._subclasses:
            setattr(kf, sub, getattr(self, sub).combine(getattr(kf2, sub), inplace=inplace))
        return kf

    @overload
    def chop(
        self, ti: _Time = ..., tf: _Time = ..., *, include_last: bool = ..., inplace: bool = ..., return_ends: Literal[True]
    ) -> Tuple[Kf, Kf, Kf]:
        ...

    @overload
    def chop(
        self,
        ti: _Time = ...,
        tf: _Time = ...,
        *,
        include_last: bool = ...,
        inplace: bool = ...,
        return_ends: Literal[False] = ...,
    ) -> Kf:
        ...

    def chop(
        self, ti: _Time = None, tf: _Time = None, *, include_last: bool = True, inplace: bool = False, return_ends: bool = False
    ) -> Union[Kf, Tuple[Kf, Kf, Kf]]:
        r"""Chops the Kf structure to the given time bounds."""
        exclude = frozenset({"name", "chan", "active", "istate"} | self._subclasses)
        out = _chop_wrapper(
            self,
            exclude=exclude,
            ti=ti,
            tf=tf,
            include_last=include_last,
            inplace=inplace,
            return_ends=return_ends,
            subclasses=self._subclasses,
        )  # type: ignore[call-overload]
        return out  # type: ignore[no-any-return]


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

    def __init__(
        self, num_points: int = 0, num_states: int = 0, num_active: int = 0, num_axes: int = 0, time_dtype: DTypeLike = float
    ):
        self.time: Optional[np.ndarray]
        self.P: Optional[np.ndarray]
        self.stm: Optional[np.ndarray]
        self.H: Optional[np.ndarray]
        self.Pz: Optional[np.ndarray]
        self.K: Optional[np.ndarray]
        self.z: Optional[np.ndarray]
        # fmt: off
        if num_points > 0:
            self.time = np.empty(num_points, dtype=time_dtype)
            self.P    = np.empty((num_active, num_active, num_points), order="F")
            self.stm  = np.empty((num_active, num_active, num_points), order="F")
            self.H    = np.empty((num_axes, num_states, num_points), order="F")
            self.Pz   = np.empty((num_axes, num_axes, num_points), order="F")
            self.K    = np.empty((num_active, num_axes, num_points), order="F")
            self.z    = np.empty((num_axes, num_points), order="F")
        else:
            self.time = None
            self.P    = None
            self.stm  = None
            self.H    = None
            self.Pz   = None
            self.K    = None
            self.z    = None
        # fmt: on

    def keep_subset(self, ix_keep: np.ndarray) -> None:
        r"""Returns only the specified indices (likely only accepted measurements)."""
        # fmt: off
        self.time = self.time[ix_keep].copy() if self.time is not None else None
        self.P    = self.P[:, :, ix_keep].copy() if self.P is not None else None
        self.stm  = self.stm[:, :, ix_keep].copy() if self.stm is not None else None
        self.H    = self.H[:, :, ix_keep].copy() if self.H is not None else None
        self.Pz   = self.Pz[:, :, ix_keep].copy() if self.Pz is not None else None
        self.K    = self.K[:, :, ix_keep].copy() if self.K is not None else None
        self.z    = self.z[:, ix_keep].copy() if self.z is not None else None
        # fmt: on

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
        convert_times = hasattr(self.time, "dtype") and np.issubdtype(self.time.dtype, np.datetime64)  # type: ignore[union-attr]
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
        if hasattr(out.time, "dtype") and out.time.dtype == np.int64:  # type: ignore[union-attr]
            out.time.dtype = NP_DATETIME_FORM  # type: ignore[misc, union-attr]
        return out

    def combine(self, kfrecord2: KfRecord, /, *, inplace: bool = False) -> KfRecord:
        r"""Combines two KfRecord structures together."""
        # allow an empty structure to be passed through
        if self.time is None:
            if inplace:
                for (key, value) in vars(kfrecord2).items():
                    setattr(self, key, value)
            return kfrecord2  # TODO: make a copy?
        # concatenate fields
        if inplace:
            kfrecord = self
        else:
            kfrecord = copy.deepcopy(self)
        if kfrecord2.time is None:
            return kfrecord
        assert kfrecord.time is not None
        assert kfrecord2.time is not None
        kfrecord.time = np.hstack((self.time, kfrecord2.time))
        for field in frozenset({"P", "stm", "H", "Pz", "K", "z"}):
            if (x := getattr(self, field)) is not None and (y := getattr(kfrecord2, field)) is not None:
                setattr(kfrecord, field, np.concatenate((x, y), axis=x.ndim - 1))
        return kfrecord

    @overload
    def chop(
        self, ti: _Time = ..., tf: _Time = ..., *, include_last: bool = ..., inplace: bool = ..., return_ends: Literal[True]
    ) -> Tuple[KfRecord, KfRecord, KfRecord]:
        ...

    @overload
    def chop(
        self,
        ti: _Time = ...,
        tf: _Time = ...,
        *,
        include_last: bool = ...,
        inplace: bool = ...,
        return_ends: Literal[False] = ...,
    ) -> KfRecord:
        ...

    def chop(
        self, ti: _Time = None, tf: _Time = None, *, include_last: bool = True, inplace: bool = False, return_ends: bool = False
    ) -> Union[KfRecord, Tuple[KfRecord, KfRecord, KfRecord]]:
        r"""Chops the KfRecord structure to the given time bounds."""
        exclude: FrozenSet[str] = frozenset({})
        out = _chop_wrapper(
            self, exclude=exclude, ti=ti, tf=tf, include_last=include_last, inplace=inplace, return_ends=return_ends
        )  # type: ignore[call-overload]
        return out  # type: ignore[no-any-return]


#%% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_classes", exit=False)
    doctest.testmod(verbose=False)
