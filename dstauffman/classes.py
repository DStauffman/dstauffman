r"""
Contains the high level classes used to subclass other classes.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Added mutable integer Counter class in January 2016.
#.  Updated by David C. Stauffer in June 2020 to make MetaClass methods public for direct use if desired.

"""

# %% Imports
from __future__ import annotations

import doctest
from pathlib import Path
import sys
from typing import Any, Callable, Literal, overload, Type, TYPE_CHECKING, TypedDict, TypeVar
try:
    from typing import NotRequired, Unpack
except ImportError:
    from typing_extensions import NotRequired, Unpack  # for Python v3.10
import unittest
import warnings

from slog import IntEnumPlus, is_dunder

from dstauffman.constants import HAVE_H5PY, HAVE_NUMPY, HAVE_PANDAS, NP_DATETIME_FORM
from dstauffman.time import is_datetime
from dstauffman.utils import find_in_range

if HAVE_H5PY:
    import h5py
if HAVE_NUMPY:
    from numpy import all as np_all, inf, int64, ndarray, printoptions
else:
    from array import array as ndarray  # type: ignore[assignment]
    from math import inf

    from nubs import np_all  # type: ignore[no-redef]
if HAVE_PANDAS:
    from pandas import DataFrame

# %% Constants
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    _B = NDArray[np.bool_]
    _T = TypeVar("_T")
    _SingleNum = int | float | np.datetime64
    _Sets = set[str] | frozenset[str]
    _Time = float | np.datetime64

    class _PPrintKwArgs(TypedDict):
        name: NotRequired[str]
        indent: NotRequired[int]
        align: NotRequired[bool]
        disp: NotRequired[bool]
        offset: NotRequired[int]
        max_elements: NotRequired[int | None]


# %% Functions - _frozen
def _frozen(set_: Callable) -> Callable:
    r"""
    Support function for Frozen class.

    Raise an error when trying to set an undeclared name, or when calling
    from a method other than Frozen.__init__ or the __init__ method of
    a class derived from Frozen.

    """

    def set_attr(self: Any, name: str, value: Any) -> None:
        r"""Define a custom set_attr function (instead of default setattr)."""
        if hasattr(self, name):
            # If attribute already exists, simply set it
            set_(self, name, value)
            return
        if sys._getframe(1).f_code.co_name == "__init__":  # pylint: disable=protected-access
            # Allow __setattr__ calls in __init__ calls of proper object types
            for key, val in sys._getframe(1).f_locals.items():  # pragma: no branch  # pylint: disable=protected-access
                if key == "self" and isinstance(val, self.__class__):  # pragma: no branch
                    set_(self, name, value)
                    return
        # fmt: off
        raise AttributeError(f"You cannot add attribute of {name} to {self} in {sys._getframe(1).f_code.co_name}.")  # pylint: disable=protected-access
        # fmt: on

    # return the custom defined function
    return set_attr


# %% Methods - save_hdf5
def save_hdf5(  # noqa: C901
    self: Any,
    filename: Path | None = None,
    *,
    file: h5py.File | None = None,
    base_group: str = "self",
    meta: dict[str, Any] | None = None,
    exclusions: _Sets | None = None,
    **kwargs: Any,
) -> None:
    r"""
    Save the object to disk as an HDF5 file.

    Parameters
    ----------
    self : class instance or dict
        Instance that this method is added to, otherwise, use a dictionary
    filename : str
        Name of the file to save
    meta : dict, optional
        Meta information to write to the file attributes
    exclusions : set, optional
        Fieldnames to not write out to disk
    kwargs : dict, optional
        Extra arguments to pass to the HDF5 dataset creation

    Notes
    -----
    #.  Written by David C. Stauffer in May 2015.
    #.  Updated by David C. Stauffer to include meta information and expose compression options in
        October 2021.

    Examples
    --------
    >>> from dstauffman import save_hdf5, get_tests_dir
    >>> data = {"time": [1, 2, 3, 4, 5], "data": [0, 0.5, 1.0, 1.5, 2], "ver": 1.0}
    >>> filename = get_tests_dir() / "test_file.hdf5"
    >>> meta = {"num_pts": 5}
    >>> exclusions = {"ver", }
    >>> save_hdf5(data, filename, meta=meta, exclusions=exclusions)  # doctest: +SKIP

    """

    def _save_data(
        self: Any,
        *,
        file: h5py.File,
        base_group: str,
        meta: dict[str, Any] | None,
        exclusions: _Sets | None,
        **kwargs: Any,
    ) -> None:
        # alias keyword options
        compression = kwargs.pop("compression", "gzip")
        shuffle = kwargs.pop("shuffle", True)
        # create group
        grp = file.create_group(base_group)
        # write meta data
        if meta is not None:
            for key, value in meta.items():
                grp.attrs[key] = value
        # figure out how to loop over self
        types = (dict, DataFrame) if HAVE_PANDAS else (dict,)
        temp = vars(self) if not isinstance(self, types) else self
        # loop and write data by name and type criteria
        for key in temp:
            if is_dunder(key):
                continue
            if exclusions is not None and key in exclusions:
                continue
            value = temp[key]
            if value is not None:
                if isinstance(value, dict):
                    _save_data(value, file=file, base_group=f"{base_group}/{key}", meta=None, exclusions=exclusions, **kwargs)
                    continue
                if hasattr(value, "__metaclass__") and value.__metaclass__ is Frozen.__metaclass__:
                    # TODO: is their a more generic way other than just checking the metaclass?
                    _save_data(value, file=file, base_group=f"{base_group}/{key}", meta=None, exclusions=exclusions, **kwargs)
                    continue
                if isinstance(value, (str, bytes)):
                    force_no_compression = True
                else:
                    try:
                        iter(value)
                    except TypeError:
                        force_no_compression = True
                    else:
                        force_no_compression = False
                if force_no_compression:
                    grp.create_dataset(key, data=value, compression=None, shuffle=False, **kwargs)
                else:
                    try:
                        grp.create_dataset(key, data=value, compression=compression, shuffle=shuffle, **kwargs)
                    except TypeError as exception:
                        raise TypeError(f'Problem converting field: "{key}"') from exception

    # check for nominal process of creating new file
    if file is None:
        # exit if no filename
        if filename is None:
            return
        # Save data
        with h5py.File(filename, "w") as new_file:
            _save_data(self, file=new_file, base_group=base_group, meta=meta, exclusions=exclusions, **kwargs)
        return
    # if file is given, then continue writing to the already open file
    _save_data(self, file=file, base_group=base_group, meta=meta, exclusions=exclusions, **kwargs)


# %% Methods - load_hdf5
@overload
def load_hdf5(cls: Type[_T], filename: Path | None, return_meta: Literal[False] = ...) -> _T: ...
@overload
def load_hdf5(
    cls: dict[str, None] | list[str] | set[str] | tuple[str, ...],
    filename: Path | None,
    return_meta: Literal[False] = ...,
) -> Type[Any]: ...
@overload
def load_hdf5(cls: Literal[None], filename: Path | None, return_meta: Literal[False] = ...) -> Type[Any]: ...
@overload
def load_hdf5(cls: Type[_T], filename: Path | None, return_meta: Literal[True]) -> tuple[_T, dict[str, Any]]: ...
@overload
def load_hdf5(
    cls: dict[str, None] | list[str] | set[str] | tuple[str, ...], filename: Path | None, return_meta: Literal[True]
) -> tuple[Type[Any], dict[str, Any]]: ...
@overload
def load_hdf5(cls: Literal[None], filename: Path | None, return_meta: Literal[True]) -> tuple[Type[Any], dict[str, Any]]: ...
def load_hdf5(  # type: ignore[misc]  # noqa: C901
    cls: Type[_T] | dict[str, None] | list[str] | set[str] | tuple[str, ...] | None,
    filename: Path | None = None,
    return_meta: bool = False,
) -> _T | Type[Any] | tuple[_T, dict[str, Any]] | tuple[Type[Any] | dict[str, Any]]:
    r"""
    Load the object from disk.

    Parameters
    ----------
    filename : str
        Name of the file to load

    Notes
    -----
    #.  Written by David C. Stauffer in May 2015.
    #.  Updated by David C. Stauffer to include meta information in October 2021.

    Examples
    --------
    >>> from dstauffman import load_hdf5, get_tests_dir
    >>> filename = get_tests_dir() / "test_file.hdf5"
    >>> (data, meta) = load_hdf5(None, filename, return_meta=True)  # doctest: +SKIP

    """

    def _load_dataset(out: Any, group: h5py.Group, prefix: str, limit_fields: bool) -> h5py.Dataset:
        for field in group:
            if limit_fields and not hasattr(out, field):
                continue
                # raise AttributeError(f'type object "{out.__name__}" has not attribute "{field}"')
            if isinstance(group[field], h5py.Group):
                if hasattr(out, field):
                    _load_dataset(getattr(out, field), group[field], prefix=f"{prefix}/field", limit_fields=limit_fields)
                else:
                    if limit_fields:
                        continue  # or raise AttributeError?
                    temp = type("Temp", (object,), {})  # TODO: be able to specify this?
                    _load_dataset(temp, group[field], prefix=f"{prefix}/field", limit_fields=limit_fields)
                    if isinstance(out, dict):
                        out[field] = temp
                    else:
                        setattr(out, field, temp)
                continue
            # Note grp[field].value is now grp[field][()] because of updated HDF5 API
            setattr(out, field, group[field][()])

    if filename is None:
        raise ValueError("No file specified to load.")
    if return_meta:
        meta: dict[str, Any] = {}
    # Load data
    out: _T | Type[Any]
    if cls is None:
        out = type("Temp", (object,), {})
        limit_fields = False
    elif isinstance(cls, dict):
        out = type("Temp", (object,), cls)
        limit_fields = True
    elif isinstance(cls, (list, set, tuple)):
        out = type("Temp", (object,), {k: None for k in cls})  # noqa: C420
        limit_fields = True
    else:
        out = cls()
        limit_fields = False  # TODO: set to True or have option to raise or not raise error?
    with h5py.File(filename, "r") as file:
        for key in file:
            group = file[key]
            if return_meta:
                for key2, value in group.attrs.items():
                    meta[key2] = value
            _load_dataset(out, group=group, prefix=f"/{key}", limit_fields=limit_fields)
    if return_meta:
        return (out, meta)  # type: ignore[return-value]
    return out


# %% Methods - save_method
def save_method(
    self: Any,
    filename: Path | None = None,
    *,
    meta: dict[str, Any] | None = None,
    exclusions: _Sets | None = None,
    **kwargs: Any,
) -> None:
    r"""
    Save the object to disk.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of the file to save
    meta : dict, optional
        Meta information to save to the file, keys are strings, values can be anything saveable to HDF5 datasets
    exclusions : set, optional
        Names of fields to exclude when saving to file

    """
    # exit if no filename is given
    if filename is None:
        return
    if hasattr(self, "_save_convert_hdf5") and callable(self._save_convert_hdf5):  # pylint: disable=protected-access
        restore_kwargs = self._save_convert_hdf5()  # pylint: disable=protected-access
    else:
        restore_kwargs = {}
    try:
        additional_exclusions = self._exclude_fields()  # pylint: disable=protected-access
    except AttributeError:
        pass
    else:
        if exclusions is None:
            exclusions = set(additional_exclusions)
        else:
            exclusions |= set(additional_exclusions)
    save_hdf5(self, filename, meta=meta, exclusions=exclusions, **kwargs)
    if hasattr(self, "_save_restore_hdf5") and callable(self._save_restore_hdf5):  # pylint: disable=protected-access
        self._save_restore_hdf5(**restore_kwargs)  # pylint: disable=protected-access


# %% Methods - load_method
@overload
def load_method(cls: Type[_T], filename: Path | None, return_meta: Literal[False] = ..., **kwargs: Any) -> _T: ...
@overload
def load_method(
    cls: Type[_T], filename: Path | None, return_meta: Literal[True], **kwargs: Any
) -> tuple[_T, dict[str, Any]]: ...
def load_method(
    cls: Type[_T], filename: Path | None = None, return_meta: bool = False, **kwargs: Any
) -> _T | tuple[_T, dict[str, Any]]:
    r"""
    Load the object from disk.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of the file to load
    return_meta : bool, optional, defaults to False
        Return any meta information found in the file

    """
    if filename is None:
        raise ValueError("No file specified to load.")
    out = load_hdf5(cls, filename, return_meta=return_meta)  # type: ignore[call-overload]
    if hasattr(out, "_save_restore_hdf5") and callable(out._save_restore_hdf5):  # pylint: disable=protected-access
        out._save_restore_hdf5(**kwargs)  # pylint: disable=protected-access
    return out  # type: ignore[no-any-return]


# %% save_convert_hdf5
def save_convert_hdf5(self: Any, **kwargs: Any) -> dict[str, bool]:
    r"""Supporting function for saving to HDF5."""
    if "datetime_fields" in kwargs:
        datetime_fields = kwargs["datetime_fields"]
    else:
        try:
            datetime_fields = self._datetime_fields()  # pylint: disable=protected-access
        except AttributeError:
            datetime_fields = ()
    convert_dates = all(map(lambda key: is_datetime(getattr(self, key)), datetime_fields))  # noqa: C417
    if convert_dates:
        assert HAVE_NUMPY, "Must have numpy to convert the dates."
        for key in datetime_fields:
            if (value := getattr(self, key)) is not None:
                setattr(self, key, value.astype(int64))
    return {"convert_dates": convert_dates}


# %% save_restore_hdf5
def save_restore_hdf5(self: Any, *, convert_dates: bool = False, **kwargs: Any) -> None:  # noqa: C901
    r"""Supporting function for loading from HDF5."""
    if convert_dates:
        assert HAVE_NUMPY, "Must have numpy to convert dates."
        if "datetime_fields" in kwargs:
            datetime_fields = kwargs["datetime_fields"]
        else:
            try:
                datetime_fields = self._datetime_fields()  # pylint: disable=protected-access
            except AttributeError:
                datetime_fields = ()
        for key in datetime_fields:
            if (value := getattr(self, key)) is not None:
                setattr(self, key, value.astype(NP_DATETIME_FORM))
    if "string_fields" in kwargs:
        string_fields = kwargs["string_fields"]
    else:
        try:
            string_fields = self._string_fields()  # pylint: disable=protected-access
        except AttributeError:
            string_fields = ()
    for key in string_fields:
        if not isinstance(getattr(self, key), str):
            setattr(self, key, getattr(self, key).decode("utf-8"))


# %% pprint
def pprint(self: Any, return_text: bool = False, **kwargs: Any) -> str | None:
    r"""Displays a pretty print version of the class."""
    name = kwargs.pop("name") if "name" in kwargs else self.__class__.__name__
    text = pprint_dict(self.__dict__, name=name, **kwargs)
    return text if return_text else None


# %% pprint_dict
def pprint_dict(  # noqa: C901
    dct: dict[Any, Any],
    *,
    name: str = "",
    indent: int = 1,
    align: bool = True,
    disp: bool = True,
    offset: int = 0,
    max_elements: int | None = None,
) -> str:
    r"""
    Print all the fields and their values.

    Parameters
    ----------
    dct : dict
        Dictionary to print
    name : str, optional, default is empty string
        Name title to print first
    indent : int, optional, default is 1
        Number of characters to indent before all the fields
    align : bool, optional, default is True
        Whether to align all the equal signs
    disp : bool, optional, default is True
        Whether to display the text to the screen
    offset : int, optional, default is 0
        Additional offset for recursive calls
    max_elements : int, optional, default is None meaning don't change
        Maximum number of elements to show in array, if zero, then only show shape of array

    Notes
    -----
    #.  Written by David C. Stauffer in February 2017.
    #.  Updated by David C. Stauffer in June 2020 for better recursive support.

    Examples
    --------
    >>> from dstauffman import pprint_dict
    >>> dct = {"a": 1, "bb": 2, "ccc": 3}
    >>> name = "Demonstration"
    >>> text = pprint_dict(dct, name=name)
    Demonstration
     a   = 1
     bb  = 2
     ccc = 3

    """
    # print the name of the class/dictionary
    lines: list[str] = []
    if name:
        lines.append(" " * offset + name)
    # build indentation padding
    this_indent = " " * (indent + offset)
    # find the length of the longest field name
    pad_len = max(len(x) for x in dct)
    # loop through fields
    for this_key, this_value in dct.items():
        if hasattr(this_value, "pprint"):
            this_name = f"{this_key} (class {this_value.__class__.__name__})"
            try:
                this_line = this_value.pprint(
                    name=this_name,
                    indent=indent,
                    align=align,
                    disp=False,
                    return_text=True,
                    offset=offset + indent,
                    max_elements=max_elements,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # TODO: do I need this check or just let it fail?
                warnings.warn("pprint recursive call failed, reverting to default.")
                this_pad = " " * (pad_len - len(this_key)) if align else ""
                this_line = f"{this_indent}{this_key}{this_pad} = {this_value}"
        else:
            this_pad = " " * (pad_len - len(this_key)) if align else ""
            if max_elements is None or not HAVE_NUMPY:
                this_line = f"{this_indent}{this_key}{this_pad} = {this_value}"
            else:
                if max_elements == 0:
                    if isinstance(this_value, ndarray):
                        this_line = f"{this_indent}{this_key}{this_pad} = <ndarray {this_value.dtype} {this_value.shape}>"
                    elif isinstance(this_value, IntEnumPlus):  # TODO: may not be necessary on newer versions of Python?
                        this_line = f"{this_indent}{this_key}{this_pad} = {this_value.__class__.__name__}"
                    else:
                        this_line = f"{this_indent}{this_key}{this_pad} = {type(this_value)}"
                else:
                    with printoptions(threshold=max_elements):
                        this_line = f"{this_indent}{this_key}{this_pad} = {this_value}"
        lines.append(this_line)
    text = "\n".join(lines)
    if disp:
        print(text)
    return text


# %% Functions - chop_time
def chop_time(
    self: Any,
    time_field: str,
    *,
    exclude: _Sets | None = None,
    ti: _Time = -inf,
    tf: _Time = inf,
    inclusive: bool = False,
    mask: bool | _B | None = None,
    precision: _SingleNum = 0,
    left: bool = True,
    right: bool = True,
) -> None:
    r"""
    Chops the class to only include values within the given time span.

    Parameters
    ----------
    self : class Any
        Instance of the class that this method is operating on
    time_field : str
        The name of the time field to use for reference
    exclude : list[str]
        Names of any fields to exclude from chopping
    ti : float, optional
        Time to start from, inclusive
    tf : float, optional
        Time to end at, inclusize
    inclusive : bool, optional, default is False
        Whether to inclusively count both endpoints (overrules left and right)
    mask : (N,) ndarray of bool, optional
        A mask to preapply to the results
    precision : int or float, optional, default is zero
        A precision to apply to the comparisons
    left : bool, optional, default is True
        Whether to include the left endpoint in the range
    right : bool, optional, default is True
        Whether to include the right endpoint in the range

    Notes
    -----
    #.  Written by David C. Stauffer in October 2020.
    #.  Updated by David C. Stauffer in June 2021 to better wrap the find_in_range function.

    Examples
    --------
    >>> from dstauffman import chop_time
    >>> import numpy as np
    >>> self = type("Temp", (object, ), {"time": np.array([1, 3, 4, 8]), "data": np.array([7, 8, 9, 1])})
    >>> time_field = "time"
    >>> ti = 2
    >>> tf = 5
    >>> print(self.time)
    [1 3 4 8]

    >>> print(self.data)
    [7 8 9 1]

    >>> chop_time(self, time_field=time_field, ti=ti, tf=tf)
    >>> print(self.time)
    [3 4]

    >>> print(self.data)
    [8 9]

    """
    # build the new index
    ix = find_in_range(
        getattr(self, time_field), min_=ti, max_=tf, inclusive=inclusive, mask=mask, precision=precision, left=left, right=right
    )
    # exit early if no data is getting dropped
    if np_all(ix):
        return
    # drop data
    for key in vars(self):
        if key.startswith("_"):
            continue
        if exclude is not None and key in exclude:
            continue
        if (old := getattr(self, key)) is not None:
            setattr(self, key, old[..., ix])


# %% Functions - subsample_class
def subsample_class(self: Any, skip: int = 30, start: int = 0, skip_fields: frozenset[str] | set[str] | None = None) -> None:
    r"""
    Subsamples the class instance to every `skip` data point.

    Parameters
    ----------
    self : class Any
        Instance of the class that this method is operating on
    skip : int, optional
        The number of points between data to ignore
    start : int, optional
        The point to start skipping from
    skip_fields : set[str], optional
        The name of any fields to ignore

    Notes
    -----
    #.  Written by David C. Stauffer in October 2020.

    Examples
    --------
    >>> from dstauffman import subsample_class
    >>> import numpy as np
    >>> self = type("Temp", (object, ), {"time": np.array([1, 3, 4, 8]), \
    ...     "data": np.array([7, 8, 9, 1]), "name": "name"})
    >>> skip = 2
    >>> start = 1
    >>> skip_fields = {"name", }
    >>> print(self.time)
    [1 3 4 8]

    >>> print(self.data)
    [7 8 9 1]

    >>> subsample_class(self, skip=skip, start=start, skip_fields=skip_fields)
    >>> print(self.time)
    [3 8]

    >>> print(self.data)
    [8 1]

    >>> print(self.name)
    name

    """
    for key in vars(self):
        if key.startswith("_"):
            continue
        if skip_fields is not None and key in skip_fields:
            continue
        if (old := getattr(self, key)) is not None:
            setattr(self, key, old[..., start::skip])


# %% Classes - Frozen
class Frozen:
    r"""
    Frozen class that doesn't allow new attributes.

    Subclasses of Frozen are frozen, i.e. it is impossibile to add new attributes to them or their
    instances.

    """

    # freeze the set attributes function based on the above `frozen` funcion
    __setattr__ = _frozen(object.__setattr__)

    class __metaclass__(type):
        __setattr__ = _frozen(type.__setattr__)

    @overload
    def pprint(self: Any, return_text: Literal[True], **kwargs: Unpack[_PPrintKwArgs]) -> str: ...
    @overload
    def pprint(self: Any, return_text: Literal[False], **kwargs: Unpack[_PPrintKwArgs]) -> None: ...
    @overload
    def pprint(self: Any, **kwargs: Unpack[_PPrintKwArgs]) -> str | None: ...
    def pprint(self: Any, return_text: bool = False, **kwargs: Unpack[_PPrintKwArgs]) -> str | None:
        r"""Displays a pretty print version of the class."""
        name = kwargs.pop("name") if "name" in kwargs else self.__class__.__name__
        text = pprint_dict(self.__dict__, name=name, **kwargs)  # type: ignore[misc]
        return text if return_text else None


# %% MetaClasses - SaveAndLoad
class SaveAndLoad(type):
    r"""Metaclass to add "save" and "load" methods to the given class."""

    def __init__(cls, name: Any, bases: Any, dct: Any):
        r"""Add the "save" and "load" classes if they are not already present."""
        if not hasattr(cls, "save"):
            setattr(cls, "save", save_method)
        if not hasattr(cls, "load"):
            setattr(cls, "load", classmethod(load_method))
        if not hasattr(cls, "_save_convert_hdf5"):
            setattr(cls, "_save_convert_hdf5", save_convert_hdf5)
        if not hasattr(cls, "_save_restore_hdf5"):
            setattr(cls, "_save_restore_hdf5", save_restore_hdf5)

        super().__init__(name, bases, dct)


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_classes", exit=False)
    doctest.testmod(verbose=False)
