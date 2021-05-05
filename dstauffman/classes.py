r"""
Contains the high level classes used to subclass other classes.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Added mutable integer Counter class in January 2016.
#.  Updated by David C. Stauffer in June 2020 to make MetaClass methods public for direct use if desired.
"""

#%% Imports
from __future__ import annotations
import copy
import doctest
from pathlib import Path
import pickle
import sys
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Literal, NoReturn, Optional, \
    overload, Set, Tuple, Type, TypeVar, Union
import unittest
import warnings

from dstauffman.constants import HAVE_H5PY, HAVE_NUMPY
from dstauffman.paths import is_dunder
from dstauffman.utils import find_in_range

if HAVE_H5PY:
    import h5py
if HAVE_NUMPY:
    from numpy import all as np_all, ndarray, inf, printoptions
else:
    from dstauffman.numba import np_all  # type: ignore[no-redef]
    from math import inf
    from array import array as ndarray  # type: ignore[misc]

#%% Constants
_T = TypeVar('_T')
_C = TypeVar('_C', int, 'Counter')

#%% Functions - _frozen
def _frozen(set: Callable) -> Callable:
    r"""
    Support function for Frozen class.

    Raise an error when trying to set an undeclared name, or when calling
    from a method other than Frozen.__init__ or the __init__ method of
    a class derived from Frozen.

    """
    # define a custom set_attr function (instead of default setattr)
    def set_attr(self, name, value):
        if hasattr(self, name):
            # If attribute already exists, simply set it
            set(self, name, value)
            return
        if sys._getframe(1).f_code.co_name == '__init__':
            # Allow __setattr__ calls in __init__ calls of proper object types
            for key, val in sys._getframe(1).f_locals.items(): # pragma: no branch
                if key == 'self' and isinstance(val, self.__class__): # pragma: no branch
                    set(self, name, value)
                    return
        raise AttributeError('You cannot add attribute of {} to {} in {}.'.format(\
            name, self, sys._getframe(1).f_code.co_name))
    # return the custom defined function
    return set_attr

#%% Methods - save_hdf5
def save_hdf5(self, filename: Path=None) -> None:
    r"""
    Save the object to disk as an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to save

    """
    # exit if no filename is given
    if filename is None:
        return
    # Save data
    with h5py.File(filename, 'w') as file:
        grp = file.create_group('self')
        temp = vars(self) if not isinstance(self, dict) else self
        for key in temp:
            if is_dunder(key):
                continue
            value = temp[key]
            if value is not None:
                grp.create_dataset(key, data=value)

#%% Methods - load_hdf5
@overload
def load_hdf5(cls: Type[_T], filename: Optional[Path]) -> _T: ...
@overload
def load_hdf5(cls: Union[Dict[str, None], List[str], Set[str], Tuple[str, ...]], filename: Optional[Path]) -> Type[Any]: ...
@overload
def load_hdf5(cls: Literal[None], filename: Optional[Path]) -> Type[Any]: ...

def load_hdf5(cls: Union[None, Type[_T], Dict[str, None], List[str], Set[str], Tuple[str, ...]], \
        filename: Path=None) -> Union[_T, Type[Any]]:
    r"""
    Load the object from disk.

    Parameters
    ----------
    filename : str
        Name of the file to load

    """
    if filename is None:
        raise ValueError('No file specified to load.')
    # Load data
    out: Union[_T, Type[Any]]
    if cls is None:
        out = type('Temp', (object, ), {})
        limit_fields = False
    elif isinstance(cls, dict):
        out = type('Temp', (object, ), cls)
        limit_fields = True
    elif isinstance(cls, (list, set, tuple)):
        out = type('Temp', (object, ), {k: None for k in cls})
        limit_fields = True
    else:
        out = cls()
        limit_fields = False  # TODO: set to True or have option to raise or not raise error?
    with h5py.File(filename, 'r') as file:
        for key in file:
            grp = file[key]
            for field in grp:
                if limit_fields and not hasattr(out, field):
                    continue
                    #raise AttributeError(f"type object '{out.__name__}' has not attribute '{field}'")
                # Note grp[field].value is now grp[field][()] because of updated HDF5 API
                setattr(out, field, grp[field][()])
    return out

#%% Methods - save_pickle
def save_pickle(self, filename: Path = None) -> None:
    r"""
    Save a class instances to a pickle file.

    Parameters
    ----------
    results : list
        List of the objects to save
    filename : str
        Name of the file to load

    """
    # exit if no filename is given
    if filename is None:
        return
    with open(filename, 'wb') as file:
        pickle.dump(self, file)

#%% Methods - load_pickle
def load_pickle(cls: Type[_T], filename: Path = None) -> _T:
    r"""
    Load a class instance from a pickle file.

    Parameters
    ----------
    filename : str
        Name of the file to load

    Returns
    -------
    results : list
        List of the objects found within the file

    """
    if filename is None:
        raise ValueError('No file specified to load.')
    with open(filename, 'rb') as file:
        out: _T = pickle.load(file)
    return out

#%% Methods - save_method
def save_method(self, filename: Path = None, use_hdf5: bool = True) -> None:
    r"""
    Save the object to disk.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of the file to save
    use_hdf5 : bool, optional, defaults to False
        Write as *.hdf5 instead of *.pkl

    """
    # exit if no filename is given
    if filename is None:
        return
    if not use_hdf5:
        # Version 1 (Pickle):
        save_pickle(self, filename.with_suffix('.pkl'))
    else:
        # Version 2 (HDF5):
        save_hdf5(self, filename)

#%% Methods - load_method
def load_method(cls: Type[_T], filename: Path = None, use_hdf5: bool = True) -> _T:
    r"""
    Load the object from disk.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of the file to load
    use_hdf5 : bool, optional, defaults to False
        Write as *.hdf5 instead of *.pkl

    """
    if filename is None:
        raise ValueError('No file specified to load.')
    if not use_hdf5:
        # Version 1 (Pickle):
        out: _T = load_pickle(cls, filename.with_suffix('.pkl'))
    else:
        # Version 2 (HDF5):
        out = load_hdf5(cls, filename)
    return out

#%% pprint_dict
def pprint_dict(dct: Dict[Any, Any], *, name: str='', indent: int = 1, align: bool = True, \
        disp: bool = True, offset: int = 0, max_elements=None) -> str:
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
    >>> dct = {'a': 1, 'bb': 2, 'ccc': 3}
    >>> name = 'Demonstration'
    >>> text = pprint_dict(dct, name=name)
    Demonstration
     a   = 1
     bb  = 2
     ccc = 3

    """
    # print the name of the class/dictionary
    lines: List[str] = []
    if name:
        lines.append(' ' * offset + name)
    # build indentation padding
    this_indent = ' ' * (indent + offset)
    # find the length of the longest field name
    pad_len = max(len(x) for x in dct)
    # loop through fields
    for (this_key, this_value) in dct.items():
        if hasattr(this_value, 'pprint'):
            this_name = f'{this_key} (class {this_value.__class__.__name__})'
            try:
                this_line = this_value.pprint(name=this_name, indent=indent, align=align, \
                    disp=False, return_text=True, offset=offset+indent, max_elements=max_elements)
            except:
                # TODO: do I need this check or just let it fail?
                warnings.warn('pprint recursive call failed, reverting to default.')
                this_pad = ' ' * (pad_len - len(this_key)) if align else ''
                this_line = f'{this_indent}{this_key}{this_pad} = {this_value}'
        else:
            this_pad = ' ' * (pad_len - len(this_key)) if align else ''
            if max_elements is None or not HAVE_NUMPY:
                this_line = f'{this_indent}{this_key}{this_pad} = {this_value}'
            else:
                if max_elements == 0:
                    if isinstance(this_value, ndarray):
                        this_line = f'{this_indent}{this_key}{this_pad} = <ndarray {this_value.dtype} {this_value.shape}>'
                    else:
                        this_line = f'{this_indent}{this_key}{this_pad} = {type(this_value)}'
                else:
                    with printoptions(threshold=max_elements):
                        this_line = f'{this_indent}{this_key}{this_pad} = {this_value}'
        lines.append(this_line)
    text = '\n'.join(lines)
    if disp:
        print(text)
    return text

#%% Functions - chop_time
def chop_time(self: Any, time_field: str, exclude: Set[str] = None, ti: float = -inf, tf: float = inf) -> None:
    r"""
    Chops the class to only include values within the given time span.

    Parameters
    ----------
    self : class Any
        Instance of the class that this method is operating on
    time_field : str
        The name of the time field to use for reference
    ti : float, optional
        Time to start from, inclusive
    tf : float, optional
        Time to end at, inclusize

    Notes
    -----
    #.  Written by David C. Stauffer in October 2020

    Examples
    --------
    >>> from dstauffman import chop_time
    >>> import numpy as np
    >>> self = type('Temp', (object, ), {'time': np.array([1, 3, 4, 8]), 'data': np.array([7, 8, 9, 1])})
    >>> time_field = 'time'
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
    ix = find_in_range(getattr(self, time_field), min_=ti, max_=tf, inclusive=True)
    # exit early if no data is getting dropped
    if np_all(ix):
        return
    # drop data
    for key in vars(self):
        if key.startswith('_'):
            continue
        if exclude is not None and key in exclude:
            continue
        if (old := getattr(self, key)) is not None:
            setattr(self, key, old[..., ix])

#%% Functions - subsample_class
def subsample_class(self, skip: int = 30, start: int = 0, skip_fields: Union[FrozenSet[str], Set[str]] = None) -> None:
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
    >>> self = type('Temp', (object, ), {'time': np.array([1, 3, 4, 8]), \
    ...     'data': np.array([7, 8, 9, 1]), 'name': 'name'})
    >>> skip = 2
    >>> start = 1
    >>> skip_fields = {'name', }
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
        if key.startswith('_'):
            continue
        if skip_fields is not None and key in skip_fields:
            continue
        if (old := getattr(self, key)) is not None:
            setattr(self, key, old[..., start::skip])

#%% Classes - Frozen
class Frozen(object):
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
    def pprint(self, return_text: Literal[True], **kwargs) -> str: ...
    @overload
    def pprint(self, return_text: Literal[False], **kwargs) -> None: ...
    @overload
    def pprint(self, **kwargs) -> Optional[str]: ...

    def pprint(self, return_text: bool = False, **kwargs) -> Optional[str]:
        r"""Displays a pretty print version of the class."""
        name = kwargs.pop('name') if 'name' in kwargs else self.__class__.__name__
        text = pprint_dict(self.__dict__, name=name, **kwargs)
        return text if return_text else None

#%% MetaClasses - SaveAndLoad
class SaveAndLoad(type):
    r"""Metaclass to add 'save' and 'load' methods to the given class."""
    def __init__(cls, name, bases, dct):
        r"""Add the 'save' and 'load' classes if they are not already present."""
        if not hasattr(cls, 'save'):
            setattr(cls, 'save', save_method)
        if not hasattr(cls, 'load'):
            setattr(cls, 'load', classmethod(load_method))
        super().__init__(name, bases, dct)

#%% MetaClasses - SaveAndLoadPickle
class SaveAndLoadPickle(type):
    r"""Metaclass to add 'save' and 'load' methods to the given class."""
    def __init__(cls, name, bases, dct):
        r"""Add the 'save' and 'load' classes if they are not already present."""
        if not hasattr(cls, 'save'):
            setattr(cls, 'save', save_pickle)
        if not hasattr(cls, 'load'):
            setattr(cls, 'load', classmethod(load_pickle))
        super().__init__(name, bases, dct)

#%% Classes - Counter
class Counter(Frozen):
    r"""
    Mutable integer counter wrapper class.

    Has methods for comparisons, negations, adding and subtracting, hashing for sets, and sorting.

    Parameters
    ----------
    other : int
        Initial value

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------
    >>> from dstauffman import Counter
    >>> c = Counter(0)
    >>> c += 1
    >>> print(c)
    1

    """
    def __init__(self, other: Any = 0):
        self._val = int(other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Counter):
            return self._val == other._val
        return self._val == other  # type: ignore[no-any-return]

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Counter):
            return self._val < other._val
        return self._val < other  # type: ignore[no-any-return]

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Counter):
            return self._val <= other._val
        return self._val <= other  # type: ignore[no-any-return]

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Counter):
            return self._val > other._val
        return self._val > other  # type: ignore[no-any-return]

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, Counter):
            return self._val >= other._val
        return self._val >= other  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self._val)

    def __index__(self) -> int:
        return self._val

    def __pos__(self) -> Counter:
        return Counter(self._val)

    def __neg__(self) -> Counter:
        return Counter(-self._val)

    def __abs__(self) -> Counter:
        return Counter(abs(self._val))

    @overload
    def __add__(self, other: int) -> int: ...
    @overload
    def __add__(self, other: Counter) -> Counter: ...

    def __add__(self, other: _C) -> _C:
        if isinstance(other, Counter):
            return Counter(self._val + other._val)
        if isinstance(other, int):
            return self._val + other
        return NotImplemented

    def __iadd__(self, other: _C) -> Counter:  # type: ignore[misc]
        if isinstance(other, Counter):
            self._val += other._val
        elif isinstance(other, int):
            self._val += other
        else:
            return NotImplemented
        return self

    def __radd__(self, other: _C) -> _C:
        return self.__add__(other)

    @overload
    def __sub__(self, other: int) -> int: ...
    @overload
    def __sub__(self, other: Counter) -> Counter: ...

    def __sub__(self, other: _C) -> _C:
        if isinstance(other, Counter):
            return Counter(self._val - other._val)
        if isinstance(other, int):
            return self._val - other
        return NotImplemented

    def __isub__(self, other: _C) -> Counter:  # type: ignore[misc]
        if isinstance(other, Counter):
            self._val -= other._val
        elif isinstance(other, int):
            self._val -= other
        else:
            return NotImplemented
        return self

    def __rsub__(self, other: _C) -> _C:
        return -self.__sub__(other)

    def __truediv__(self, other: Union[int, float]) -> float:
        if isinstance(other, (float, int)):
            return self._val / other
        return NotImplemented  # type: ignore[unreachable]

    def __floordiv__(self, other: _C) -> _C:
        if isinstance(other, Counter):
            return Counter(self._val // other._val)
        if isinstance(other, int):
            return self._val // other
        return NotImplemented

    def __mod__(self, other: _C) -> _C:
        if isinstance(other, Counter):
            return Counter(self._val % other._val)
        if isinstance(other, int):
            return self._val % other
        return NotImplemented

    def __str__(self) -> str:
        return str(self._val)

    def __repr__(self) -> str:
        return 'Counter({})'.format(self._val)

#%% FixedDict
class FixedDict(dict):
    r"""
    A dictionary with immutable keys, but mutable values.

    Notes
    -----
    #.  Taken from http://stackoverflow.com/questions/14816341/define-a-python-dictionary-
        with-immutable-keys-but-mutable-values by bereal.
    #.  Modified by David C. Stauffer in January 2017 to include alternative initializations
        and freeze methods.
    #.  Updated by David C. Stauffer in November 2017 to include __new__ method. Otherwise instances
        could be made that wouldn't have a self._frozen attribute.  Also added an empty
        __getnewargs__ method to ensure that pickling calls the __new__ method.

    Examples
    --------
    >>> from dstauffman import FixedDict
    >>> fixed = FixedDict({'key1': 1, 'key2': None})
    >>> assert 'key1' in fixed

    >>> fixed.freeze()
    >>> fixed['new_key'] = 5 # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    KeyError: 'new_key'

    """
    def __new__(cls, *args, **kwargs) -> FixedDict:
        r"""Creats a new instance of the class."""
        instance = super().__new__(cls, *args, **kwargs)
        instance._frozen = False
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen: bool = False

    def __getitem__(self, k: Any) -> Any:
        return super().__getitem__(k)

    def __setitem__(self, k: Any, v: Any) -> Any:
        if self._frozen:
            if k not in self:
                raise KeyError(k)
        return super().__setitem__(k, v)

    def __delitem__(self, k: Any) -> None:
        raise NotImplementedError

    def __contains__(self, k: Any) -> Any:
        return super().__contains__(k)

    def __copy__(self) -> FixedDict:
        new = type(self)(self.items())
        new._frozen = self._frozen
        return new

    def __deepcopy__(self, memo: Any) -> FixedDict:
        new = type(self)((k, copy.deepcopy(v, memo)) for (k,v) in self.items())
        new._frozen = self._frozen
        return new

    def __getnewargs__(self) -> Tuple:
        # Call __new__ (and thus __init__) on unpickling.
        return ()

    def get(self, k: Any, default: Any = None) -> Any:
        r""".get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."""
        return super().get(k, default)

    def setdefault(self, k: Any, default: Any = None) -> Any:
        r"""D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D."""
        if self._frozen:
            if k not in self:
                raise KeyError(k)
        return super().setdefault(k, default)

    def pop(self, k: Any) -> NoReturn:  # type: ignore[override]
        r"""D.pop(k[,d]) -> v, is not valid on a fixeddict, as it removes the key."""
        raise NotImplementedError

    def update(self, mapping=(), **kwargs) -> None:  # type: ignore[override]
        r"""
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
        """
        # check if valid keys otherwise, raise error
        if self._frozen:
            for k in mapping:
                if k not in self:
                    raise KeyError(k)
            for k in kwargs:
                if k not in self:
                    raise KeyError(k)
        # otherwise keys are good, pass on to super
        super().update(mapping, **kwargs)

    @classmethod
    def fromkeys(cls, keys: Iterable) -> Any:  # type: ignore[override]
        """Returns a new dict with keys from iterable and values equal to value."""
        return super().fromkeys(k for k in keys)

    def freeze(self) -> None:
        """Freeze the internal dictionary, such that no more keys may be added."""
        self._frozen = True

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_classes', exit=False)
    doctest.testmod(verbose=False)
