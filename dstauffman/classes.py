# -*- coding: utf-8 -*-
r"""
Contains the high level classes used to subclass other classes.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Added mutable integer Counter class in January 2016.

"""

#%% Imports
import copy
import doctest
import pickle
import sys
import unittest
import warnings

import numpy as np

try:
    import h5py
except ImportError: # pragma: no cover
    warnings.warn('h5py was not imported, so some file save and load capabilities will be limited.')

#%% Set error state for module
np.seterr(invalid='ignore', divide='ignore')

#%% Set NumPy printing options
np.set_printoptions(threshold=1000) # TODO: make user configurable in constants or something?

#%% Functions - _frozen
def _frozen(set):
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
        elif sys._getframe(1).f_code.co_name == '__init__':
            # Allow __setattr__ calls in __init__ calls of proper object types
            for key, val in sys._getframe(1).f_locals.items(): # pragma: no branch
                if key == 'self' and isinstance(val, self.__class__): # pragma: no branch
                    set(self, name, value)
                    return
        raise AttributeError('You cannot add attribute of {} to {} in {}.'.format(\
            name, self, sys._getframe(1).f_code.co_name))
    # return the custom defined function
    return set_attr

#%% Methods - _save_method
def _save_method(self, filename='', use_hdf5=True):
    r"""
    Save the object to disk.

    Parameters
    ----------
    filename : str
        Name of the file to save
    use_hdf5 : bool, optional, defaults to False
        Write as *.hdf5 instead of *.pkl

    """
    # exit if no filename is given
    if not filename:
        return
    # potentially convert the filename
    if not use_hdf5:
        # Version 1 (Pickle):
        with open(filename.replace('hdf5', 'pkl'), 'wb') as file:
            pickle.dump(self, file)
    else:
        # Version 2 (HDF5):
        with h5py.File(filename, 'w') as file:
            grp = file.create_group('self')
            for key in vars(self):
                value = getattr(self, key)
                if value is not None:
                    grp.create_dataset(key, data=value)

#%% Methods - _load_method
@classmethod
def _load_method(cls, filename='', use_hdf5=True):
    r"""
    Load the object from disk.

    Parameters
    ----------
    filename : str
        Name of the file to load
    use_hdf5 : bool, optional, defaults to False
        Write as *.hdf5 instead of *.pkl

    """
    if not filename:
        raise ValueError('No file specified to load.')
    if not use_hdf5:
        # Version 1 (Pickle):
        with open(filename.replace('hdf5', 'pkl'), 'rb') as file:
            out = pickle.load(file)
    else:
        # Version 2 (HDF5):
        out = cls()
        with h5py.File(filename, 'r') as file:
            for key in file:
                grp = file[key]
                for field in grp:
                    setattr(out, field, grp[field].value)
    return out

#%% Methods - _save_pickle
def _save_pickle(self, filename):
    r"""
    Save a class instances to a pickle file.

    Parameters
    ----------
    results : list
        List of the objects to save
    filename : str
        Name of the file to load

    """
    with open(filename, 'wb') as file:
        pickle.dump(self, file)

#%% Methods - _load_pickle
@classmethod
def _load_pickle(cls, filename):
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
    with open(filename, 'rb') as file:
        out = pickle.load(file)
    return out

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

#%% MetaClasses - SaveAndLoad
class SaveAndLoad(type):
    r"""Metaclass to add 'save' and 'load' methods to the given class."""
    def __init__(cls, name, bases, dct):
        r"""Add the 'save' and 'load' classes if they are not already present."""
        if not hasattr(cls, 'save'):
            setattr(cls, 'save', _save_method)
        if not hasattr(cls, 'load'):
            setattr(cls, 'load', _load_method)
        super().__init__(name, bases, dct)

#%% MetaClasses - SaveAndLoadPickle
class SaveAndLoadPickle(type):
    r"""Metaclass to add 'save' and 'load' methods to the given class."""
    def __init__(cls, name, bases, dct):
        r"""Add the 'save' and 'load' classes if they are not already present."""
        if not hasattr(cls, 'save'):
            setattr(cls, 'save', _save_pickle)
        if not hasattr(cls, 'load'):
            setattr(cls, 'load', _load_pickle)
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
    def __init__(self, other=0):
        self._val = int(other)
    def __eq__(self, other):
        if type(other) == Counter:
            return self._val == other._val
        return self._val == other
    def __lt__(self, other):
        if type(other) == Counter:
            return self._val < other._val
        return self._val < other
    def __le__(self, other):
        if type(other) == Counter:
            return self._val <= other._val
        return self._val <= other
    def __gt__(self, other):
        if type(other) == Counter:
            return self._val > other._val
        return self._val > other
    def __ge__(self, other):
        if type(other) == Counter:
            return self._val >= other._val
        return self._val >= other
    def __hash__(self):
        return hash(self._val)
    def __index__(self):
        return self._val
    def __pos__(self):
        return Counter(self._val)
    def __neg__(self):
        return Counter(-self._val)
    def __abs__(self):
        return Counter(abs(self._val))
    def __add__(self, other):
        if type(other) == Counter:
            return Counter(self._val + other._val)
        elif type(other) == int:
            return self._val + other
        else:
            return NotImplemented
    def __iadd__(self, other):
        if type(other) == Counter:
            self._val += other._val
        elif type(other) == int:
            self._val += other
        else:
            return NotImplemented
        return self
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if type(other) == Counter:
            return Counter(self._val - other._val)
        elif type(other) == int:
            return self._val - other
        else:
            return NotImplemented
    def __isub__(self, other):
        if type(other) == Counter:
            self._val -= other._val
        elif type(other) == int:
            self._val -= other
        else:
            return NotImplemented
        return self
    def __rsub__(self, other):
        return -self.__sub__(other)
    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            return self._val / other
        else:
            return NotImplemented
    def __floordiv__(self, other):
        if type(other) == Counter:
            return Counter(self._val // other._val)
        elif type(other) == int:
            return self._val // other
        else:
            return NotImplemented
    def __mod__(self, other):
        if type(other) == Counter:
            return Counter(self._val % other._val)
        elif type(other) == int:
            return self._val % other
        else:
            return NotImplemented
    def __str__(self):
        return str(self._val)
    def __repr__(self):
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
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance._frozen = False
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = False

    def __getitem__(self, k):
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if self._frozen:
            if k not in self:
                raise KeyError(k)
        return super().__setitem__(k, v)

    def __delitem__(self, k):
        raise NotImplementedError

    def __contains__(self, k):
        return super().__contains__(k)

    def __copy__(self):
        new = type(self)(self.items())
        new._frozen = self._frozen
        return new

    def __deepcopy__(self, memo):
        new = type(self)((k, copy.deepcopy(v, memo)) for (k,v) in self.items())
        new._frozen = self._frozen
        return new

    def __getnewargs__(self):
        # Call __new__ (and thus __init__) on unpickling.
        return ()

    def get(self, k, default=None):
        r""".get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."""
        return super().get(k, default)

    def setdefault(self, k, default=None):
        r"""D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D."""
        if self._frozen:
            if k not in self:
                raise KeyError(k)
        return super().setdefault(k, default)

    def pop(self, k):
        r"""D.pop(k[,d]) -> v, is not valid on a fixeddict, as it removes the key."""
        raise NotImplementedError

    def update(self, mapping=(), **kwargs):
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
    def fromkeys(cls, keys):
        """Returns a new dict with keys from iterable and values equal to value."""
        return super().fromkeys(k for k in keys)

    def freeze(self):
        """Freeze the internal dictionary, such that no more keys may be added."""
        self._frozen = True

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_classes', exit=False)
    doctest.testmod(verbose=False)
