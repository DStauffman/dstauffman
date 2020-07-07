r"""
Contains the high level classes used to subclass other classes.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Added mutable integer Counter class in January 2016.
#.  Updated by David C. Stauffer in June 2020 to make MetaClass methods public for direct use if desired.
"""

#%% Imports
import copy
import doctest
import pickle
import sys
import unittest
import warnings

try:
    import h5py
except ImportError: # pragma: no cover
    warnings.warn('h5py was not imported, so some file save and load capabilities will be limited.')

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

#%% Methods - save_hdf5
def save_hdf5(self, filename=''):
    r"""
    Save the object to disk as an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to save

    """
    # exit if no filename is given
    if not filename:
        return
    # Save data
    with h5py.File(filename, 'w') as file:
        grp = file.create_group('self')
        for key in vars(self):
            value = getattr(self, key)
            if value is not None:
                grp.create_dataset(key, data=value)

#%% Methods - load_hdf5
def load_hdf5(cls, filename=''):
    r"""
    Load the object from disk.

    Parameters
    ----------
    filename : str
        Name of the file to load

    """
    if not filename:
        raise ValueError('No file specified to load.')
    # Load data
    out = cls()
    with h5py.File(filename, 'r') as file:
        for key in file:
            grp = file[key]
            for field in grp:
                # Note grp[field].value is now grp[field][()] because of updated HDF5 API
                setattr(out, field, grp[field][()])
    return out

#%% Methods - save_pickle
def save_pickle(self, filename):
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

#%% Methods - load_pickle
def load_pickle(cls, filename):
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

#%% Methods - save_method
def save_method(self, filename='', use_hdf5=True):
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
    if not use_hdf5:
        # Version 1 (Pickle):
        save_pickle(self, filename.replace('hdf5', 'pkl'))
    else:
        # Version 2 (HDF5):
        save_hdf5(self, filename)

#%% Methods - load_method
def load_method(cls, filename='', use_hdf5=True):
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
        out = load_pickle(cls, filename.replace('hdf5', 'pkl'))
    else:
        # Version 2 (HDF5):
        out = load_hdf5(cls, filename)
    return out

#%% pprint_dict
def pprint_dict(dct, *, name='', indent=1, align=True, disp=True, offset=0):
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
    text = []
    if name:
        text.append(' ' * offset + name)
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
                    disp=False, return_text=True, offset=offset+indent)
            except:
                # TODO: do I need this check or just let it fail?
                warnings.warn('pprint recursive call failed, reverting to default.')
                this_pad = ' ' * (pad_len - len(this_key)) if align else ''
                this_line = f'{this_indent}{this_key}{this_pad} = {this_value}'
        else:
            this_pad = ' ' * (pad_len - len(this_key)) if align else ''
            this_line = f'{this_indent}{this_key}{this_pad} = {this_value}'
        text.append(this_line)
    text = '\n'.join(text)
    if disp:
        print(text)
    return text

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

    def pprint(self, return_text=False, **kwargs):
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
    unittest.main(module='dstauffman.tests.test_classes', exit=False)
    doctest.testmod(verbose=False)
