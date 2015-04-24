# -*- coding: utf-8 -*-
r"""
Classes module file for the "dstauffman" library.  It contains the high level classes used to
subclass other classes.

Notes
-----
#. Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import sys
import unittest
from dstauffman.utils import disp

#%% Functions - frozen
def frozen(set):
    r"""
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
        elif sys._getframe(1).f_code.co_name is '__init__':
            # Allow __setattr__ calls in __init__ calls of proper object types
            for key, val in sys._getframe(1).f_locals.items():
                if key=='self' and isinstance(val, self.__class__):
                    set(self, name, value)
                    return
        raise AttributeError('You cannot add attributes to {}'.format(self))
    # return the custom defined function
    return set_attr

#%% Classes - Frozen
class Frozen(object):
    r"""
    Subclasses of Frozen are frozen, i.e. it is impossibile to add
    new attributes to them and their instances.

    Additionally a more pretty print and explicit form of __repr__ is
    defined based on the `disp` function.
    """
    # freeze the set attributes function based on the above `frozen` funcion
    __setattr__ = frozen(object.__setattr__)
    class __metaclass__(type):
        __setattr__ = frozen(type.__setattr__)
    # override the default repr options
    def __repr__(self):
        return disp(self, suppress_output=True)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_classes', exit=False)
