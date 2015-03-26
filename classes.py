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
import unittest

#%% Functions - frozen
def frozen(set):
    r"""
    Raise an error when trying to set an undeclared name, or when calling
    from a method other than Frozen.__init__ or the __init__ method of
    a class derived from Frozen.
    """
    def set_attr(self, name, value):
        import sys
        if hasattr(self, name):
            # If attribute already exists, simply set it
            set(self, name, value)
            return
        elif sys._getframe(1).f_code.co_name is '__init__':
            # Allow __setattr__ calls in __init__ calls of proper object types
            for k, v in sys._getframe(1).f_locals.items():
                if k == 'self' and isinstance(v, self.__class__):
                    set(self, name, value)
                    return
        raise AttributeError('You cannot add attributes to {}'.format(self))
    return set_attr

#%% Classes - Frozen
class Frozen(object):
    r"""
    Subclasses of Frozen are frozen, i.e. it is impossibile to add
    new attributes to them and their instances.
    """
    __setattr__ = frozen(object.__setattr__)
    class __metaclass__(type):
        __setattr__ = frozen(type.__setattr__)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_classes', exit=False)
