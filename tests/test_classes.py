# -*- coding: utf-8 -*-
r"""
Test file for the `classes` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import unittest
import dstauffman as dcs

#%% Classes for testing
# Frozen
class Test_Frozen(unittest.TestCase):
    r"""
    Test Opts class, and by extension the frozen function and Frozen class using cases:
        normal mode
        add new attribute to existing instance
    """
    def setUp(self):
        self.opts_fields = ['case_name']

    def test_calling(self):
        opts = dcs.Opts()
        for field in self.opts_fields:
            self.assertTrue(hasattr(opts, field))
            setattr(opts, field, getattr(opts, field))

    def test_new_attr(self):
        opts = dcs.Opts()
        with self.assertRaises(AttributeError):
            opts.new_field_that_does_not_exist = 1

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
