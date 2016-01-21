# -*- coding: utf-8 -*-
r"""
Test file for the `classes` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import unittest
import dstauffman as dcs

#%% Locals classes for testing
class _Example_Frozen(dcs.Frozen):
    def __init__(self, dummy=None):
        if dummy is None:
            dummy = 0
        self.field_one = 1
        self.field_two = 2
        self.field_ten = 10
        self.dummy     = dummy

#%% Classes for testing
# Frozen
class Test_Frozen(unittest.TestCase):
    r"""
    Test Opts class, and by extension the frozen function and Frozen class using cases:
        normal mode
        add new attribute to existing instance
    """
    def setUp(self):
        self.fields = ['field_one', 'field_two', 'field_ten']

    def test_calling(self):
        temp = _Example_Frozen()
        for field in self.fields:
            self.assertTrue(hasattr(temp, field))
            setattr(temp, field, getattr(temp, field))

    def test_override_existing(self):
        temp = _Example_Frozen(dummy=5)
        temp.field_one = 'not one'
        temp.dummy = 10
        setattr(temp, 'dummy', 15)
        self.assertTrue(True)

    def test_new_attr(self):
        temp = _Example_Frozen()
        with self.assertRaises(AttributeError):
            temp.new_field_that_does_not_exist = 1

# Integer
class Test_Counter(unittest.TestCase):
    r"""
    Tests Counter class with the following cases:
        TBD
    """ 
    def test_incrementing1(self):
        i = dcs.Counter()
        i = i + 1
        self.assertEqual(i, 1)
        
    def test_incrementing2(self):
        i = dcs.Counter(1)
        i = i + 5
        self.assertEqual(i, 6)
        
    def test_incrementing3(self):
        i = dcs.Counter(7)
        i += 3
        self.assertEqual(i, 10)
        
    def test_comparing1(self):
        i = dcs.Counter()
        self.assertEqual(i, 0)
        
    def test_comparing2(self):
        i = dcs.Counter()
        self.assertNotEqual(i, 1)
        
    def test_comparing3(self):
        i1 = dcs.Counter(10)
        i2 = dcs.Counter(11)
        i1 += 1
        self.assertEqual(i1, i2)
        
    def test_comparing4(self):
        i1 = dcs.Counter(20)
        i2 = dcs.Counter(21)
        self.assertNotEqual(i1, i2)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
