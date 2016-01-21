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
        c = dcs.Counter()
        c = c + 1
        self.assertEqual(c, 1)

    def test_incrementing2(self):
        c = dcs.Counter(1)
        c = c + 5
        self.assertEqual(c, 6)

    def test_incrementing3(self):
        c = dcs.Counter(7)
        c += 3
        self.assertEqual(c, 10)

    def test_incrementing4(self):
        c = dcs.Counter(0)
        with self.assertRaises(TypeError):
            c = c + 1.5

    def test_incrementing5(self):
        c = dcs.Counter(0)
        with self.assertRaises(TypeError):
            c += 1.5

    def test_incrementing6(self):
        c1 = dcs.Counter(10)
        c2 = dcs.Counter(2)
        c1 += c2
        self.assertEqual(c1, 12)

    def test_comparing1(self):
        c = dcs.Counter()
        self.assertEqual(c, 0)

    def test_comparing2(self):
        c = dcs.Counter()
        self.assertNotEqual(c, 1)

    def test_comparing3(self):
        c1 = dcs.Counter(10)
        c2 = dcs.Counter(11)
        c1 += 1
        self.assertEqual(c1, c2)

    def test_comparing4(self):
        c1 = dcs.Counter(20)
        c2 = dcs.Counter(21)
        self.assertNotEqual(c1, c2)

    def test_less_than1(self):
        c = dcs.Counter()
        self.assertTrue(c < 5)

    def test_less_than2(self):
        c1 = dcs.Counter()
        c2 = dcs.Counter(5)
        self.assertLess(c1, c2)

    def test_less_than3(self):
        c = dcs.Counter()
        self.assertTrue(c <= 0)

    def test_less_than4(self):
        c1 = dcs.Counter()
        c2 = dcs.Counter(0)
        self.assertLessEqual(c1, c2)

    def test_greater_than1(self):
        c = dcs.Counter(10)
        self.assertTrue(c > 5)

    def test_greater_than2(self):
        c1 = dcs.Counter(4)
        c2 = dcs.Counter(3)
        self.assertGreater(c1, c2)

    def test_greater_than3(self):
        c = dcs.Counter()
        self.assertTrue(c >= 0)

    def test_greater_than4(self):
        c1 = dcs.Counter()
        c2 = dcs.Counter(0)
        self.assertGreaterEqual(c1, c2)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
