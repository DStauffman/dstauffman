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
    def test_math_int(self):
        c = dcs.Counter()
        c = c + 1
        self.assertEqual(c, 1)

    def test_math_int2(self):
        c = dcs.Counter()
        c += 1
        self.assertEqual(c, 1)

    def test_math_int3(self):
        c = dcs.Counter()
        c = c - 5
        self.assertEqual(c, -5)

    def test_math_int4(self):
        c = dcs.Counter()
        c -= 2
        self.assertEqual(c, -2)

    def test_math_int5(self):
        c = dcs.Counter(10)
        c = 0 + c
        self.assertEqual(c, 10)

    def test_math_int6(self):
        c = 0 - dcs.Counter(10)
        self.assertEqual(c, -10)

    def test_math_int7(self):
        c = dcs.Counter(10)
        c = 0 +c
        self.assertEqual(c, 10)

    def test_math_int8(self):
        c = 0 -dcs.Counter(10)
        self.assertEqual(c, -10)

    def test_math_counter(self):
        c1 = dcs.Counter(10)
        c2 = dcs.Counter(-5)
        c3 = c1 + c2
        self.assertEqual(c3, dcs.Counter(5))
        c1 += c2
        self.assertEqual(c1, dcs.Counter(5))
        c3 = c1 - c2
        self.assertEqual(c3, dcs.Counter(10))
        c3 = c1 + (-c2)
        self.assertEqual(c3, dcs.Counter(10))
        c3 = +c1 -c2
        self.assertEqual(c3, dcs.Counter(10))
        c1 -= c2
        self.assertEqual(c1, dcs.Counter(10))

    def test_math_float(self):
        c = dcs.Counter(0)
        with self.assertRaises(TypeError):
            c = c + 1.5
        with self.assertRaises(TypeError):
            c = c - 1.5
        with self.assertRaises(TypeError):
            c += 1.5
        with self.assertRaises(TypeError):
            c -= 1.5

    def test_divide(self):
        c1 = dcs.Counter(2)
        c2 = dcs.Counter(4)
        self.assertEqual(c1 // 4, 0)
        self.assertEqual(c1 // c2, 0)
        self.assertAlmostEqual(c1 / 4, 0.5)
        with self.assertRaises(TypeError):
            c1 / c2
        with self.assertRaises(TypeError):
            c1 // 5.

    def test_comp_int(self):
        c = dcs.Counter(10)
        self.assertEqual(c, 10)
        self.assertNotEqual(c, 0)
        self.assertTrue(c < 100)
        self.assertTrue(c > 0)
        self.assertTrue(c <= 10)
        self.assertTrue(c >= 10)

    def test_comp_counter(self):
        c1 = dcs.Counter(1)
        c2 = dcs.Counter(2)
        self.assertEqual(c1, c1)
        self.assertNotEqual(c1, c2)
        self.assertLess(c1, c2)
        self.assertGreater(c2, c1)
        self.assertLessEqual(c1, c1)
        self.assertGreaterEqual(c2, c2)

    def test_lists(self):
        c_list = [dcs.Counter(3), dcs.Counter(-5), dcs.Counter(1)]
        c_list.sort()
        self.assertEqual(c_list[0], -5)
        self.assertEqual(c_list[1], 1)
        self.assertEqual(c_list[2], 3)

    def test_index_list(self):
        a_list = [0, 1, 2]
        c = dcs.Counter(1)
        self.assertEqual(a_list[c], 1)

    def test_sets(self):
        c1 = dcs.Counter(1)
        c2 = dcs.Counter(2)
        c3 = dcs.Counter(3)
        s1 = {1, 2, 3}
        s2 = set((c1, c2, c3))
        self.assertEqual(s1, s2)

    def test_abs(self):
        c1 = dcs.Counter(11)
        c2 = dcs.Counter(-12)
        self.assertEqual(abs(c1), 11)
        self.assertEqual(abs(c2), 12)

    def test_mod(self):
        c1 = dcs.Counter(5)
        c2 = dcs.Counter(4)
        self.assertEqual(c1 % 4, 1)
        self.assertEqual(c1 % c2, 1)
        with self.assertRaises(TypeError):
            c1 % 4.

    def test_print(self):
        c1 = dcs.Counter(1)
        with dcs.capture_output() as out:
            print(c1)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '1')
        output = repr(c1)
        self.assertEqual(output, 'Counter(1)')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
