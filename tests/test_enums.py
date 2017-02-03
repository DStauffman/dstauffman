# -*- coding: utf-8 -*-
r"""
Test file for the `enums` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in July 2015.
"""

#%% Imports
from enum import unique
import numpy as np
import unittest
import dstauffman as dcs

#%% Support
class _Example_Enum(dcs.IntEnumPlus):
    field_one = 1
    field_two = 2
    field_ten = 10

class _Example_non_unique(dcs.IntEnumPlus):
    one         = 1
    two         = 2
    another_one = 1

#%% IntEnumPlus
class Test_IntEnumPlus(unittest.TestCase):
    r"""
    Tests the IntEnumPlus class by making a enum instance and testing all the methods.
    """
    def test_printing_instance_str(self):
        with dcs.capture_output() as out:
            print(_Example_Enum.field_one)
            print(_Example_Enum.field_two)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '_Example_Enum.field_one: 1\n_Example_Enum.field_two: 2')

    def test_printing_instance_repr(self):
        with dcs.capture_output() as out:
            print(repr(_Example_Enum.field_one))
            print(repr(_Example_Enum.field_two))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '<_Example_Enum.field_one: 1>\n<_Example_Enum.field_two: 2>')

    def test_printing_class_str(self):
        with dcs.capture_output() as out:
            print(_Example_Enum)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '_Example_Enum.field_one: 1\n_Example_Enum.field_two: 2\n_Example_Enum.field_ten: 10')

    def test_printing_class_repr(self):
        with dcs.capture_output() as out:
            print(repr(_Example_Enum))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '<_Example_Enum.field_one: 1>\n<_Example_Enum.field_two: 2>\n<_Example_Enum.field_ten: 10>')

    def test_list_of_names(self):
        list_of_names = _Example_Enum.list_of_names()
        np.testing.assert_array_equal(list_of_names, ['field_one', 'field_two', 'field_ten'])

    def test_list_of_values(self):
        list_of_values = _Example_Enum.list_of_values()
        np.testing.assert_array_equal(list_of_values, [1, 2, 10])

    def test_num_values(self):
        num_values = _Example_Enum.num_values
        self.assertEqual(num_values, 3)

    def test_min_value(self):
        min_value = _Example_Enum.min_value
        self.assertEqual(min_value, 1)

    def test_max_value(self):
        max_value = _Example_Enum.max_value
        self.assertEqual(max_value, 10)

    def test_bad_attribute(self):
        with self.assertRaises(AttributeError):
            _Example_Enum.non_existant_field

    def test_bad_uniqueness(self):
        with self.assertRaises(ValueError):
            @unique
            class _BadUnique(dcs.IntEnumPlus):
                a = 1
                b = 2
                c = 2

#%% consecutive
class Test_consecutive(unittest.TestCase):
    r"""
    Tests the consecutive function with the following cases:
        Nominal consecutive enum
        Unique, but not consecutive
        Not unique
    """
    def setUp(self):
        self.enum1 = dcs.IntEnumPlus('Enum1', 'one two three')

    def test_consecutive(self):
        enum1 = dcs.consecutive(self.enum1)
        self.assertTrue(isinstance(enum1, dcs.enums._EnumMetaPlus))

    def test_unique_but_non_consecutive(self):
        with self.assertRaises(ValueError):
            dcs.consecutive(_Example_Enum)

    def test_not_unique(self):
        with self.assertRaises(ValueError):
            dcs.consecutive(_Example_non_unique)

#%% dist_enum_and_mons
class Test_dist_enum_and_mons(unittest.TestCase):
    r"""
    Tests the dist_enum_and_mons function with the following cases:
        Nominal usage
        All in one bin
    """
    def setUp(self):
        self.num = 100000
        self.distribution = 1./100*np.array([10, 20, 30, 40])
        self.max_months = np.array([1, 10, 50, 5])
        self.max_months = np.array([1, 1, 1, 1])
        self.prng = np.random.RandomState()
        self.per_lim = 0.01

    def test_calling(self):
        (state, mons) = dcs.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months)
        breakout = np.histogram(state, bins = [0.5, 1.5, 2.5, 3.5, 4.5])[0]
        breakout_per = breakout / self.num
        for ix in range(len(self.distribution)):
            self.assertTrue(np.abs(breakout_per[ix] - self.distribution[ix]) <= self.per_lim)
        self.assertTrue(np.all(mons <= self.max_months[state-1]) and np.all(mons >= 1))

    def test_all_in_one_bin(self):
        for i in range(4):
            temp = np.zeros(4)
            temp[i] = 1
            (tb_state, _) = dcs.dist_enum_and_mons(self.num, temp, self.prng, max_months=self.max_months)
            self.assertTrue(np.all(tb_state == i+1))

    def test_alpha_and_beta(self):
        pass #TODO: write this

    def test_different_start_num(self):
        (state1, mons1) = dcs.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months)
        (state2, mons2) = dcs.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=self.max_months, start_num=101)
        np.testing.assert_equal(set(state1), {1, 2, 3, 4})
        np.testing.assert_equal(set(state2), {101, 102, 103, 104})
        np.testing.assert_equal(set(mons1), {1})
        np.testing.assert_equal(set(mons2), {1})

    def test_scalar_max_months(self):
        (state1, mons1) = dcs.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=1)
        (state2, mons2) = dcs.dist_enum_and_mons(self.num, self.distribution, self.prng, max_months=3)
        np.testing.assert_equal(set(state1), {1, 2, 3, 4})
        np.testing.assert_equal(set(state2), {1, 2, 3, 4})
        np.testing.assert_equal(set(mons1), {1})
        np.testing.assert_equal(set(mons2), {1, 2, 3})

    def test_max_months_is_none(self):
        (state, mons) = dcs.dist_enum_and_mons(self.num, self.distribution, self.prng)
        np.testing.assert_equal(set(state), {1, 2, 3, 4})
        self.assertTrue(mons is None)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
