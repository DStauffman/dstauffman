r"""
Test file for the `enums` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2015.
"""

#%% Imports
from enum import unique
import unittest

import numpy as np

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

class _Example_Consecutive(dcs.IntEnumPlus):
    zero  = 0
    one   = 1
    two   = 2

class _Example_Consecutive2(dcs.IntEnumPlus):
    zero  = 0
    one   = 1
    skip  = 9

class _Example_Consecutive3(dcs.IntEnumPlus):
    zero = 0
    one  = 1
    dup  = 0

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
        self.enum = dcs.IntEnumPlus('Enum1', 'one two three')

    def test_consecutive(self):
        enum = dcs.consecutive(_Example_Consecutive)
        self.assertTrue(isinstance(enum, dcs.enums._EnumMetaPlus))

    def test_consecutive_but_not_zero(self):
        with self.assertRaises(ValueError) as context:
            dcs.consecutive(self.enum)
        self.assertEqual(str(context.exception), 'Bad starting value (should be zero): 1')

    def test_unique_but_non_consecutive(self):
        with self.assertRaises(ValueError) as context:
            dcs.consecutive(_Example_Consecutive2)
        self.assertEqual(str(context.exception), 'Non-consecutive values found in _Example_Consecutive2: skip:9')

    def test_not_unique(self):
        with self.assertRaises(ValueError) as context:
            dcs.consecutive(_Example_Consecutive3)
        self.assertEqual(str(context.exception), 'Duplicate values found in _Example_Consecutive3: dup -> zero')

#%% ReturnCodes
class Test_ReturnCodes(unittest.TestCase):
    r"""
    Tests the ReturnCodes enumerator with the following cases:
        Clean code
        Not clean codes
    """
    def test_clean(self):
        # A clean exit should return zero
        self.assertEqual(dcs.ReturnCodes.clean, 0)

    def test_not_clean(self):
        # All non-clean exists should return an integer greater than 0
        rc = dcs.ReturnCodes
        for key in rc.__members__:
            if key == 'clean':
                continue
            value = getattr(rc, key)
            self.assertGreater(value, 0)
            self.assertIsInstance(value, int)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
