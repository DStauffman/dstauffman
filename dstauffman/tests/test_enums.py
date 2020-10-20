r"""
Test file for the `enums` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in July 2015.
"""

#%% Imports
from enum import unique
from typing import ClassVar
import unittest

import numpy as np

import dstauffman as dcs

#%% Support
class _Example_Enum(dcs.IntEnumPlus):
    field_one: ClassVar[int] = 1
    field_two: ClassVar[int] = 2
    field_ten: ClassVar[int] = 10

class _Example_non_unique(dcs.IntEnumPlus):
    one: ClassVar[int]         = 1
    two: ClassVar[int]         = 2
    another_one: ClassVar[int] = 1

class _Example_Consecutive(dcs.IntEnumPlus):
    zero: ClassVar[int]  = 0
    one: ClassVar[int]   = 1
    two: ClassVar[int]   = 2

class _Example_Consecutive2(dcs.IntEnumPlus):
    zero: ClassVar[int]  = 0
    one: ClassVar[int]   = 1
    skip: ClassVar[int]  = 9

class _Example_Consecutive3(dcs.IntEnumPlus):
    zero: ClassVar[int] = 0
    one: ClassVar[int]  = 1
    dup: ClassVar[int]  = 0

#%% IntEnumPlus
class Test_IntEnumPlus(unittest.TestCase):
    r"""
    Tests the IntEnumPlus class by making a enum instance and testing all the methods.
    """
    def test_printing_instance_str(self) -> None:
        with dcs.capture_output() as out:
            print(_Example_Enum.field_one)
            print(_Example_Enum.field_two)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '_Example_Enum.field_one: 1\n_Example_Enum.field_two: 2')

    def test_printing_instance_repr(self) -> None:
        with dcs.capture_output() as out:
            print(repr(_Example_Enum.field_one))
            print(repr(_Example_Enum.field_two))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '<_Example_Enum.field_one: 1>\n<_Example_Enum.field_two: 2>')

    def test_printing_class_str(self) -> None:
        with dcs.capture_output() as out:
            print(_Example_Enum)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '_Example_Enum.field_one: 1\n_Example_Enum.field_two: 2\n_Example_Enum.field_ten: 10')

    def test_printing_class_repr(self) -> None:
        with dcs.capture_output() as out:
            print(repr(_Example_Enum))
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, '<_Example_Enum.field_one: 1>\n<_Example_Enum.field_two: 2>\n<_Example_Enum.field_ten: 10>')

    def test_list_of_names(self) -> None:
        list_of_names = _Example_Enum.list_of_names()
        np.testing.assert_array_equal(list_of_names, ['field_one', 'field_two', 'field_ten'])

    def test_list_of_values(self) -> None:
        list_of_values = _Example_Enum.list_of_values()
        np.testing.assert_array_equal(list_of_values, [1, 2, 10])

    def test_num_values(self) -> None:
        num_values = _Example_Enum.num_values
        self.assertEqual(num_values, 3)

    def test_min_value(self) -> None:
        min_value = _Example_Enum.min_value
        self.assertEqual(min_value, 1)

    def test_max_value(self) -> None:
        max_value = _Example_Enum.max_value
        self.assertEqual(max_value, 10)

    def test_bad_attribute(self) -> None:
        with self.assertRaises(AttributeError):
            _Example_Enum.non_existant_field

    def test_bad_uniqueness(self) -> None:
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
    def setUp(self) -> None:
        self.enum = dcs.IntEnumPlus('Enum1', 'one two three')  # type: ignore[call-overload]

    def test_consecutive(self) -> None:
        enum = dcs.consecutive(_Example_Consecutive)
        self.assertTrue(isinstance(enum, dcs.enums._EnumMetaPlus))

    def test_consecutive_but_not_zero(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.consecutive(self.enum)
        self.assertEqual(str(context.exception), 'Bad starting value (should be zero): 1')

    def test_unique_but_non_consecutive(self) -> None:
        with self.assertRaises(ValueError) as context:
            dcs.consecutive(_Example_Consecutive2)
        self.assertEqual(str(context.exception), 'Non-consecutive values found in _Example_Consecutive2: skip:9')

    def test_not_unique(self) -> None:
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
    def test_clean(self) -> None:
        # A clean exit should return zero
        self.assertEqual(dcs.ReturnCodes.clean, 0)

    def test_not_clean(self) -> None:
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
