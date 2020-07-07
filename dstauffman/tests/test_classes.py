r"""
Test file for the `classes` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import copy
import os
import pickle
import unittest
from collections.abc import Mapping

import numpy as np

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

class _Example_SaveAndLoad(dcs.Frozen, metaclass=dcs.SaveAndLoad):
    def __init__(self):
        self.x = np.array([1, 3, 5])
        self.y = np.array([2, 4, 6])
        self.z = None

class _Example_SaveAndLoadPickle(dcs.Frozen, metaclass=dcs.SaveAndLoadPickle):
    def __init__(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([4, 5, 6])

class _Example_No_Override(object, metaclass=dcs.SaveAndLoad):
    @staticmethod
    def save():
        return 1
    @staticmethod
    def load():
        return 2

class _Example_No_Override2(object, metaclass=dcs.SaveAndLoadPickle):
    @staticmethod
    def save():
        return 1
    @staticmethod
    def load():
        return 2

#%% Frozen
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

#%% pprint_dict
class Test_pprint_dict(unittest.TestCase):
    r"""
    Tests the pprint_dict function with the following cases:
        Nominal
        No name
        Different indentation
        No alignment
    """
    def setUp(self):
        self.name   = 'Example'
        self.dct    = {'a': 1, 'bb': 2, 'ccc': 3}

    def test_nominal(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')

    def test_no_name(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'a   = 1')
        self.assertEqual(lines[1], ' bb  = 2')
        self.assertEqual(lines[2], ' ccc = 3')

    def test_indent(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name, indent=4)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], '    a   = 1')
        self.assertEqual(lines[2], '    bb  = 2')
        self.assertEqual(lines[3], '    ccc = 3')

    def test_no_align(self):
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name, align=False)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a = 1')
        self.assertEqual(lines[2], ' bb = 2')
        self.assertEqual(lines[3], ' ccc = 3')

    def test_printed(self):
        with dcs.capture_output() as out:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=True)
        output = out.getvalue().strip()
        lines = output.split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')
        self.assertEqual(text, output)

    def test_not_printed(self):
        with dcs.capture_output() as out:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=False)
        output = out.getvalue().strip()
        self.assertEqual(output, '')
        lines = text.split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')

#%% SaveAndLoad
class Test_SaveAndLoad(unittest.TestCase):
    r"""
    Tests SaveAndLoad metaclass.
    """
    def setUp(self):
        folder          = dcs.get_tests_dir()
        self.results1   = _Example_SaveAndLoad()
        self.results2   = _Example_SaveAndLoadPickle()
        self.save_path1 = os.path.join(folder, 'results_test_save.hdf5')
        self.save_path2 = os.path.join(folder, 'results_test_save.pkl')

    def test_save1(self):
        self.assertTrue(hasattr(self.results1, 'save'))
        self.assertTrue(hasattr(self.results1, 'load'))

    def test_save2(self):
        self.assertTrue(hasattr(self.results2, 'save'))
        self.assertTrue(hasattr(self.results2, 'load'))

    def test_save3(self):
        temp = _Example_No_Override()
        self.assertTrue(hasattr(temp, 'save'))
        self.assertTrue(hasattr(temp, 'load'))
        self.assertEqual(temp.save(), 1)
        self.assertEqual(temp.load(), 2)

    def test_save4(self):
        temp = _Example_No_Override2()
        self.assertTrue(hasattr(temp, 'save'))
        self.assertTrue(hasattr(temp, 'load'))
        self.assertEqual(temp.save(), 1)
        self.assertEqual(temp.load(), 2)

    def test_saving_hdf5(self):
        self.results1.save(self.save_path1)
        results = self.results1.load(self.save_path1)
        self.assertTrue(dcs.compare_two_classes(results, self.results1, suppress_output=True, compare_recursively=True))

    def test_saving_pickle1(self):
        self.results1.save(self.save_path1, use_hdf5=False)
        results = self.results1.load(self.save_path2, use_hdf5=False)
        self.assertTrue(dcs.compare_two_classes(results, self.results1, suppress_output=True, compare_recursively=True))

    def test_saving_pickle2(self):
        self.results2.save(self.save_path2)
        results = self.results2.load(self.save_path2)
        self.assertTrue(dcs.compare_two_classes(results, self.results2, suppress_output=True, compare_recursively=True))

    def test_no_filename(self):
        self.results1.save('')
        with self.assertRaises(ValueError):
            self.results1.load('')

    def tearDown(self):
        if os.path.isfile(self.save_path1):
            os.remove(self.save_path1)
        if os.path.isfile(self.save_path2):
            os.remove(self.save_path2)

#%% Counter
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

#%% FixedDict
class Test_FixedDict(unittest.TestCase):
    r"""
    Tests FixedDict class with the following cases:
        TBD
    """
    def setUp(self):
        self.keys = {'key1', 'key2'}
        self.fixed = dcs.FixedDict({'key1': 1, 'key2': 2})

    def test_nominal(self):
        self.assertEqual(self.keys, set(self.fixed))

    def test_key_creation_and_freeze(self):
        self.fixed['new_key'] = 5
        self.assertTrue('new_key' in self.fixed)
        self.fixed.freeze()
        with self.assertRaises(KeyError):
            self.fixed['bad_key'] = 6

    def test_change_value(self):
        self.fixed.freeze()
        self.assertEqual(self.fixed['key1'], 1)
        self.fixed['key1'] = 5
        self.assertEqual(self.fixed['key1'], 5)

    def test_iteration(self):
        c = 0
        for (k, v) in self.fixed.items():
            c += 1
            self.assertIn(k, self.keys)
            if k == 'key1':
                self.assertEqual(v, 1)
            elif k == 'key2':
                self.assertEqual(v, 2)
        self.assertEqual(c, 2)

    def test_fromkeys(self):
        fixed = dcs.FixedDict.fromkeys(self.keys)
        self.assertEqual(self.keys, set(fixed))

    def test_get(self):
        value = self.fixed.get('key1')
        self.assertEqual(value, 1)

    def test_setdefault(self):
        self.fixed.setdefault('new_key', 1)
        self.assertEqual(self.fixed['new_key'], 1)
        self.fixed.freeze()
        self.fixed.setdefault('key1', 5)
        self.assertEqual(self.fixed['key1'], 1)
        with self.assertRaises(KeyError):
            self.fixed.setdefault('newest_key', 5)

    def test_update(self):
        self.fixed.freeze()
        dict2 = {'key1': 3}
        self.fixed.update(dict2)
        self.assertEqual(self.fixed['key1'], 3)
        dict2['bad_key'] = 5
        with self.assertRaises(KeyError):
            self.fixed.update(dict2)

    def test_update_not_frozen(self):
        dict2 = {'key1': 3, 'new_key': 5}
        self.fixed.update(dict2)
        self.assertEqual(self.fixed['key1'], 3)
        self.assertTrue('new_key' in self.fixed)

    def test_update_kwargs(self):
        self.fixed.freeze()
        dict2 = {'key1': 3}
        self.fixed.update(**dict2)
        self.assertEqual(self.fixed['key1'], 3)
        dict2['bad_key'] = 5
        with self.assertRaises(KeyError):
            self.fixed.update(**dict2)

    def test_isinstance(self):
        self.assertTrue(isinstance(self.fixed, Mapping))
        self.assertTrue(isinstance(self.fixed, dict))

    def test_bad_delete(self):
        with self.assertRaises(NotImplementedError):
            del self.fixed['key1']

    def test_bad_pop(self):
        with self.assertRaises(NotImplementedError):
            self.fixed.pop('key1')

    def test_copy(self):
        self.fixed['mutable'] = [1, 2, 3]
        new = copy.copy(self.fixed)
        self.assertEqual(self.keys | {'mutable'}, set(new))
        self.assertIs(new['mutable'], self.fixed['mutable'])
        new['mutable'][1] = 5
        self.assertEqual(new['mutable'][1], 5)
        self.assertEqual(self.fixed['mutable'][1], 5)

    def test_deepcopy(self):
        self.fixed['mutable'] = [1, 2, 3]
        new = copy.deepcopy(self.fixed)
        self.assertEqual(self.keys | {'mutable'}, set(new))
        self.assertFalse(new['mutable'] is self.fixed['mutable'])
        new['mutable'][1] = 5
        self.assertEqual(new['mutable'][1], 5)
        self.assertEqual(self.fixed['mutable'][1], 2)

    def test_pickling(self):
        data = pickle.dumps(self.fixed)
        new = pickle.loads(data)
        self.assertEqual(self.fixed, new)
        self.assertEqual(self.fixed._frozen, new._frozen)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
