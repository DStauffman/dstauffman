r"""
Test file for the `classes` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import annotations
from collections.abc import Mapping
import contextlib
import copy
import os
import pickle
from typing import Callable, ClassVar, TYPE_CHECKING
import unittest

if TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np
    inf = np.inf
else:
    from math import inf

#%% Locals classes for testing
class _Example_Frozen(dcs.Frozen):
    def __init__(self, dummy=None):
        self.field_one = 1
        self.field_two = 2
        self.field_ten = 10
        self.dummy     = dummy if dummy is not None else 0

class _Example_SaveAndLoad(dcs.Frozen, metaclass=dcs.SaveAndLoad):
    load: ClassVar[Callable[[str, DefaultNamedArg(bool, 'use_hdf5')], _Example_SaveAndLoad]]
    save: Callable[[_Example_SaveAndLoad, str, DefaultNamedArg(bool, 'use_hdf5')], None]
    def __init__(self):
        if dcs.HAVE_NUMPY:
            self.x = np.array([1, 3, 5])
            self.y = np.array([2, 4, 6])
        else:
            self.x = [1, 3, 5]
            self.y = [2, 4, 6]
        self.z = None

class _Example_SaveAndLoadPickle(dcs.Frozen, metaclass=dcs.SaveAndLoadPickle):
    load: ClassVar[Callable[[str], _Example_SaveAndLoadPickle]]
    save: Callable[[_Example_SaveAndLoadPickle, str], None]
    def __init__(self):
        if dcs.HAVE_NUMPY:
            self.a = np.array([1, 2, 3])
            self.b = np.array([4, 5, 6])
        else:
            self.a = [1, 2, 3]
            self.b = [4, 5, 6]

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

class _Example_Times(object):
    def __init__(self, time, data, name='name'):
        self.time = time
        self.data = data
        self.name = name
    def chop(self, ti=-inf, tf=inf):
        dcs.chop_time(self, ti=ti, tf=tf, time_field='time', exclude=frozenset({'name',}))
    def subsample(self, skip=30, start=0):
        dcs.subsample_class(self, skip=skip, start=start, skip_fields=frozenset({'name',}))

#%% save_hdf5 - covered by SaveAndLoad
#%% load_hdf5 - covered by SaveAndLoad
#%% save_pickle - covered by SaveAndLoad
#%% load_pickle - covered by SaveAndLoad
#%% save_method - covered by SaveAndLoad
#%% load_method - covered by SaveAndLoad

#%% pprint_dict
class Test_pprint_dict(unittest.TestCase):
    r"""
    Tests the pprint_dict function with the following cases:
        Nominal
        No name
        Different indentation
        No alignment
    """
    def setUp(self) -> None:
        self.name   = 'Example'
        self.dct    = {'a': 1, 'bb': 2, 'ccc': 3}

    def test_nominal(self) -> None:
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')

    def test_no_name(self) -> None:
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'a   = 1')
        self.assertEqual(lines[1], ' bb  = 2')
        self.assertEqual(lines[2], ' ccc = 3')

    def test_indent(self) -> None:
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name, indent=4)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], '    a   = 1')
        self.assertEqual(lines[2], '    bb  = 2')
        self.assertEqual(lines[3], '    ccc = 3')

    def test_no_align(self) -> None:
        with dcs.capture_output() as out:
            dcs.pprint_dict(self.dct, name=self.name, align=False)
        lines = out.getvalue().strip().split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a = 1')
        self.assertEqual(lines[2], ' bb = 2')
        self.assertEqual(lines[3], ' ccc = 3')

    def test_printed(self) -> None:
        with dcs.capture_output() as out:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=True)
        output = out.getvalue().strip()
        lines = output.split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')
        self.assertEqual(text, output)

    def test_not_printed(self) -> None:
        with dcs.capture_output() as out:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=False)
        output = out.getvalue().strip()
        self.assertEqual(output, '')
        lines = text.split('\n')
        self.assertEqual(lines[0], 'Example')
        self.assertEqual(lines[1], ' a   = 1')
        self.assertEqual(lines[2], ' bb  = 2')
        self.assertEqual(lines[3], ' ccc = 3')

#%% chop_time
@unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_chop_time(unittest.TestCase):
    r"""
    Tests the chop_time method with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.time = np.array([7, 1, 3, 5])
        self.data = np.array([2, 4, 6, 8])
        self.name = 'name'
        self.telm = _Example_Times(self.time, self.data, name=self.name)
        self.ti = 3
        self.tf = 6
        self.exp_time = np.array([3, 5])
        self.exp_data = np.array([6, 8])

    def test_nominal(self) -> None:
        self.telm.chop(self.ti, self.tf)
        np.testing.assert_array_equal(self.telm.time, self.exp_time)
        np.testing.assert_array_equal(self.telm.data, self.exp_data)
        self.assertEqual(self.telm.name, self.name)

#%% subsample_class
@unittest.skipIf(not dcs.HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_subsample_class(unittest.TestCase):
    r"""
    Tests the subsample_class method with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.time = np.array([1, 7, 5, 3])
        self.data = np.array([8, 6, 4, 2])
        self.name = 'nombre'
        self.telm = _Example_Times(self.time, self.data, name=self.name)
        self.skip = 2
        self.start = 1
        self.exp_time = np.array([7, 3])
        self.exp_data = np.array([6, 2])

    def test_nominal(self) -> None:
        self.telm.subsample(self.skip, start=self.start)
        np.testing.assert_array_equal(self.telm.time, self.exp_time)
        np.testing.assert_array_equal(self.telm.data, self.exp_data)
        self.assertEqual(self.telm.name, self.name)

#%% Frozen
class Test_Frozen(unittest.TestCase):
    r"""
    Test the Frozen class with the following cases:
        normal mode
        add new attribute to existing instance
    """
    def setUp(self) -> None:
        self.fields = ['field_one', 'field_two', 'field_ten']

    def test_calling(self) -> None:
        temp = _Example_Frozen()
        for field in self.fields:
            self.assertTrue(hasattr(temp, field))
            setattr(temp, field, getattr(temp, field))

    def test_override_existing(self) -> None:
        temp = _Example_Frozen(dummy=5)
        temp.field_one = 'not one'
        temp.dummy = 10
        setattr(temp, 'dummy', 15)
        self.assertTrue(True)

    def test_new_attr(self) -> None:
        temp = _Example_Frozen()
        with self.assertRaises(AttributeError):
            temp.new_field_that_does_not_exist = 1  # type: ignore[attr-defined]

#%% SaveAndLoad
class Test_SaveAndLoad(unittest.TestCase):
    r"""
    Tests SaveAndLoad metaclass with the following cases:
        has methods (x4)
        save/load hdf5
        savel/oad pickle (x2)
    """
    def setUp(self) -> None:
        folder            = dcs.get_tests_dir()
        self.results1_cls = _Example_SaveAndLoad
        self.results1     = self.results1_cls()
        self.results2_cls = _Example_SaveAndLoadPickle
        self.results2     = self.results2_cls()
        self.save_path1   = os.path.join(folder, 'results_test_save.hdf5')
        self.save_path2   = os.path.join(folder, 'results_test_save.pkl')

    def test_save1(self) -> None:
        self.assertTrue(hasattr(self.results1, 'save'))
        self.assertTrue(hasattr(self.results1, 'load'))

    def test_save2(self) -> None:
        self.assertTrue(hasattr(self.results2, 'save'))
        self.assertTrue(hasattr(self.results2, 'load'))

    def test_save3(self) -> None:
        temp = _Example_No_Override()
        self.assertTrue(hasattr(temp, 'save'))
        self.assertTrue(hasattr(temp, 'load'))
        self.assertEqual(temp.save(), 1)
        self.assertEqual(temp.load(), 2)

    def test_save4(self) -> None:
        temp = _Example_No_Override2()
        self.assertTrue(hasattr(temp, 'save'))
        self.assertTrue(hasattr(temp, 'load'))
        self.assertEqual(temp.save(), 1)
        self.assertEqual(temp.load(), 2)

    @unittest.skipIf(not dcs.HAVE_H5PY, 'Skipping due to missing h5py dependency.')
    def test_saving_hdf5(self) -> None:
        self.results1.save(self.save_path1)
        results = self.results1_cls.load(self.save_path1)
        self.assertTrue(dcs.compare_two_classes(results, self.results1, suppress_output=True, compare_recursively=True))

    def test_saving_pickle1(self) -> None:
        self.results1.save(self.save_path1, use_hdf5=False)
        results = self.results1_cls.load(self.save_path2, use_hdf5=False)
        self.assertTrue(dcs.compare_two_classes(results, self.results1, suppress_output=True, compare_recursively=True))

    def test_saving_pickle2(self) -> None:
        self.results2.save(self.save_path2)
        results = self.results2_cls.load(self.save_path2)
        self.assertTrue(dcs.compare_two_classes(results, self.results2, suppress_output=True, compare_recursively=True))

    def test_no_filename(self) -> None:
        self.results1.save('')
        with self.assertRaises(ValueError):
            self.results1_cls.load('')

    def tearDown(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.save_path1)
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.save_path2)

#%% SaveAndLoadPickle
class Test_SaveAndLoadPickle(unittest.TestCase):
    r"""
    Tests the SaveAndLoadPickle class with the following cases:
        TBD
    """
    pass # TODO: write this

#%% Counter
class Test_Counter(unittest.TestCase):
    r"""
    Tests the Counter class with the following cases:
        TBD
    """
    def test_math_int(self) -> None:
        c = dcs.Counter()
        c = c + 1  # type: ignore[assignment]
        self.assertEqual(c, 1)
        self.assertEqual(type(c), int)

    def test_math_int2(self) -> None:
        c = dcs.Counter()
        c += 1
        self.assertEqual(c, 1)
        self.assertEqual(type(c), dcs.Counter)

    def test_math_int3(self) -> None:
        c = dcs.Counter()
        c = c - 5  # type: ignore[assignment]
        self.assertEqual(c, -5)
        self.assertEqual(type(c), int)

    def test_math_int4(self) -> None:
        c = dcs.Counter()
        c -= 2
        self.assertEqual(c, -2)
        self.assertEqual(type(c), dcs.Counter)

    def test_math_int5(self) -> None:
        c = dcs.Counter(10)
        c = 0 + c  # type: ignore[assignment]
        self.assertEqual(c, 10)
        self.assertEqual(type(c), int)

    def test_math_int6(self) -> None:
        c = 0 - dcs.Counter(10)
        self.assertEqual(c, -10)
        self.assertEqual(type(c), int)

    def test_math_int7(self) -> None:
        c = dcs.Counter(10)
        c = 0 +c  # type: ignore[assignment]
        self.assertEqual(c, 10)
        self.assertEqual(type(c), int)

    def test_math_int8(self) -> None:
        c = 0 -dcs.Counter(10)
        self.assertEqual(c, -10)
        self.assertEqual(type(c), int)

    def test_math_counter(self) -> None:
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
        self.assertEqual(type(c1), dcs.Counter)
        self.assertEqual(type(c2), dcs.Counter)
        self.assertEqual(type(c3), dcs.Counter)

    def test_math_float(self) -> None:
        c = dcs.Counter(0)
        with self.assertRaises(TypeError):
            c = c + 1.5  # type: ignore[operator]
        with self.assertRaises(TypeError):
            c = c - 1.5  # type: ignore[operator]
        with self.assertRaises(TypeError):
            c += 1.5  # type: ignore[type-var]
        with self.assertRaises(TypeError):
            c -= 1.5  # type: ignore[type-var]

    def test_divide(self) -> None:
        c1 = dcs.Counter(2)
        c2 = dcs.Counter(4)
        self.assertEqual(c1 // 4, 0)
        self.assertEqual(c1 // c2, 0)
        self.assertAlmostEqual(c1 / 4, 0.5)
        with self.assertRaises(TypeError):
            c1 / c2  # type: ignore[operator]
        with self.assertRaises(TypeError):
            c1 // 5.  # type: ignore[type-var]

    def test_comp_int(self) -> None:
        c = dcs.Counter(10)
        self.assertEqual(c, 10)
        self.assertNotEqual(c, 0)
        self.assertTrue(c < 100)
        self.assertTrue(c > 0)
        self.assertTrue(c <= 10)
        self.assertTrue(c >= 10)

    def test_comp_counter(self) -> None:
        c1 = dcs.Counter(1)
        c2 = dcs.Counter(2)
        self.assertEqual(c1, c1)
        self.assertNotEqual(c1, c2)
        self.assertLess(c1, c2)
        self.assertGreater(c2, c1)
        self.assertLessEqual(c1, c1)
        self.assertGreaterEqual(c2, c2)

    def test_lists(self) -> None:
        c_list = [dcs.Counter(3), dcs.Counter(-5), dcs.Counter(1)]
        c_list.sort()
        self.assertEqual(c_list[0], -5)
        self.assertEqual(c_list[1], 1)
        self.assertEqual(c_list[2], 3)

    def test_index_list(self) -> None:
        a_list = [0, 1, 2]
        c = dcs.Counter(1)
        self.assertEqual(a_list[c], 1)  # type: ignore[call-overload]

    def test_sets(self) -> None:
        c1 = dcs.Counter(1)
        c2 = dcs.Counter(2)
        c3 = dcs.Counter(3)
        s1 = {1, 2, 3}
        s2 = set((c1, c2, c3))
        self.assertEqual(s1, s2)

    def test_abs(self) -> None:
        c1 = dcs.Counter(11)
        c2 = dcs.Counter(-12)
        self.assertEqual(abs(c1), 11)
        self.assertEqual(abs(c2), 12)

    def test_mod(self) -> None:
        c1 = dcs.Counter(5)
        c2 = dcs.Counter(4)
        self.assertEqual(c1 % 4, 1)
        self.assertEqual(c1 % c2, 1)
        with self.assertRaises(TypeError):
            c1 % 4.  # type: ignore[type-var]

    def test_print(self) -> None:
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
    Tests the FixedDict class with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.keys = {'key1', 'key2'}
        self.fixed = dcs.FixedDict({'key1': 1, 'key2': 2})

    def test_nominal(self) -> None:
        self.assertEqual(self.keys, set(self.fixed))

    def test_key_creation_and_freeze(self) -> None:
        self.fixed['new_key'] = 5
        self.assertTrue('new_key' in self.fixed)
        self.fixed.freeze()
        with self.assertRaises(KeyError):
            self.fixed['bad_key'] = 6

    def test_change_value(self) -> None:
        self.fixed.freeze()
        self.assertEqual(self.fixed['key1'], 1)
        self.fixed['key1'] = 5
        self.assertEqual(self.fixed['key1'], 5)

    def test_iteration(self) -> None:
        c = 0
        for (k, v) in self.fixed.items():
            c += 1
            self.assertIn(k, self.keys)
            if k == 'key1':
                self.assertEqual(v, 1)
            elif k == 'key2':
                self.assertEqual(v, 2)
        self.assertEqual(c, 2)

    def test_fromkeys(self) -> None:
        fixed = dcs.FixedDict.fromkeys(self.keys)
        self.assertEqual(self.keys, set(fixed))

    def test_get(self) -> None:
        value = self.fixed.get('key1')
        self.assertEqual(value, 1)

    def test_setdefault(self) -> None:
        self.fixed.setdefault('new_key', 1)
        self.assertEqual(self.fixed['new_key'], 1)
        self.fixed.freeze()
        self.fixed.setdefault('key1', 5)
        self.assertEqual(self.fixed['key1'], 1)
        with self.assertRaises(KeyError):
            self.fixed.setdefault('newest_key', 5)

    def test_update(self) -> None:
        self.fixed.freeze()
        dict2 = {'key1': 3}
        self.fixed.update(dict2)
        self.assertEqual(self.fixed['key1'], 3)
        dict2['bad_key'] = 5
        with self.assertRaises(KeyError):
            self.fixed.update(dict2)

    def test_update_not_frozen(self) -> None:
        dict2 = {'key1': 3, 'new_key': 5}
        self.fixed.update(dict2)
        self.assertEqual(self.fixed['key1'], 3)
        self.assertTrue('new_key' in self.fixed)

    def test_update_kwargs(self) -> None:
        self.fixed.freeze()
        dict2 = {'key1': 3}
        self.fixed.update(**dict2)
        self.assertEqual(self.fixed['key1'], 3)
        dict2['bad_key'] = 5
        with self.assertRaises(KeyError):
            self.fixed.update(**dict2)

    def test_isinstance(self) -> None:
        self.assertTrue(isinstance(self.fixed, Mapping))
        self.assertTrue(isinstance(self.fixed, dict))

    def test_bad_delete(self) -> None:
        with self.assertRaises(NotImplementedError):
            del self.fixed['key1']

    def test_bad_pop(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.fixed.pop('key1')

    def test_copy(self) -> None:
        self.fixed['mutable'] = [1, 2, 3]
        new = copy.copy(self.fixed)
        self.assertEqual(self.keys | {'mutable'}, set(new))
        self.assertIs(new['mutable'], self.fixed['mutable'])
        new['mutable'][1] = 5
        self.assertEqual(new['mutable'][1], 5)
        self.assertEqual(self.fixed['mutable'][1], 5)

    def test_deepcopy(self) -> None:
        self.fixed['mutable'] = [1, 2, 3]
        new = copy.deepcopy(self.fixed)
        self.assertEqual(self.keys | {'mutable'}, set(new))
        self.assertFalse(new['mutable'] is self.fixed['mutable'])
        new['mutable'][1] = 5
        self.assertEqual(new['mutable'][1], 5)
        self.assertEqual(self.fixed['mutable'][1], 2)

    def test_pickling(self) -> None:
        data = pickle.dumps(self.fixed)
        new = pickle.loads(data)
        self.assertEqual(self.fixed, new)
        self.assertEqual(self.fixed._frozen, new._frozen)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
