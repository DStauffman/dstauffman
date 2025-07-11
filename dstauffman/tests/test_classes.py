r"""
Test file for the `classes` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.

"""

# %% Imports
from __future__ import annotations

import pathlib
from typing import Callable, ClassVar, TYPE_CHECKING
import unittest

from slog import capture_output

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

    inf = np.inf
else:
    from math import inf
if dcs.HAVE_PANDAS:
    from pandas import DataFrame

if TYPE_CHECKING:
    from mypy_extensions import DefaultNamedArg
    from numpy.typing import NDArray

    _I = NDArray[np.int_]
    _N = NDArray[np.floating]


# %% Locals classes for testing
class _Example_Frozen(dcs.Frozen):
    def __init__(self, dummy: int | None = None):
        self.field_one: int | str = 1
        self.field_two: int = 2
        self.field_ten: int = 10
        self.dummy: int | None = dummy if dummy is not None else 0


class _Example_SaveAndLoad(dcs.Frozen, metaclass=dcs.SaveAndLoad):
    load: ClassVar[
        Callable[
            [pathlib.Path | None, DefaultNamedArg(bool, "return_meta")],  # noqa: F821
            _Example_SaveAndLoad,
        ]
    ]
    save: Callable[
        [
            pathlib.Path | None,
            DefaultNamedArg(dict, "meta"),  # noqa: F821
            DefaultNamedArg(set, "exclusions"),  # noqa: F821
        ],
        None,
    ]
    x: _I | list[int]
    y: _I | list[int]
    z: int | None

    def __init__(self) -> None:
        if dcs.HAVE_NUMPY:
            self.x = np.array([1, 3, 5])
            self.y = np.array([2, 4, 6])
        else:
            self.x = [1, 3, 5]
            self.y = [2, 4, 6]
        self.z = None


class _Example_No_Override(object, metaclass=dcs.SaveAndLoad):
    @staticmethod
    def save() -> int:
        return 1

    @staticmethod
    def load() -> int:
        return 2


class _Example_Times(object):
    def __init__(self, time: _N, data: _N, name: str = "name"):
        self.time = time
        self.data = data
        self.name = name

    def chop(self, ti: float = -inf, tf: float = inf) -> None:
        dcs.chop_time(
            self, ti=ti, tf=tf, time_field="time", exclude=frozenset({"name",})  # fmt: skip
        )

    def subsample(self, skip: int = 30, start: int = 0) -> None:
        dcs.subsample_class(
            self, skip=skip, start=start, skip_fields=frozenset({"name",})  # fmt: skip
        )


# %% save_hdf5 & load_hdf5 - mostly covered by SaveAndLoad
class Test_save_and_load_hdf5(unittest.TestCase):
    r"""Additionally tests the save and load HDF5 functions with a pandas DataFrame."""

    def setUp(self) -> None:
        self.filename = dcs.get_tests_dir() / "results_test_df_save.hdf5"

    @unittest.skipIf(not dcs.HAVE_PANDAS or not dcs.HAVE_H5PY, "Skipping due to missing pandas/h5py dependency.")
    def test_pandas(self) -> None:
        data = {"a": np.array([1, 3, 5]), "b": np.array([2.0, 4.0, 6.0])}
        df = DataFrame(data)
        dcs.save_hdf5(df, self.filename)
        data2 = dcs.load_hdf5(None, self.filename)
        for key in data.keys():
            np.testing.assert_array_equal(data[key], getattr(data2, key))

    def tearDown(self) -> None:
        self.filename.unlink(missing_ok=True)


# %% save_method - covered by SaveAndLoad
# %% load_method - covered by SaveAndLoad
# %% save_convert_hdf5
# TODO: add me to Class examples
# %% save_restore_hdf5
# TODO: add me to Class examples
# %% pprint
# TODO: add me to Class examples


# %% pprint_dict
class Test_pprint_dict(unittest.TestCase):
    r"""
    Tests the pprint_dict function with the following cases:
        Nominal
        No name
        Different indentation
        No alignment
    """

    def setUp(self) -> None:
        self.name = "Example"
        self.dct = {"a": 1, "bb": 2, "ccc": 3}

    def test_nominal(self) -> None:
        with capture_output() as ctx:
            dcs.pprint_dict(self.dct, name=self.name)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "Example")
        self.assertEqual(lines[1], " a   = 1")
        self.assertEqual(lines[2], " bb  = 2")
        self.assertEqual(lines[3], " ccc = 3")

    def test_no_name(self) -> None:
        with capture_output() as ctx:
            dcs.pprint_dict(self.dct)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "a   = 1")
        self.assertEqual(lines[1], " bb  = 2")
        self.assertEqual(lines[2], " ccc = 3")

    def test_indent(self) -> None:
        with capture_output() as ctx:
            dcs.pprint_dict(self.dct, name=self.name, indent=4)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "Example")
        self.assertEqual(lines[1], "    a   = 1")
        self.assertEqual(lines[2], "    bb  = 2")
        self.assertEqual(lines[3], "    ccc = 3")

    def test_no_align(self) -> None:
        with capture_output() as ctx:
            dcs.pprint_dict(self.dct, name=self.name, align=False)
        lines = ctx.get_output().split("\n")
        ctx.close()
        self.assertEqual(lines[0], "Example")
        self.assertEqual(lines[1], " a = 1")
        self.assertEqual(lines[2], " bb = 2")
        self.assertEqual(lines[3], " ccc = 3")

    def test_printed(self) -> None:
        with capture_output() as ctx:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=True)
        output = ctx.get_output()
        lines = output.split("\n")
        ctx.close()
        self.assertEqual(lines[0], "Example")
        self.assertEqual(lines[1], " a   = 1")
        self.assertEqual(lines[2], " bb  = 2")
        self.assertEqual(lines[3], " ccc = 3")
        self.assertEqual(text, output)

    def test_not_printed(self) -> None:
        with capture_output() as ctx:
            text = dcs.pprint_dict(self.dct, name=self.name, disp=False)
        output = ctx.get_output()
        lines = text.split("\n")
        ctx.close()
        self.assertEqual(output, "")
        self.assertEqual(lines[0], "Example")
        self.assertEqual(lines[1], " a   = 1")
        self.assertEqual(lines[2], " bb  = 2")
        self.assertEqual(lines[3], " ccc = 3")

    @unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
    def test_max_elements(self) -> None:
        self.dct["a"] = np.arange(10)  # type: ignore[assignment]
        text1 = dcs.pprint_dict(self.dct, name=self.name, disp=False, max_elements=2)
        text2 = dcs.pprint_dict(self.dct, name=self.name, disp=False, max_elements=20)
        text3 = dcs.pprint_dict(self.dct, name=self.name, disp=False, max_elements=0)
        lines1 = text1.split("\n")
        lines2 = text2.split("\n")
        lines3 = text3.split("\n")
        self.assertEqual(lines1[1], " a   = [0 1 2 ... 7 8 9]")
        self.assertEqual(lines2[1], " a   = [0 1 2 3 4 5 6 7 8 9]")
        if dcs.IS_WINDOWS and np.__version__.startswith("1."):
            self.assertEqual(lines3[1], " a   = <ndarray int32 (10,)>")  # pragma: no cover
        else:
            self.assertEqual(lines3[1], " a   = <ndarray int64 (10,)>")
        self.assertEqual(lines3[2], " bb  = <class 'int'>")
        self.assertEqual(lines3[3], " ccc = <class 'int'>")


# %% chop_time
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_chop_time(unittest.TestCase):
    r"""
    Tests the chop_time method with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.time = np.array([7, 1, 3, 5])
        self.data = np.array([2, 4, 6, 8])
        self.name = "name"
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


# %% subsample_class
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_subsample_class(unittest.TestCase):
    r"""
    Tests the subsample_class method with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.time = np.array([1, 7, 5, 3])
        self.data = np.array([8, 6, 4, 2])
        self.name = "nombre"
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


# %% Frozen
class Test_Frozen(unittest.TestCase):
    r"""
    Tests the Frozen class with the following cases:
        normal mode
        add new attribute to existing instance
    """

    def setUp(self) -> None:
        self.fields = ["field_one", "field_two", "field_ten"]

    def test_calling(self) -> None:
        temp = _Example_Frozen()
        for field in self.fields:
            self.assertTrue(hasattr(temp, field))
            setattr(temp, field, getattr(temp, field))

    def test_override_existing(self) -> None:
        temp = _Example_Frozen(dummy=5)
        temp.field_one = "not one"
        temp.dummy = 10
        setattr(temp, "dummy", 15)
        self.assertTrue(True)

    def test_new_attr(self) -> None:
        temp = _Example_Frozen()
        with self.assertRaises(AttributeError):
            temp.new_field_that_does_not_exist = 1  # type: ignore[attr-defined]


# %% SaveAndLoad
class Test_SaveAndLoad(unittest.TestCase):
    r"""
    Tests the SaveAndLoad metaclass with the following cases:
        has methods (x4)
        save/load hdf5
    """

    def setUp(self) -> None:
        folder = dcs.get_tests_dir()
        self.results_cls = _Example_SaveAndLoad
        self.results = self.results_cls()
        self.save_path = folder / "results_test_save.hdf5"

    def test_save1(self) -> None:
        self.assertTrue(hasattr(self.results, "save"))
        self.assertTrue(hasattr(self.results, "load"))

    def test_save2(self) -> None:
        temp = _Example_No_Override()
        self.assertTrue(hasattr(temp, "save"))
        self.assertTrue(hasattr(temp, "load"))
        self.assertEqual(temp.save(), 1)
        self.assertEqual(temp.load(), 2)

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_saving_hdf5(self) -> None:
        self.results.save(self.save_path)
        results = self.results_cls.load(self.save_path)
        self.assertTrue(dcs.compare_two_classes(results, self.results, suppress_output=True, compare_recursively=True))

    def test_no_filename(self) -> None:
        self.results.save(None)
        with self.assertRaises(ValueError):
            self.results_cls.load(None)

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_classes_none(self) -> None:
        self.results.z = 5
        self.results.save(self.save_path)
        results = dcs.load_hdf5(None, self.save_path)
        np.testing.assert_array_equal(results.x, self.results.x)
        np.testing.assert_array_equal(results.y, self.results.y)
        self.assertEqual(results.z, 5)

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_classes_dict(self) -> None:
        self.results.z = 5
        self.results.save(self.save_path)
        results = dcs.load_hdf5({"y": None}, self.save_path)
        self.assertFalse(hasattr(results, "x"))
        np.testing.assert_array_equal(results.y, self.results.y)
        self.assertFalse(hasattr(results, "z"))

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_classless_list(self) -> None:
        self.results.z = 5
        self.results.save(self.save_path)
        results = dcs.load_hdf5(["x", "y"], self.save_path)
        np.testing.assert_array_equal(results.x, self.results.x)
        np.testing.assert_array_equal(results.y, self.results.y)
        self.assertFalse(hasattr(results, "z"))

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_classless_set(self) -> None:
        self.results.z = 5
        self.results.save(self.save_path)
        results = dcs.load_hdf5({"x", "z"}, self.save_path)
        np.testing.assert_array_equal(results.x, self.results.x)
        self.assertFalse(hasattr(results, "y"))
        self.assertEqual(results.z, 5)

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_classless_tuple(self) -> None:
        self.results.z = 5
        self.results.save(self.save_path)
        results = dcs.load_hdf5(("x", "y", "z"), self.save_path)
        np.testing.assert_array_equal(results.x, self.results.x)
        np.testing.assert_array_equal(results.y, self.results.y)
        self.assertEqual(results.z, 5)

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_bad_class_field(self) -> None:
        dcs.save_hdf5({"x": self.results.x, "y": self.results.y, "z": self.results.z, "new": 5}, self.save_path)
        with self.assertRaises(AttributeError):
            self.results_cls.load(self.save_path)
        results = dcs.load_hdf5(None, self.save_path)
        np.testing.assert_array_equal(results.x, self.results.x)
        np.testing.assert_array_equal(results.y, self.results.y)
        self.assertFalse(hasattr(results, "z"))
        self.assertEqual(results.new, 5)

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_meta(self) -> None:
        meta = {"num_pts": len(self.results.x)}
        self.results.save(self.save_path, meta=meta, compression=None, shuffle=False)  # type: ignore[call-arg]
        results = self.results_cls.load(self.save_path)
        self.assertTrue(dcs.compare_two_classes(results, self.results, suppress_output=True, compare_recursively=True))
        (results2, meta2) = self.results_cls.load(self.save_path, return_meta=True)  # type: ignore[misc]
        self.assertTrue(dcs.compare_two_classes(results2, self.results, suppress_output=True, compare_recursively=True))  # type: ignore[has-type]
        self.assertEqual(meta2, meta)  # type: ignore[has-type]

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_compression_and_shuffle(self) -> None:
        self.results.save(self.save_path, compression=6, shuffle=True)  # type: ignore[call-arg]
        results = self.results_cls.load(self.save_path)
        self.assertTrue(dcs.compare_two_classes(results, self.results, suppress_output=True, compare_recursively=True))
        (results2, meta2) = self.results_cls.load(self.save_path, return_meta=True)  # type: ignore[misc]
        self.assertTrue(dcs.compare_two_classes(results2, self.results, suppress_output=True, compare_recursively=True))  # type: ignore[has-type]
        self.assertEqual(meta2, {})  # type: ignore[has-type]

    @unittest.skipIf(not dcs.HAVE_H5PY, "Skipping due to missing h5py dependency.")
    def test_exclusions(self) -> None:
        orig = self.results.x.copy()
        self.results.save(self.save_path, exclusions={"x", "z"})
        self.results.x = "Not original"  # type: ignore[assignment]
        results = self.results_cls.load(self.save_path)
        np.testing.assert_array_equal(results.y, self.results.y)
        np.testing.assert_array_equal(results.x, orig)
        self.assertIsNone(results.z)

    def tearDown(self) -> None:
        self.save_path.unlink(missing_ok=True)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
