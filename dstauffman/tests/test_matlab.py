r"""
Test file for the `matlab` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in April 2016.
#.  Expanded by David C. Stauffer in December 2018 for additional linalg functions.
"""

# %% Imports
from typing import ClassVar
import unittest

from slog import IntEnumPlus

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np


# %% Classes
class _Gender(IntEnumPlus):
    r"""Enumeration to match the MATLAB one from the test cases."""

    # fmt: off
    null: ClassVar[int]        = 0
    female: ClassVar[int]      = 1
    male: ClassVar[int]        = 2
    uncirc_male: ClassVar[int] = 3
    circ_male: ClassVar[int]   = 4
    non_binary: ClassVar[int]  = 5
    # fmt: on


# %% load_matlab
@unittest.skipIf(not dcs.HAVE_H5PY or not dcs.HAVE_NUMPY, "Skipping due to missing h5py/numpy dependency.")
class Test_load_matlab(unittest.TestCase):
    r"""
    Tests the load_matlab function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.filename1 = dcs.get_tests_dir() / "test_numbers.mat"
        self.filename2 = dcs.get_tests_dir() / "test_struct.mat"
        self.filename3 = dcs.get_tests_dir() / "test_enums.mat"
        self.filename4 = dcs.get_tests_dir() / "test_cell_array.mat"
        self.filename5 = dcs.get_tests_dir() / "test_nested.mat"
        self.row_nums = np.array([[1.0, 2.2, 3.0]])
        self.col_nums = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.mat_nums = np.array([[1, 2, 3], [4, 5, 6]])
        self.exp_enum = np.array(
            [_Gender.male, _Gender.female, _Gender.female, _Gender.circ_male, _Gender.circ_male, _Gender.circ_male], dtype=int
        )
        self.offsets = {"r": 10, "c": 20, "m": 30}
        self.enums = {"Gender": [getattr(_Gender, x) for x in _Gender.list_of_names()]}
        self.row_nums_1d = np.squeeze(self.row_nums)
        self.col_nums_1d = np.squeeze(self.col_nums)

    def test_nominal(self) -> None:
        out = dcs.load_matlab(self.filename1, squeeze=False)
        self.assertEqual(set(out.keys()), {"col_nums", "row_nums", "mat_nums"})
        np.testing.assert_array_equal(out["row_nums"], self.row_nums)
        np.testing.assert_array_equal(out["col_nums"], self.col_nums)
        np.testing.assert_array_equal(out["mat_nums"], self.mat_nums)

    def test_struct(self) -> None:
        out = dcs.load_matlab(self.filename2, squeeze=True)
        self.assertEqual(set(out.keys()), {"x"})
        np.testing.assert_array_equal(out["x"]["r"], self.row_nums_1d)
        np.testing.assert_array_equal(out["x"]["c"], self.col_nums_1d)
        np.testing.assert_array_equal(out["x"]["m"], self.mat_nums)

    def test_load_varlist(self) -> None:
        out = dcs.load_matlab(self.filename2, varlist=["y"])
        self.assertEqual(out.keys(), set())

    @unittest.expectedFailure
    def test_enum(self) -> None:
        # Enum test case not working.  Need to better understand how categorical arrays are stored.
        out = dcs.load_matlab(self.filename3, enums=self.enums)
        self.assertEqual(set(out.keys()), {"enum"})
        np.testing.assert_array_equal(out["enum"], self.exp_enum)

    def test_unknown_enum(self) -> None:
        with self.assertRaises(ValueError):
            dcs.load_matlab(self.filename3, enums={"Nope": [1, 2]})

    def test_cell_arrays(self) -> None:
        out = dcs.load_matlab(self.filename4, varlist=("cdat",))
        self.assertEqual(out.keys(), {"cdat",})  # fmt: skip
        np.testing.assert_array_equal(out["cdat"][0], self.row_nums_1d)
        np.testing.assert_array_equal(out["cdat"][1], self.col_nums_1d)
        np.testing.assert_array_equal(out["cdat"][2], self.mat_nums)
        self.assertEqual("".join(chr(x) for x in out["cdat"][3]), "text")
        self.assertEqual("".join(chr(x) for x in out["cdat"][4]), "longer text")
        self.assertEqual("".join(chr(x) for x in out["cdat"][5]), "\x00\x00")  # TODO: expect '' instead of [0, 0]
        np.testing.assert_array_equal(out["cdat"][6], np.zeros(2))  # TODO: expect [] instead of [0, 0]

    def test_nested(self) -> None:
        out = dcs.load_matlab(self.filename5, enums=self.enums)
        self.assertEqual(set(out.keys()), {"col_nums", "row_nums", "mat_nums", "x", "enum", "cdat", "data"})
        np.testing.assert_array_equal(out["row_nums"], self.row_nums_1d)
        np.testing.assert_array_equal(out["col_nums"], self.col_nums_1d)
        np.testing.assert_array_equal(out["mat_nums"], self.mat_nums)
        np.testing.assert_array_equal(out["x"]["r"], self.row_nums_1d)
        np.testing.assert_array_equal(out["x"]["c"], self.col_nums_1d)
        np.testing.assert_array_equal(out["x"]["m"], self.mat_nums)
        # np.testing.assert_array_equal(out['enum'], self.exp_enum) # TODO: fix this one along with the other case
        np.testing.assert_array_equal(out["cdat"][0], self.row_nums_1d)
        np.testing.assert_array_equal(out["cdat"][1], self.col_nums_1d)
        np.testing.assert_array_equal(out["cdat"][2], self.mat_nums)
        np.testing.assert_array_equal(out["data"]["x"]["r"], self.row_nums_1d)
        np.testing.assert_array_equal(out["data"]["x"]["c"], self.col_nums_1d)
        np.testing.assert_array_equal(out["data"]["x"]["m"], self.mat_nums)
        np.testing.assert_array_equal(out["data"]["y"]["r"], self.row_nums_1d + self.offsets["r"])
        np.testing.assert_array_equal(out["data"]["y"]["c"], self.col_nums_1d + self.offsets["c"])
        np.testing.assert_array_equal(out["data"]["y"]["m"], self.mat_nums + self.offsets["m"])
        np.testing.assert_array_equal(out["data"]["z"]["a"], np.array([1.0, 2.0, 3.0]))
        # np.testing.assert_array_equal(out['data']['z']['b'], self.exp_enum) # TODO: fix this one along with the other case
        np.testing.assert_array_equal(out["data"]["c"][0], self.row_nums_1d)
        np.testing.assert_array_equal(out["data"]["c"][1], self.col_nums_1d)
        np.testing.assert_array_equal(out["data"]["c"][2], self.mat_nums)
        np.testing.assert_array_equal(out["data"]["nc"][0], self.row_nums_1d)
        np.testing.assert_array_equal(out["data"]["nc"][1], self.col_nums_1d)
        np.testing.assert_array_equal(out["data"]["nc"][2]["r"], self.row_nums_1d)
        np.testing.assert_array_equal(out["data"]["nc"][2]["c"], self.col_nums_1d)
        np.testing.assert_array_equal(out["data"]["nc"][2]["m"], self.mat_nums)


# %% orth
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_orth(unittest.TestCase):
    r"""
    Tests the orth function with the following cases:
        Rank 3 matrix
        Rank 2 matrix
    """

    def setUp(self) -> None:
        self.A1 = np.array([[1, 0, 1], [-1, -2, 0], [0, 1, -1]])
        self.r1 = 3
        self.Q1 = np.array(
            [[-0.12000026, -0.80971228, 0.57442663], [0.90175265, 0.15312282, 0.40422217], [-0.41526149, 0.5664975, 0.71178541]]
        )
        self.A2 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.r2 = 2
        self.Q2 = np.array([[-0.70710678, 0.0,], [0.0, 1.0], [-0.70710678, 0.0]])  # fmt: skip

    def test_rank3(self) -> None:
        r = np.linalg.matrix_rank(self.A1)
        self.assertEqual(r, self.r1)
        Q = dcs.orth(self.A1)
        np.testing.assert_array_almost_equal(Q, self.Q1)

    def test_rank2(self) -> None:
        r = np.linalg.matrix_rank(self.A2)
        self.assertEqual(r, self.r2)
        Q = dcs.orth(self.A2)
        np.testing.assert_array_almost_equal(Q, self.Q2)


# %% subspace
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_subspace(unittest.TestCase):
    r"""
    Tests the subspace function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.A = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]])
        # fmt: off
        self.B = np.array([
            [ 1,  1,  1,  1],
            [ 1, -1,  1, -1],
            [ 1,  1, -1, -1],
            [ 1, -1, -1,  1],
            [-1, -1, -1, -1],
            [-1,  1, -1,  1],
            [-1, -1,  1,  1],
            [-1,  1,  1, -1],
        ])
        # fmt: on
        self.theta = np.pi / 2

    def test_nominal(self) -> None:
        theta = dcs.subspace(self.A, self.B)
        self.assertAlmostEqual(theta, self.theta)

    def test_swapped_rank(self) -> None:
        theta = dcs.subspace(self.B, self.A)
        self.assertAlmostEqual(theta, self.theta)


# %% mat_divide
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_mat_divide(unittest.TestCase):
    r"""
    Tests the mat_divide function with the following cases:
        Nominal
        Poorly-conditioned
    """

    def test_nominal(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        exp = np.array([1.0, -1.0])
        b = a @ exp
        x = dcs.mat_divide(a, b)
        np.testing.assert_array_almost_equal(x, exp, 14)

    def test_rcond(self) -> None:
        a = np.array([[1e6, 1e6], [1e6, 1e6 + 1e-8]])
        exp = np.array([1.0, -1.0])
        b = a @ exp
        x1 = dcs.mat_divide(a, b, rcond=1e-16)
        x2 = dcs.mat_divide(a, b, rcond=1e-6)
        np.testing.assert_array_almost_equal(x1, exp, 2)
        np.testing.assert_array_almost_equal(x2, np.zeros(2), 2)


# %% find_first
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_find_first(unittest.TestCase):
    r"""
    Tests the find_first function with the following cases:
        Nominal
        No trues
        2D
    """

    def test_nominal(self) -> None:
        x = np.array([0, 0, 1, 0, 1, 0, 0], dtype=bool)
        ix = dcs.find_first(x)
        self.assertEqual(ix, 2)

    def test_none(self) -> None:
        x = np.zeros(10, dtype=bool)
        ix = dcs.find_first(x)
        self.assertEqual(ix, -1)

    def test_2d(self) -> None:
        x = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
        ix = dcs.find_first(x)
        self.assertEqual(ix, 5)


# %% find_last
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_find_last(unittest.TestCase):
    r"""
    Tests the find_last function with the following cases:
        Nominal
        No trues
        2D
    """

    def test_nominal(self) -> None:
        x = np.array([0, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
        ix = dcs.find_last(x)
        self.assertEqual(ix, 5)

    def test_none(self) -> None:
        x = np.zeros(10, dtype=bool)
        ix = dcs.find_last(x)
        self.assertEqual(ix, -1)

    def test_2d(self) -> None:
        x = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
        ix = dcs.find_last(x)
        self.assertEqual(ix, 5)


# %% prepend
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_prepend(unittest.TestCase):
    r"""
    Tests the prepend function with the following cases:
        Integer
        Float
        Boolean
        Vector
    """

    def test_integer(self) -> None:
        vec = np.array([2, 3, 5])
        new = 1
        exp = np.array([1, 2, 3, 5])
        out = dcs.prepend(vec, new)
        np.testing.assert_array_equal(out, exp)

    def test_float(self) -> None:
        vec = np.array([2.2, -3.3, 5.0])
        new = -10.0
        exp = np.array([-10.0, 2.2, -3.3, 5.0])
        out = dcs.prepend(vec, new)
        np.testing.assert_array_almost_equal(out, exp, 14)

    def test_bool(self) -> None:
        vec = np.array([True, True, True, False], dtype=bool)
        new = False
        exp = np.array([False, True, True, True, False], dtype=bool)
        out = dcs.prepend(vec, new)
        np.testing.assert_array_equal(out, exp)

    def test_vec(self) -> None:
        vec = np.array([2, 3, 5])
        new = np.array([0, 1])
        exp = np.array([0, 1, 2, 3, 5])
        # TODO: should I allow this use case or throw it out within the function call?
        # or just consider it a typing exception?
        out = dcs.prepend(vec, new)  # type: ignore[call-overload]
        np.testing.assert_array_equal(out, exp)


# %% postpend
@unittest.skipIf(not dcs.HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_postpend(unittest.TestCase):
    r"""
    Tests the postpend function with the following cases:
        Integer
        Float
        Boolean
        Vector
    """

    def test_integer(self) -> None:
        vec = np.array([2, 3, 5])
        new = 10
        exp = np.array([2, 3, 5, 10])
        out = dcs.postpend(vec, new)
        np.testing.assert_array_equal(out, exp)

    def test_float(self) -> None:
        vec = np.array([2.2, -3.3, 5.0])
        new = -10.0
        exp = np.array([2.2, -3.3, 5.0, -10.0])
        out = dcs.postpend(vec, new)
        np.testing.assert_array_almost_equal(out, exp, 14)

    def test_bool(self) -> None:
        vec = np.array([True, True, True, False], dtype=bool)
        new = False
        exp = np.array([True, True, True, False, False], dtype=bool)
        out = dcs.postpend(vec, new)
        np.testing.assert_array_equal(out, exp)

    def test_vec(self) -> None:
        vec = np.array([2, 3, 5])
        new = np.array([0, 1])
        exp = np.array([2, 3, 5, 0, 1])
        # TODO: should I allow this use case or throw it out within the function call?
        # or just consider it a typing exception?
        out = dcs.postpend(vec, new)  # type: ignore[call-overload]
        np.testing.assert_array_equal(out, exp)


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
