r"""
Test file for the `latex` module of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""

# %% Imports
from __future__ import annotations

from typing import List, TYPE_CHECKING, Union
import unittest

import dstauffman as dcs

if dcs.HAVE_NUMPY:
    import numpy as np

    nan = np.nan
    inf = np.inf
else:
    from math import inf, nan

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _I = NDArray[np.int_]


# %% make_preamble
class Test_make_preamble(unittest.TestCase):
    r"""
    Tests the make_preamble function with the following cases:
        Nominal
        Different size
        Minipage
        Short caption
    """

    def setUp(self) -> None:
        self.caption = "This caption"
        self.label = "tab:this_label"
        self.cols = "lcc"

    def test_nominal(self) -> None:
        out = dcs.make_preamble(self.caption, self.label, self.cols)
        self.assertIn("    \\caption{This caption}%", out)
        self.assertIn("    \\label{tab:this_label}", out)
        self.assertIn("    \\begin{tabular}{lcc}", out)
        self.assertIn("    \\small", out)

    def test_size(self) -> None:
        out = dcs.make_preamble(self.caption, self.label, self.cols, size="\\footnotesize")
        self.assertIn("    \\footnotesize", out)
        self.assertNotIn("    \\small", out)

    def test_minipage(self) -> None:
        out = dcs.make_preamble(self.caption, self.label, self.cols, use_mini=True)
        self.assertIn("    \\begin{minipage}{\\linewidth}", out)
        self.assertIn("        \\begin{tabular}{lcc}", out)

    def test_short_cap(self) -> None:
        out = dcs.make_preamble(self.caption, self.label, self.cols, short_cap="Short cap")
        self.assertIn("    \\caption[Short cap]{This caption}%", out)
        self.assertNotIn("    \\caption{This caption}%", out)

    def test_numbered_false1(self) -> None:
        out = dcs.make_preamble(self.caption, self.label, self.cols, numbered=False)
        self.assertIn("    \\caption*{This caption}%", out)

    def test_numbered_false2(self) -> None:
        with self.assertRaises(AssertionError):
            dcs.make_preamble(self.caption, self.label, self.cols, short_cap="Short cap", numbered=False)


# %% make_conclusion
class Test_make_conclusion(unittest.TestCase):
    r"""
    Tests the make_conclusion function with the following cases:
        Nominal
        Minipage
    """

    def test_nominal(self) -> None:
        out = dcs.make_conclusion()
        self.assertEqual(out, ["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""])

    def test_minipage(self) -> None:
        out = dcs.make_conclusion(use_mini=True)
        self.assertEqual(out, ["            \\bottomrule", "        \\end{tabular}", "    \\end{minipage}", "\\end{table}", ""])


# %% bins_to_str_ranges
class Test_bins_to_str_ranges(unittest.TestCase):
    r"""
    Tests the bins_to_str_ranges function with the following cases:
        Nominal
        Different dt
        High cut-off
        Low cut-off
        Bad cut-off
        Single value ranges
        String passthrough
    """

    def setUp(self) -> None:
        self.bins: Union[_I, List[int]]
        if dcs.HAVE_NUMPY:
            self.bins = np.array([0, 20, 40, 60, 10000], dtype=int)
        else:
            self.bins = [0, 20, 40, 60, 10000]
        self.strs = ["0-19", "20-39", "40-59", "60+"]

    def test_nominal(self) -> None:
        out = dcs.bins_to_str_ranges(self.bins)
        self.assertEqual(out, self.strs)

    def test_dt(self) -> None:
        out = dcs.bins_to_str_ranges(self.bins, dt=0.1)
        self.assertEqual(out, ["0-19.9", "20-39.9", "40-59.9", "60+"])

    def test_no_cutoff(self) -> None:
        out = dcs.bins_to_str_ranges(self.bins, cutoff=1e6)
        self.assertEqual(out, ["0-19", "20-39", "40-59", "60-9999"])

    def test_no_cutoff2(self) -> None:
        out = dcs.bins_to_str_ranges([-10, 10, 30])
        self.assertEqual(out, ["-10-9", "10-29"])

    def test_bad_cutoff(self) -> None:
        out = dcs.bins_to_str_ranges(self.bins, cutoff=30)
        self.assertEqual(out, ["0-19", "20+", "40+", "60+"])

    def test_single_ranges(self) -> None:
        x: Union[_I, List[int]]
        if dcs.HAVE_NUMPY:
            x = np.array([0, 1, 5, 6, 10000], dtype=int)
        else:
            x = [0, 1, 5, 6, 10000]
        out = dcs.bins_to_str_ranges(x)
        self.assertEqual(out, ["0", "1-4", "5", "6+"])

    def test_str_passthrough(self) -> None:
        out = dcs.bins_to_str_ranges(["Urban", "Rural", "ignored"])
        self.assertEqual(out, ["Urban", "Rural"])


# %% latex_str
class Test_latex_str(unittest.TestCase):
    r"""
    Tests the latex_str function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.value = 101.666666666666
        self.value2 = 0.016528546178382508  # dcs.rate_to_prob(0.2/12)

    def test_string1(self) -> None:
        value_str = dcs.latex_str("test")
        self.assertEqual(value_str, "test")

    def test_string2(self) -> None:
        value_str = dcs.latex_str("N_O")
        self.assertEqual(value_str, r"N\_O")

    def test_int1(self) -> None:
        value_str = dcs.latex_str(2015)
        self.assertEqual(value_str, "2015")

    def test_int2(self) -> None:
        value_str = dcs.latex_str(2015, 0, fixed=True)
        self.assertEqual(value_str, "2015")

    def test_int3(self) -> None:
        value_str = dcs.latex_str(2015, 1, fixed=True)
        self.assertEqual(value_str, "2015.0")

    def test_digits_all(self) -> None:
        value_str = dcs.latex_str(self.value)
        self.assertEqual(value_str, "101.666666666666")

    def test_digits0(self) -> None:
        value_str = dcs.latex_str(self.value, 0)
        self.assertEqual(value_str, "1e+02")

    def test_digits1(self) -> None:
        value_str = dcs.latex_str(self.value, 1)
        self.assertEqual(value_str, "1e+02")

    def test_digits2(self) -> None:
        value_str = dcs.latex_str(self.value, 2)
        self.assertEqual(value_str, "1e+02")

    def test_digits3(self) -> None:
        value_str = dcs.latex_str(self.value, 3)
        self.assertEqual(value_str, "102")

    def test_digits4(self) -> None:
        value_str = dcs.latex_str(self.value, 4)
        self.assertEqual(value_str, "101.7")

    def test_fixed_digits0(self) -> None:
        value_str = dcs.latex_str(self.value, 0, fixed=True)
        self.assertEqual(value_str, "102")

    def test_fixed_digits1(self) -> None:
        value_str = dcs.latex_str(self.value, 1, fixed=True)
        self.assertEqual(value_str, "101.7")

    def test_fixed_digits2(self) -> None:
        value_str = dcs.latex_str(self.value, 2, fixed=True)
        self.assertEqual(value_str, "101.67")

    def test_fixed_digits3(self) -> None:
        value_str = dcs.latex_str(self.value, 3, fixed=True)
        self.assertEqual(value_str, "101.667")

    def test_cmp2ar1(self) -> None:
        value_str = dcs.latex_str(self.value2, 3, fixed=True)
        self.assertEqual(value_str, "0.017")

    def test_cmp2ar2(self) -> None:
        value_str = dcs.latex_str(self.value2, 3, fixed=False)
        self.assertEqual(value_str, "0.0165")

    def test_cmp2ar3(self) -> None:
        value_str = dcs.latex_str(self.value2, 3, fixed=True, cmp2ar=True)
        self.assertEqual(value_str, "0.200")

    def test_cmp2ar4(self) -> None:
        value_str = dcs.latex_str(self.value2, 3, fixed=False, cmp2ar=True)
        self.assertEqual(value_str, "0.2")

    def test_nan(self) -> None:
        value_str = dcs.latex_str(nan)
        self.assertEqual(value_str, "NaN")

    def test_infinity(self) -> None:
        value_str = dcs.latex_str(inf)
        self.assertEqual(value_str, r"$\infty$")


# %% Unit test execution
if __name__ == "__main__":
    unittest.main(exit=False)
