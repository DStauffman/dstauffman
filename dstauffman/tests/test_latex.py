# -*- coding: utf-8 -*-
r"""
Test file for the `bpe` module of the dstauffman code.  It is intented to contain test cases to
demonstrate functionality and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""

#%% Imports
import unittest

import numpy as np

import dstauffman as dcs

#%% make_preamble
class Test_make_preamble(unittest.TestCase):
    r"""
    Tests the make_preamble function with the following cases:
        Nominal
        Different size
        Minipage
        Short caption
    """
    def setUp(self):
        self.caption = 'This caption'
        self.label   = 'tab:this_label'
        self.cols    = 'lcc'

    def test_nominal(self):
        out = dcs.make_preamble(self.caption, self.label, self.cols)
        self.assertIn('    \\caption{This caption}%', out)
        self.assertIn('    \\label{tab:this_label}', out)
        self.assertIn('    \\begin{tabular}{lcc}', out)
        self.assertIn('    \\small', out)

    def test_size(self):
        out = dcs.make_preamble(self.caption, self.label, self.cols, size='\\footnotesize')
        self.assertIn('    \\footnotesize', out)
        self.assertNotIn('    \\small', out)

    def test_minipage(self):
        out = dcs.make_preamble(self.caption, self.label, self.cols, use_mini=True)
        self.assertIn('    \\begin{minipage}{\\linewidth}', out)
        self.assertIn('        \\begin{tabular}{lcc}', out)

    def test_short_cap(self):
        out = dcs.make_preamble(self.caption, self.label, self.cols, short_cap='Short cap')
        self.assertIn('    \caption[Short cap]{This caption}%', out)
        self.assertNotIn('    \caption{This caption}%', out)

#%% make_conclusion
class Test_make_conclusion(unittest.TestCase):
    r"""
    Tests the make_conclusion function with the following cases:
        Nominal
        Minipage
    """
    def test_nominal(self):
        out = dcs.make_conclusion()
        self.assertEqual(out, ['        \\bottomrule', '    \\end{tabular}', '\\end{table}', ''])

    def test_minipage(self):
        out = dcs.make_conclusion(use_mini=True)
        self.assertEqual(out, ['            \\bottomrule', '        \\end{tabular}', \
            '    \\end{minipage}', '\\end{table}', ''])

#%% bins_to_str_ranges
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
    def setUp(self):
        self.bins = np.array([0, 20, 40, 60, 10000], dtype=int)
        self.strs = ['0-19', '20-39', '40-59', '60+']

    def test_nominal(self):
        out = dcs.bins_to_str_ranges(self.bins)
        self.assertEqual(out, self.strs)

    def test_dt(self):
        out = dcs.bins_to_str_ranges(self.bins, dt=0.1)
        self.assertEqual(out, ['0-19.9', '20-39.9', '40-59.9', '60+'])

    def test_no_cutoff(self):
        out = dcs.bins_to_str_ranges(self.bins, cutoff=1e6)
        self.assertEqual(out, ['0-19', '20-39', '40-59', '60-9999'])

    def test_no_cutoff2(self):
        out = dcs.bins_to_str_ranges([-10, 10, 30])
        self.assertEqual(out, ['-10-9', '10-29'])

    def test_bad_cutoff(self):
        out = dcs.bins_to_str_ranges(self.bins, cutoff=30)
        self.assertEqual(out, ['0-19', '20+', '40+', '60+'])

    def test_single_ranges(self):
        out = dcs.bins_to_str_ranges(np.array([0, 1, 5, 6, 10000], dtype=int))
        self.assertEqual(out, ['0', '1-4', '5', '6+'])

    def test_str_passthrough(self):
        out = dcs.bins_to_str_ranges(['Urban', 'Rural', 'ignored'])
        self.assertEqual(out, ['Urban', 'Rural'])

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
