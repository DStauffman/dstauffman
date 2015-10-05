# -*- coding: utf-8 -*-
r"""
Test file for the `scoring` submodule of the dstauffman archery code.  It is intented to contain
test cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in October 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import unittest
import numpy as np
import dstauffman.archery.scoring as arch

#%% Classes for testing
# score_text_to_number
class Test_score_text_to_number(unittest.TestCase):
    r"""
    Tests the score_text_to_number function with the following cases:
        Text to num NFAA
        Text to num USAA
        Int to int NFAA
        Int to int USAA
        Large number
        Bad float
        Bad string (raises ValueError)
    """
    def setUp(self):
        self.text_scores = ['X', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0', 'M', 'x', 'm']
        self.num_scores  = [ 10,   10,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,   0,  10,   0]
        self.usaa_scores = [ 10,    9,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,   0,  10,   0]

    def test_conversion(self):
        for (this_text, this_num) in zip(self.text_scores, self.num_scores):
            num = arch.score_text_to_number(this_text)
            self.assertEqual(num, this_num)

    def test_usaa_conversion(self):
        for (this_text, this_num) in zip(self.text_scores, self.usaa_scores):
            num = arch.score_text_to_number(this_text, flag='usaa')
            self.assertEqual(num, this_num)

    def test_int_to_int(self):
        for this_num in self.num_scores:
            num = arch.score_text_to_number(this_num)
            self.assertEqual(num, this_num)

    def test_int_to_int_usaa(self):
        for this_num in range(0, 11):
            num = arch.score_text_to_number(this_num, flag='usaa')
            if this_num == 10:
                self.assertEqual(num, 9)
            else:
                self.assertEqual(num, this_num)

    def test_large_values(self):
        num = arch.score_text_to_number('1001')
        self.assertEqual(num, 1001)

    def test_bad_float(self):
        with self.assertRaises(ValueError):
            arch.score_text_to_number('10.8')

    def test_bad_value(self):
        with self.assertRaises(ValueError):
            arch.score_text_to_number('z')

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
