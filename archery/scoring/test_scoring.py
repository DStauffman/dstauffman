# -*- coding: utf-8 -*-
r"""
Test file for the `scoring` submodule of the dstauffman archery code.  It is intented to contain
test cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in October 2015.
"""

#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import unittest
import dstauffman.archery.scoring as arch

#%% get_root_dir
class Test_get_root_dir(unittest.TestCase):
    r"""
    Tests the get_root_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = arch.get_root_dir()
        self.assertTrue(folder) # TODO: don't know an independent way to test this

#%% score_text_to_number
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

#%% convert_data_to_scores
class Test_convert_data_to_scores(unittest.TestCase):
    r"""
    Tests the convert_data_to_scores function with these cases:
        Nominal
    """
    def setUp(self):
        self.scores = [10*['X', 10, 9], 10*[9, 9, 9]]
        self.nfaa_scores = [290, 270]
        self.usaa_scores = [280, 270]

    def test_nominal(self):
        (nfaa_score, usaa_score) = arch.convert_data_to_scores(self.scores)
        np.testing.assert_array_equal(nfaa_score, self.nfaa_scores)
        np.testing.assert_array_equal(usaa_score, self.usaa_scores)

#%% plot_mean_and_std
class Test_plot_mean_and_std(unittest.TestCase):
    r"""
    Tests the plot_mean_and_std function with these cases:
        TBD
    """
    def setUp(self):
        self.scores = [10*['X', 10, 9], 10*[9, 9, 9]]
        self.fig = None

    def test_nominal(self):
        self.fig = arch.plot_mean_and_std(self.scores)

    # TODO: write more of these

    def tearDown(self):
        if self.fig is not None:
            plt.close(self.fig)

#%% normal_curve
class Test_normal_curve(unittest.TestCase):
    r"""
    Tests the normal_curve function with these cases:
        TBD
    """
    def setUp(self):
        self.x = np.arange(-5, 5.01, 0.01)
        self.mu = 0
        self.sigma = 1
        self.y = np.exp(-self.x**2/2)/np.sqrt(2*np.pi)

    def test_nominal(self):
        y = arch.normal_curve(self.x, self.mu, self.sigma)
        np.testing.assert_array_almost_equal(y, self.y)

    def test_nonzero_mean(self):
        offset = 2.5
        y = arch.normal_curve(self.x + offset, self.mu + offset, self.sigma)
        np.testing.assert_array_almost_equal(y, self.y)

    def test_no_std(self):
        y = arch.normal_curve(self.x, 3.3, 0)
        out = np.zeros(self.x.shape)
        ix = np.nonzero(y == 3.3)[0]
        out[ix] = 1
        np.testing.assert_array_almost_equal(y, out)

#%% read_from_excel_datafile
class Test_read_from_excel_datafile(unittest.TestCase):
    r"""
    Tests the read_from_excel_datafile function with these cases:
        TBD
    """
    pass

#%% create_scoresheet
class Test_create_scoresheet(unittest.TestCase):
    r"""
    Tests the create_scoresheet function with these cases:
        TBD
    """
    pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
