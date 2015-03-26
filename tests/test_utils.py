# -*- coding: utf-8 -*-
r"""
Test file for the `utils` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#. Written by David C. Stauffer in March 2015.
"""

#%% Imoprts
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import unittest
import dstauffman as dcs

#%% Classes for testing
# rms
class Test_rms(unittest.TestCase):

    def setUp(self):
        self.inputs1   = np.array([0, 1, 0., -1])
        self.outputs1  = np.sqrt(2)/2
        self.inputs2   = [[0, 1, 0., -1], [1., 1, 1, 1]]
        self.outputs2  = np.matrix([[np.sqrt(2)/2], [1]])

    def test_rms_series(self):
        out = dcs.rms(self.inputs1)
        self.assertAlmostEqual(self.outputs1,out)

    def test_scalar_input(self):
        out = dcs.rms(-1.5)
        self.assertEqual(1.5,out)

    def test_axis_shrink(self):
        out = dcs.rms(self.inputs1, axis=0)
        self.assertAlmostEqual(self.outputs1,out)
        pass

    def test_axis_keep(self):
        out = dcs.rms(self.inputs2, axis=1, keepdims=True)
        for i in range(0,len(out)):
            for j in range(0,len(out[i])):
                self.assertAlmostEqual(self.outputs2[i,j],out[i,j])

    def test_complex_rms(self):
        out = dcs.rms(1.5j)
        self.assertEqual(np.complex(1.5,0),out)

    def test_complex_conj(self):
        out = dcs.rms(np.array([1+1j, 1-1j]))
        self.assertAlmostEqual(np.sqrt(2.),out)

# convert_annual_to_monthly_probability
class Test_convert_annual_to_monthly_probability(unittest.TestCase):

    def setUp(self):
        self.monthly = np.arange(10)/1000.
        self.annuals = self.monthly
        for i in range(1,12):
            self.annuals = 1 - (1 - self.annuals) * (1 - self.monthly)

    def test_conversion(self):
        monthly = dcs.convert_annual_to_monthly_probability(self.annuals)
        for i in range(0,len(self.monthly)):
            self.assertAlmostEqual(self.monthly[i],monthly[i])

    def test_scalar(self):
        monthly = dcs.convert_annual_to_monthly_probability(0)
        self.assertIn(monthly,self.monthly)

    def test_lt_zero(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, -1.]))

    def test_gt_one(self):
        with self.assertRaises(ValueError):
            dcs.convert_annual_to_monthly_probability(np.array([0., 0.5, 1.5]))

# read_text_file
class Test_read_text_file(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.folder   = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.contents = 'Hello, World!\n'
        self.filepath = os.path.join(self.folder,'temp_file.txt')
        self.badpath  = r'AA:\non_existent_path\bad_file.txt'
        with open(self.filepath,'wt') as file:
            file.write(self.contents)

    def test_reading(self):
        text = dcs.read_text_file(self.filepath)
        self.assertEqual(self.contents,text)

    def test_bad_reading(self):
        with self.assertRaises(OSError):
            dcs.read_text_file(self.badpath)

    @classmethod
    def tearDownClase(self):
        os.remove(self.filepath)

# write_text_file
class Test_write_text_file(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.folder   = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.contents = 'Hello, World!\n'
        self.filepath = os.path.join(self.folder,'temp_file.txt')
        self.badpath  = r'AA:\non_existent_path\bad_file.txt'

    def test_writing(self):
        dcs.write_text_file(self.filepath,self.contents)
        with open(self.filepath, 'rt') as file:
            text = file.read()
        self.assertEqual(self.contents,text)

    def test_bad_writing(self):
        pass
        with self.assertRaises(OSError):
            dcs.write_text_file(self.badpath,self.contents)

    @classmethod
    def tearDownClass(self):
        os.remove(self.filepath)

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
