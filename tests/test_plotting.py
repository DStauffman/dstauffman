# -*- coding: utf-8 -*-
r"""
Test file for the `plotting` module module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
import unittest
import dstauffman as dcs

#%% Classes for testing
# Opts
class Test_Opts(unittest.TestCase):
    r"""
    Test Opts class, and by extension the frozen function and Frozen class using cases:
        normal mode
        add new attribute to existing instance
    """
    def setUp(self):
        self.opts_fields = ['case_name']

    def test_calling(self):
        opts = dcs.Opts()
        for field in self.opts_fields:
            self.assertTrue(hasattr(opts, field))

    def test_new_attr(self):
        opts = dcs.Opts()
        with self.assertRaises(AttributeError):
            opts.new_field_that_does_not_exist = 1

# storefig
class Test_storefig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # create data
        cls.time = np.arange(0, 10, 0.1)
        cls.data = np.sin(cls.time)
        cls.title = 'Test Plot'
        cls.folder = dcs.get_tests_dir()
        cls.plot_type = ['png', 'jpg']
        # turn interaction off to make the plots draw all at once on a show() command
        plt.ioff()
        # create the figure and set the title
        cls.fig = plt.figure()
        cls.fig.canvas.set_window_title(cls.title)
        # add an axis and plot the data
        ax = cls.fig.add_subplot(111)
        ax.plot(cls.time, cls.data)
        # add labels and legends
        plt.xlabel('Time [year]')
        plt.ylabel('Value [units]')
        plt.title(cls.title)
        # show a grid
        plt.grid(True)
        # show the plot
        plt.show(block=False)

    def test_saving(self):
        dcs.storefig(self.fig, self.folder, self.plot_type[0])
        # assert that file exists
        this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type[0])
        self.assertTrue(os.path.isfile(this_filename))
        # remove file
        os.remove(this_filename)

    def test_multiple_plot_types(self):
        dcs.storefig(self.fig, self.folder, self.plot_type)
        # assert that files exist
        for this_type in self.plot_type:
            this_filename = os.path.join(self.folder, self.title + '.' + this_type)
            self.assertTrue(os.path.isfile(this_filename))
            # remove file
            os.remove(this_filename)

    def test_multiple_figures(self):
        dcs.storefig([self.fig, self.fig], self.folder, self.plot_type[0])
        # assert that file exists
        this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type[0])
        self.assertTrue(os.path.isfile(this_filename))
        # remove file
        os.remove(this_filename)

    def test_bad_folder(self):
        with self.assertRaises(ValueError):
            dcs.storefig(self.fig, 'ZZ:\\non_existant_path')
        # TODO:
        pass

    def test_bad_plot_type(self):
        # TODO:
        pass

    @classmethod
    def tearDownClass(self):
        plt.close(self.fig)

# titleprefix
class Test_titleprefix(unittest.TestCase):

    def setUp(self):
        self.fig = plt.figure()
        self.title = 'Figure Title'
        self.prefix = 'Prefix'
        self.fig.canvas.set_window_title(self.title)
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title('X vs Y')
        plt.show(block=False)

    def test_normal(self):
        dcs.titleprefix(self.fig, self.prefix)

    def test_null_prefix(self):
        dcs.titleprefix(self.fig)

    def test_multiple_figs(self):
        dcs.titleprefix([self.fig, self.fig], self.prefix)

    def tearDown(self):
        plt.close()

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
