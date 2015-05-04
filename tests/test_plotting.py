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

#%% Functions for testing
# plot_correlation_matrix
class Test_plot_correlation_matrix(unittest.TestCase):
    r"""
    Tests plot_correlation_matrix function with the following cases:
        normal mode
        non-square inputs
        default labels
        all arguments passed in
        symmetric matrix
        coloring with values above 1
        coloring with values below -1
        coloring with values in -1 to 1 instead of 0 to 1
        bad labels (should raise error)
    """
    def setUp(self):
        self.figs   = []
        self.data   = dcs.unit(np.random.rand(10, 10), axis=0)
        self.labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        self.opts   = dcs.Opts()
        self.opts.case_name = 'Testing Correlation'
        self.matrix_name    = 'Not a Correlation Matrix'

    def test_normal(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, self.labels))

    def test_nonsquare(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data[:5, :3], [self.labels[:3], \
            self.labels[:5]]))

    def test_default_labels(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data[:5, :3]))

    def test_all_args(self):
        self.figs.append(dcs.plot_correlation_matrix(self.data, self.labels, self.opts, \
            self.matrix_name))

    def test_symmetric(self):
        sym = self.data.copy()
        num = sym.shape[0]
        for j in range(num):
            for i in range(num):
                if i == j:
                    sym[i, j] = 1
                elif i > j:
                    sym[i, j] = self.data[j, i]
        self.figs.append(dcs.plot_correlation_matrix(sym))

    def test_above_one(self):
        large_data = self.data * 1000
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels))

    def test_below_one(self):
        large_data = 1000*(self.data - 0.5)
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels))

    def test_within_minus_one(self):
        large_data = self.data - 0.5
        self.figs.append(dcs.plot_correlation_matrix(large_data, self.labels))

    def test_bad_labels(self):
        with self.assertRaises(ValueError):
            self.figs.append(dcs.plot_correlation_matrix(self.data, ['a']))

    def tearDown(self):
        for i in range(len(self.figs)):
            plt.close(self.figs.pop())

# storefig
class Test_storefig(unittest.TestCase):
    r"""
    Tests the storefig function with the following cases:
        saving one plot to disk
        saving one plot to multiple plot types
        saving multiple plots to one plot type
        saving to a bad folder location (should raise error)
        specifying a bad plot type (should raise error)
    """
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
    r"""
    Tests the titleprefix function with the following cases:
        normal use
        null prefix
        multiple figures
    """
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
        plt.close()

    def tearDown(self):
        plt.close()

# setup_plots
class Test_setup_plots(unittest.TestCase):
    r"""
    Tests the setup_plots function with the following cases:
        Prepend a title
        Don't prepend a title
        Don't show the plot
        Multiple figures
        Save the plot
    """
    def setUp(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        plt.plot(x, y) # doctest: +ELLIPSIS
        plt.title('X vs Y') #doctest: +ELLIPSIS
        plt.xlabel('time [years]') #doctest: +SKIP
        plt.ylabel('value [radians]') #doctest: +SKIP
        plt.show(block=False)
        self.opts = dcs.Opts()
        self.opts.case_name = 'Testing'
        self.opts.show_plot = True
        self.opts.save_plot = False
        self.opts.save_path = dcs.get_tests_dir()

    def test_title(self):
        dcs.setup_plots(self.fig, self.opts)

    def test_no_title(self):
        self.opts.case_name = ''
        dcs.setup_plots(self.fig, self.opts)

    def test_not_showing_plot(self):
        self.opts.show_plot = False
        dcs.setup_plots(self.fig, self.opts)

    def test_multiple_figs(self):
        self.fig = [self.fig]
        new_fig = plt.figure()
        plt.plot(0, 0)
        self.fig.append(new_fig)
        dcs.setup_plots(self.fig, self.opts)

    def test_saving_plot(self):
        this_filename = os.path.join(dcs.get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        dcs.setup_plots(self.fig, self.opts)
        # remove file
        os.remove(this_filename)

    def tearDown(self):
        plt.close()

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
