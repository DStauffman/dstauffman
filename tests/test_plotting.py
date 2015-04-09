# -*- coding: utf-8 -*-
r"""
Test file for the `plotting` module module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#. Written by David C. Stauffer in March 2015.
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
# storefig
class Test_storefig(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # create data
        self.time = np.arange(0, 10, 0.1)
        self.data = np.sin(self.time)
        self.title = 'Test Plot'
        self.folder = dcs.get_tests_dir()
        self.plot_type = ['png', 'jpg']
        # turn interaction off to make the plots draw all at once on a show() command
        plt.ioff()
        # create the figure and set the title
        self.fig = plt.figure()
        self.fig.canvas.set_window_title(self.title)
        # add an axis and plot the data
        ax = self.fig.add_subplot(111)
        ax.plot(self.time, self.data)
        # add labels and legends
        plt.xlabel('Time [year]')
        plt.ylabel('Value [units]')
        plt.title(self.title)
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
        # TODO:
        pass

    def test_bad_folder(self):
        # TODO:
        pass

    def test_bad_plot_type(self):
        # TODO:
        pass

    @classmethod
    def tearDownClass(self):
        plt.close(self.fig)

# plot_time_history
class Test_plot_time_history(unittest.TestCase):
    pass

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
