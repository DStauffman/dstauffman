# -*- coding: utf-8 -*-
r"""
Test file for the `plot_support` module module of the "dstauffman" library.  It is intented to
contain test cases to demonstrate functionaliy and correct outcomes for all the functions within the
module.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
import os
import platform
import unittest

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

import dstauffman as dcs

#%% Plotter for testing
plotter = dcs.Plotter(False)

#%% Classes - Plotter
class Test_Plotter(unittest.TestCase):
    r"""
    Tests the Plotter class with the following cases:
        Get level
        Set level
        Bad level (raises ValueError)
        printing
    """
    def setUp(self):
        self.flag    = True
        self.plotter = dcs.Plotter(self.flag)
        self.print   = 'Plotter(True)'

    def test_get_plotter(self):
        flag = self.plotter.get_plotter()
        self.assertTrue(flag)

    def test_set_plotter(self):
        flag = self.plotter.get_plotter()
        self.assertTrue(flag)
        self.plotter.set_plotter(False)
        self.assertFalse(plotter.get_plotter())

    def test_printing(self):
        with dcs.capture_output() as out:
            print(self.plotter)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, self.print)

    def test_no_show(self):
        self.plotter = dcs.Plotter()

    def tearDown(self):
        self.plotter.set_plotter(False)

#%% Classes - TruthPlotter
class Test_TruthPlotter(unittest.TestCase):
    r"""
    Tests the TruthPlotter class with the following cases:
        TBD
    """
    def setUp(self):
        self.fig  = None
        self.x    = np.arange(0, 10, 0.1)
        self.y    = np.sin(self.x)
        self.data = np.vstack((self.y, self.y+0.01, self.y+0.03)).T

    def test_nominal(self):
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)
        np.testing.assert_array_almost_equal(self.y+0.03, truth.data_hi)

    def test_matrix1(self):
        truth = dcs.TruthPlotter(self.x, self.data[:, 1])
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        self.assertTrue(truth.data_lo is None)
        self.assertTrue(truth.data_hi is None)

    def test_matrix2(self):
        truth = dcs.TruthPlotter(self.x, self.data)
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)
        np.testing.assert_array_almost_equal(self.y+0.03, truth.data_hi)

    def test_matrix3(self):
        truth = dcs.TruthPlotter(self.x, self.data[:, np.array([1])])
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        self.assertTrue(truth.data_lo is None)
        self.assertTrue(truth.data_hi is None)

    def test_bad_matrix(self):
        with self.assertRaises(ValueError):
            dcs.TruthPlotter(self.x, np.random.rand(self.x.size, 4))

    def test_plotting0(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter()
        truth.plot_truth(ax)

    def test_plotting1(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        truth.plot_truth(ax)

    def test_plotting2(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_plotting3(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, None, lo=self.y, hi=self.y+0.03)
        truth.plot_truth(ax)

    def test_plotting4(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=None, hi=self.y+0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_plotting5(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='errorbar')
        # fake out data (can't be done through __init__ right now, this might need work)
        truth.data = self.data
        truth.data_lo = self.data - 0.01
        truth.data_hi = self.data + 0.01
        truth.plot_truth(ax, ix=1)

    def test_dont_hold_limits(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x-10, self.y, lo=self.y-1000, hi=self.y+1000, type_='errorbar')
        truth.plot_truth(ax, hold_xlim=False, hold_ylim=False)

    def test_hold_limits(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x-10, self.y, lo=self.y-1000, hi=self.y+1000, type_='errorbar')
        truth.plot_truth(ax, hold_xlim=True, hold_ylim=True)

    def test_bad_type(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='bad type')
        with self.assertRaises(ValueError):
            truth.plot_truth(ax)

    def test_pprint(self):
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        with dcs.capture_output() as out:
            truth.pprint()
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'TruthPlotter')
        self.assertTrue(lines[1].startswith(' time    = ['))
        self.assertEqual(lines[-1], ' name    = Observed')

    def test_is_null(self):
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        self.assertFalse(truth.is_null)

    def test_get_data1(self):
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        data = truth.get_data(self.data, scale=2)
        np.testing.assert_array_almost_equal(data, 2*self.data)

    def test_get_data2(self):
        truth = dcs.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        data = truth.get_data(self.data, scale=3, ix=0)
        np.testing.assert_array_almost_equal(data, 3*self.data[:, 0])

    def tearDown(self):
        if self.fig is not None:
            plt.close(self.fig)

#%% Classes - MyCustomToolbar
class Test_MyCustomToolbar(unittest.TestCase):
    r"""
    Tests the MyCustomToolbar class with the following cases:
        No nothing
        Next plot
        Prev plot
        Close all
        Multiple nexts
        Multiple prevs
    """
    def setUp(self):
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        self.fig1.toolbar_custom_ = dcs.MyCustomToolbar(self.fig1)
        self.fig2.toolbar_custom_ = dcs.MyCustomToolbar(self.fig2)

    def test_do_nothing(self):
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_next_plot(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)

    def test_prev_plot(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)

    def test_close_all(self):
        self.assertTrue(plt.fignum_exists(self.fig1.number))
        self.assertTrue(plt.fignum_exists(self.fig2.number))
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_close_all, Qt.LeftButton)
        self.assertFalse(plt.fignum_exists(self.fig1.number))
        self.assertFalse(plt.fignum_exists(self.fig2.number))

    def test_multiple_nexts(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_multiple_prevs(self):
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_enter_and_exit_events(self):
        # TODO: this is not causing what I want to happen, the style changes aren't happening.
        QTest.mouseMove(self.fig1.toolbar_custom_.btn_next_plot, delay=100)
        QTest.mouseMove(self.fig1.toolbar_custom_.btn_prev_plot, delay=100)

    def tearDown(self):
        plt.close(self.fig1)
        plt.close(self.fig2)

#%% Classes - ColorMap
class Test_ColorMap(unittest.TestCase):
    r"""
    Tests ColorMap class with the following cases:
        Nominal mode
        Using num_colors specifier
        No inputs
        get_color method
        get_smap method
        set_colors method
    """
    def setUp(self):
        self.colormap   = 'Paired'
        self.low        = 0
        self.high       = 1
        self.num_colors = 5
        self.fig        = None

    def test_nominal(self):
        cm = dcs.ColorMap(self.colormap, self.low, self.high)
        self.assertTrue(cm.get_color(self.low))

    def test_num_colors(self):
        cm = dcs.ColorMap(self.colormap, num_colors=self.num_colors)
        self.assertTrue(cm.get_color(self.low))

    def test_no_inputs(self):
        cm = dcs.ColorMap()
        self.assertTrue(cm.get_color(self.low))

    def test_get_color(self):
        cm = dcs.ColorMap(num_colors=self.num_colors)
        for i in range(self.num_colors):
            self.assertTrue(cm.get_color(i))

    def test_get_smap(self):
        cm = dcs.ColorMap()
        smap = cm.get_smap()
        self.assertTrue(isinstance(smap, cmx.ScalarMappable))

    def test_set_colors(self):
        cm = dcs.ColorMap(num_colors=5)
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        cm.set_colors(ax)
        ax.plot(0, 0)
        self.assertTrue(True)

    def test_set_color_failure(self):
        cm = dcs.ColorMap()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.plot(0, 0)
        with self.assertRaises(ValueError):
            cm.set_colors(ax)

    def test_existing_colormap(self):
        colormap = colors.ListedColormap(['r', 'g', 'b'])
        cm = dcs.ColorMap(colormap, num_colors=3)
        red = cm.get_color(0)
        green = cm.get_color(1)
        blue = cm.get_color(2)
        self.assertEqual(red, (1., 0, 0, 1))
        self.assertEqual(green, (0., 0.5, 0, 1))
        self.assertEqual(blue, (0., 0, 1, 1))

    def tearDown(self):
        if self.fig is not None:
            plt.close(self.fig)

#%% Functions - close_all
class Test_close_all(unittest.TestCase):
    r"""
    Tests the close_all function with the following cases:
        Nominal
        Specified list
    """
    def test_nominal(self):
        fig1 = plt.figure()
        fig2 = plt.figure()
        self.assertTrue(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        dcs.close_all()
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertFalse(plt.fignum_exists(fig2.number))

    def test_list(self):
        fig1 = plt.figure()
        fig2 = plt.figure()
        self.assertTrue(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        dcs.close_all([fig1])
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        plt.close(fig2)
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertFalse(plt.fignum_exists(fig2.number))

#%% Functions - get_color_lists
class Test_get_color_lists(unittest.TestCase):
    r"""
    Tests the get_color_lists function with the following cases:
        Nominal
    """
    def setUp(self):
        self.keys = ['default', 'single', 'double', 'vec', 'quat', 'dbl_diff', 'vec_diff', 'quat_diff']

    def test_nominal(self):
        color_lists = dcs.get_color_lists()
        for key in self.keys:
            self.assertIn(key, color_lists)
        colormap = color_lists['quat_diff']
        self.assertEqual(colormap.N, 8)
        self.assertEqual(colormap.colors[0], 'xkcd:red')
        self.assertEqual(colormap.colors[7], 'xkcd:brown')

#%% Functions - ignore_plot_data
class Test_ignore_plot_data(unittest.TestCase):
    r"""
    Tests the ignore_plot_data function with the following cases:
        None
        Not ignoring
        Ignoring full data
        Ignoring specific columns
    """
    def setUp(self):
        self.data = np.zeros((3, 10), dtype=float)
        self.ignore_empties = True

    def test_none(self):
        ignore = dcs.ignore_plot_data(None, True)
        self.assertTrue(ignore)
        ignore = dcs.ignore_plot_data(None, False)
        self.assertTrue(ignore)
        ignore = dcs.ignore_plot_data(None, False, 0)
        self.assertTrue(ignore)
        ignore = dcs.ignore_plot_data(None, True, 0)
        self.assertTrue(ignore)

    def test_not_ignoring(self):
        ignore = dcs.ignore_plot_data(self.data, False)
        self.assertFalse(ignore)
        ignore = dcs.ignore_plot_data(self.data, False, 0)
        self.assertFalse(ignore)

    def test_ignoring_no_col(self):
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties)
        self.assertTrue(ignore)
        self.data[1, 2] = np.nan
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties)
        self.assertTrue(ignore)
        self.data[2, 5] = 0.1
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties)
        self.assertFalse(ignore)

    def test_ignoring_col(self):
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties, 1)
        self.assertTrue(ignore)
        self.data[1, 2] = np.nan
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties, 0)
        self.assertTrue(ignore)
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties, 1)
        self.assertTrue(ignore)
        self.data[2, 5] = 0.1
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties, 0)
        self.assertTrue(ignore)
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties, 1)
        self.assertTrue(ignore)
        ignore = dcs.ignore_plot_data(self.data, self.ignore_empties, 5)
        self.assertFalse(ignore)

#%% Functions - whitten
class Test_whitten(unittest.TestCase):
    r"""
    Tests the whitten function with the following cases:
        Nominal
    """
    def setUp(self):
        self.color = (1, 0.4, 0)
        self.whittened_color = (1.0, 0.58, 0.3)

    def test_nominal(self):
        new_color = dcs.whitten(self.color)
        self.assertEqual(new_color, self.whittened_color)

#%% Functions - storefig
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
        cls.plot_type = 'png'
        # create the figure and set the title
        cls.fig = plt.figure()
        cls.fig.canvas.set_window_title(cls.title)
        # add an axis and plot the data
        ax = cls.fig.add_subplot(111)
        ax.plot(cls.time, cls.data)
        # add labels and legends
        ax.set_xlabel('Time [year]')
        ax.set_ylabel('Value [units]')
        ax.set_title(cls.title)
        # show a grid
        ax.grid(True)

    def test_saving(self):
        dcs.storefig(self.fig, self.folder, self.plot_type)
        # assert that file exists
        this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type)
        self.assertTrue(os.path.isfile(this_filename))
        # remove file
        os.remove(this_filename)

    def test_multiple_plot_types(self):
        plot_types = ['png', 'svg']
        dcs.storefig(self.fig, self.folder, plot_types)
        # assert that files exist
        for this_type in plot_types:
            this_filename = os.path.join(self.folder, self.title + '.' + this_type)
            self.assertTrue(os.path.isfile(this_filename))
            # remove file
            os.remove(this_filename)

    def test_save_as_jpg(self):
        # Note: this test case can fail if PIL is not installed, try "pip install Pillow"
        dcs.storefig(self.fig, self.folder, 'jpg')
        # assert that files exist
        this_filename = os.path.join(self.folder, self.title + '.jpg')
        self.assertTrue(os.path.isfile(this_filename))
        # remove file
        os.remove(this_filename)

    def test_multiple_figures(self):
        dcs.storefig([self.fig, self.fig], self.folder, self.plot_type)
        # assert that file exists
        this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type)
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

    def test_bad_characters(self):
        # change to bad name
        self.fig.canvas.set_window_title('Bad < > / names')
        # save file
        dcs.storefig(self.fig, self.folder, self.plot_type)
        # restore filename
        self.fig.canvas.set_window_title(self.title)
        # assert that file exists
        if platform.system() == 'Windows':
            this_filename = os.path.join(self.folder, 'Bad _ _ _ names' + '.' + self.plot_type)
        else:
            this_filename = os.path.join(self.folder, 'Bad < > _ names' + '.' + self.plot_type)
        self.assertTrue(os.path.isfile(this_filename))
        # remove file
        os.remove(this_filename)

    @classmethod
    def tearDownClass(cls):
        plt.close(cls.fig)

#%% Functions - titleprefix
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
        ax = self.fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title('X vs Y')

    def test_normal(self):
        dcs.titleprefix(self.fig, self.prefix)

    def test_null_prefix(self):
        dcs.titleprefix(self.fig)

    def test_multiple_figs(self):
        dcs.titleprefix([self.fig, self.fig], self.prefix)

    def tearDown(self):
        plt.close(self.fig)

#%% Functions - disp_xlimits
class Test_disp_xlimits(unittest.TestCase):
    r"""
    Tests the disp_xlimits function with the following cases:
        Normal use
        Null action
        Only xmin
        Only xmax
        Multiple figures
    """
    def setUp(self):
        self.fig = plt.figure()
        self.xmin = 2
        self.xmax = 5
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        ax = self.fig.add_subplot(111)
        ax.plot(x, y)

    def test_normal(self):
        dcs.disp_xlimits(self.fig, self.xmin, self.xmax)

    def test_null_action(self):
        dcs.disp_xlimits(self.fig)

    def test_just_xmin(self):
        dcs.disp_xlimits(self.fig, xmin=self.xmin)

    def test_just_xmax(self):
        dcs.disp_xlimits(self.fig, xmax=self.xmax)

    def test_multiple_figs(self):
        dcs.disp_xlimits([self.fig, self.fig], self.xmin, self.xmax)

    def tearDown(self):
        plt.close(self.fig)

#%% Functions - setup_plots
class Test_setup_plots(unittest.TestCase):
    r"""
    Tests the setup_plots function with the following cases:
        Prepend a title
        Don't prepend a title
        Don't show the plot
        Multiple figures
        Save the plot
        Show the plot link
    """
    def setUp(self):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('X vs Y')
        ax.set_xlabel('time [years]')
        ax.set_ylabel('value [radians]')
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
        fig_list = [self.fig]
        (new_fig, ax) = plt.subplots()
        ax.plot(0, 0)
        fig_list.append(new_fig)
        dcs.setup_plots(fig_list, self.opts)
        plt.close(new_fig)

    def test_saving_plot(self):
        this_filename = os.path.join(dcs.get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        dcs.setup_plots(self.fig, self.opts)
        # remove file
        os.remove(this_filename)

    def test_show_link(self):
        this_filename = os.path.join(dcs.get_tests_dir(), self.opts.case_name + ' - Figure Title.png')
        self.opts.save_plot = True
        self.opts.show_link = True
        with dcs.capture_output() as out:
            dcs.setup_plots(self.fig, self.opts)
        output = out.getvalue().strip()
        out.close()
        # remove file
        os.remove(this_filename)
        self.assertTrue(output.startswith('Plots saved to <a href="'))

    def tearDown(self):
        plt.close(self.fig)

#%% Functions - figmenu
class Test_figmenu(unittest.TestCase):
    r"""
    Tests the figmenu function with the following cases:
        One input
        List input
    """
    def setUp(self):
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()

    def test_one_input(self):
        dcs.figmenu(self.fig1)

    def test_list_input(self):
        dcs.figmenu([self.fig1, self.fig2])

    def tearDown(self):
        plt.close(self.fig1)
        plt.close(self.fig2)

#%% Functions - rgb_ints_to_hex
class Test_rgb_ints_to_hex(unittest.TestCase):
    r"""
    Tests the rgb_ints_to_hex function with the following cases:
        Nominal
        Out of range
    """
    def setUp(self):
        self.colors    = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (79, 129, 189), (0, 0, 0)]
        self.hex_codes = ['#ff0000', '#00ff00', '#0000ff', '#4f81bd', '#000000']

    def test_nominal(self):
        for (ix, this_color) in enumerate(self.colors):
            hex_codes = dcs.rgb_ints_to_hex(this_color)
            self.assertEqual(hex_codes, self.hex_codes[ix])

    def test_bad_range(self):
        hex_code = dcs.rgb_ints_to_hex((-100, 500, 9))
        self.assertEqual(hex_code, '#00ff09')

#%% Functions - get_screen_resolution
class Test_get_screen_resolution(unittest.TestCase):
    r"""
    Tests the get_screen_resolution function with the following cases:
        Nominal
    """
    def test_nominal(self):
        (screen_width, screen_height) = dcs.get_screen_resolution()
        self.assertGreater(screen_width, 0)
        self.assertGreater(screen_height, 0)

#%% Functions - show_zero_ylim
class Test_show_zero_ylim(unittest.TestCase):
    r"""
    Tests the show_zero_ylim function with the following cases:
        TBD
    """
    def setUp(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def test_no_change(self):
        self.ax.plot([1, 5, 10], [200, -150, 240], '.-')
        dcs.show_zero_ylim(self.ax)

    def test_all_positive(self):
        self.ax.plot([1, 5, 10], [200, 250, 240], '.-')
        dcs.show_zero_ylim(self.ax)

    def test_all_negative(self):
        self.ax.plot([1, 5, 10], [-200, -250, -240], '.-')
        dcs.show_zero_ylim(self.ax)

    def tearDown(self):
        plt.close(self.fig)

#%% Functions - plot_second_yunits
class Test_plot_second_yunits(unittest.TestCase):
    r"""
    Tests the plot_second_yunits function with the following cases:
        TBD
    """
    def setUp(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], '.-')
        self.ax.set_ylabel('Value [rad]')
        self.ylab = u'Value [µrad]'
        self.multiplier = 1e6

    def test_nominal(self):
        dcs.plot_second_yunits(self.ax, self.ylab, self.multiplier)

    def tearDown(self):
        plt.close(self.fig)

#%% Functions - plot_rms_lines
class Test_plot_rms_lines(unittest.TestCase):
    r"""
    Tests the plot_rms_lines function with the following cases:
        Nominal
    """
    def setUp(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(np.arange(10), np.arange(10), label='Data')
        self.x = (2, 5)
        self.y = (1, 10)

    def test_nominal(self):
        dcs.plot_rms_lines(self.ax, self.x, self.y, show_in_legend=False)
        self.ax.legend()

    def tearDown(self):
        plt.close(self.fig)

#%% Functions - plot_classification
class Test_plot_classification(unittest.TestCase):
    r"""
    Tests the plot_classification function with the following cases:
        Inside axes
        Outside axes
        Classified options with test banner
        Bad option (should error)
    """
    def setUp(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([0, 10], [0, 10], '.-b')

    def test_inside(self):
        dcs.plot_classification(self.ax, 'U', test=False, inside_axes=True)

    def test_outside(self):
        dcs.plot_classification(self.ax, 'U', test=False, inside_axes=False)

    def test_options(self):
        for opt in {'C', 'S', 'T', 'TS', 'F', 'FOUO', 'U//FOUO' ,'NF' ,'S//NF'}:
            dcs.plot_classification(self.ax, opt, test=True, inside_axes=False)

    def test_bad_option(self):
        with self.assertRaises(ValueError):
            dcs.plot_classification(self.ax, 'BadOption')

    def tearDown(self):
        plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    plt.ioff()
    unittest.main(exit=False)
