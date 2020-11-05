r"""
Test file for the `support` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
from __future__ import annotations
import contextlib
import datetime
import os
import platform
from typing import Dict, Optional, Union
import unittest

from dstauffman import capture_output, get_tests_dir, HAVE_MPL, HAVE_NUMPY, HAVE_SCIPY, IS_WINDOWS
import dstauffman.plotting as plot

if HAVE_MPL:
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    _HAVE_QT = True
except ModuleNotFoundError:
    _HAVE_QT = False

#%% Plotter for testing
plotter = plot.Plotter(False)

#%% plotting.DEFAULT_COLORMAP
class Test_plotting_DEFAULT_COLORMAP(unittest.TestCase):
    r"""
    Tests the plotting.DEFAULT_COLORMAP constant with the following cases:
        Exists
    """
    def test_exists(self) -> None:
        self.assertTrue(isinstance(plot.DEFAULT_COLORMAP, str))

#%% plotting.DEFAULT_CLASSIFICATION
class Test_plotting_DEFAULT_CLASSIFICATION(unittest.TestCase):
    r"""
    Tests the plotting.DEFAULT_CLASSIFICATION constant with the following cases:
        Exists
    """
    def test_exists(self) -> None:
        self.assertTrue(isinstance(plot.DEFAULT_CLASSIFICATION, str))

#%% plotting._HoverButton
class Test_plotting__HoverButton(unittest.TestCase):
    r"""
    Tests the plotting._HoverButton class with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.TruthPlotter
@unittest.skipIf(not HAVE_MPL or not HAVE_NUMPY, 'Skipping due to missing matplotlib/numpy dependency.')
class Test_plotting_TruthPlotter(unittest.TestCase):
    r"""
    Tests the plotting.TruthPlotter class with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.fig: Optional[plt.Figure] = None
        self.x    = np.arange(0, 10, 0.1)
        self.y    = np.sin(self.x)
        self.data = np.vstack((self.y, self.y+0.01, self.y+0.03)).T

    def test_nominal(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)
        np.testing.assert_array_almost_equal(self.y+0.03, truth.data_hi)

    def test_matrix1(self) -> None:
        truth = plot.TruthPlotter(self.x, self.data[:, 1])
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        self.assertTrue(truth.data_lo is None)
        self.assertTrue(truth.data_hi is None)

    def test_matrix2(self) -> None:
        truth = plot.TruthPlotter(self.x, self.data)
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        np.testing.assert_array_almost_equal(self.y, truth.data_lo)
        np.testing.assert_array_almost_equal(self.y+0.03, truth.data_hi)

    def test_matrix3(self) -> None:
        truth = plot.TruthPlotter(self.x, self.data[:, np.array([1])])
        np.testing.assert_array_almost_equal(self.y+0.01, truth.data)
        self.assertTrue(truth.data_lo is None)
        self.assertTrue(truth.data_hi is None)

    def test_bad_matrix(self) -> None:
        with self.assertRaises(ValueError):
            plot.TruthPlotter(self.x, np.random.rand(self.x.size, 4))

    def test_plotting0(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter()
        truth.plot_truth(ax)

    def test_plotting1(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        truth.plot_truth(ax)

    def test_plotting2(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_plotting3(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, None, lo=self.y, hi=self.y+0.03)
        truth.plot_truth(ax)

    def test_plotting4(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=None, hi=self.y+0.03, type_='errorbar')
        truth.plot_truth(ax)

    def test_plotting5(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='errorbar')
        # fake out data (can't be done through __init__ right now, this might need work)
        truth.data = self.data
        truth.data_lo = self.data - 0.01
        truth.data_hi = self.data + 0.01
        truth.plot_truth(ax, ix=1)

    def test_dont_hold_limits(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x-10, self.y, lo=self.y-1000, hi=self.y+1000, type_='errorbar')
        truth.plot_truth(ax, hold_xlim=False, hold_ylim=False)

    def test_hold_limits(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x-10, self.y, lo=self.y-1000, hi=self.y+1000, type_='errorbar')
        truth.plot_truth(ax, hold_xlim=True, hold_ylim=True)

    def test_bad_type(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.y, label='data')
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03, type_='bad type')
        with self.assertRaises(ValueError):
            truth.plot_truth(ax)

    def test_pprint(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        with capture_output() as out:
            truth.pprint()
        lines = out.getvalue().strip().split('\n')
        out.close()
        self.assertEqual(lines[0], 'TruthPlotter')
        self.assertTrue(lines[1].startswith(' time    = ['))
        self.assertEqual(lines[-1], ' name    = Observed')

    def test_is_null(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        self.assertFalse(truth.is_null)

    def test_get_data1(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        data = truth.get_data(self.data, scale=2)
        np.testing.assert_array_almost_equal(data, 2*self.data)

    def test_get_data2(self) -> None:
        truth = plot.TruthPlotter(self.x, self.y+0.01, lo=self.y, hi=self.y+0.03)
        data = truth.get_data(self.data, scale=3, ix=0)
        np.testing.assert_array_almost_equal(data, 3*self.data[:, 0])

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)

#%% plotting.MyCustomToolbar
@unittest.skipIf(not HAVE_MPL or not _HAVE_QT, 'Skipping due to missing matplotlib/PyQt5 dependency.')
class Test_plotting_MyCustomToolbar(unittest.TestCase):
    r"""
    Tests the plotting.MyCustomToolbar class with the following cases:
        No nothing
        Next plot
        Prev plot
        Close all
        Multiple nexts
        Multiple prevs
    """
    def setUp(self) -> None:
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        self.fig1.toolbar_custom_ = plot.MyCustomToolbar(self.fig1)
        self.fig2.toolbar_custom_ = plot.MyCustomToolbar(self.fig2)

    def test_do_nothing(self) -> None:
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_next_plot(self) -> None:
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)

    def test_prev_plot(self) -> None:
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)

    def test_close_all(self) -> None:
        self.assertTrue(plt.fignum_exists(self.fig1.number))
        self.assertTrue(plt.fignum_exists(self.fig2.number))
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_close_all, Qt.LeftButton)
        self.assertFalse(plt.fignum_exists(self.fig1.number))
        self.assertFalse(plt.fignum_exists(self.fig2.number))

    def test_multiple_nexts(self) -> None:
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_next_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_multiple_prevs(self) -> None:
        QTest.mouseClick(self.fig2.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig1.number)
        QTest.mouseClick(self.fig1.toolbar_custom_.btn_prev_plot, Qt.LeftButton)
        self.assertEqual(plt.gcf().number, self.fig2.number)

    def test_enter_and_exit_events(self) -> None:
        # TODO: this is not causing what I want to happen, the style changes aren't happening.
        QTest.mouseMove(self.fig1.toolbar_custom_.btn_next_plot, delay=100)
        QTest.mouseMove(self.fig1.toolbar_custom_.btn_prev_plot, delay=100)

    def tearDown(self) -> None:
        plt.close(self.fig1)
        plt.close(self.fig2)

#%% plotting.ColorMap
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_ColorMap(unittest.TestCase):
    r"""
    Tests the plotting.ColorMap class with the following cases:
        Nominal mode
        Using num_colors specifier
        No inputs
        get_color method
        get_smap method
        set_colors method
    """
    def setUp(self) -> None:
        self.colormap   = 'Paired'
        self.low        = 0
        self.high       = 1
        self.num_colors = 5
        self.fig: Optional[plt.Figure] = None

    def test_nominal(self) -> None:
        cm = plot.ColorMap(self.colormap, self.low, self.high)
        self.assertTrue(cm.get_color(self.low))

    def test_num_colors(self) -> None:
        cm = plot.ColorMap(self.colormap, num_colors=self.num_colors)
        self.assertTrue(cm.get_color(self.low))

    def test_no_inputs(self) -> None:
        cm = plot.ColorMap()
        self.assertTrue(cm.get_color(self.low))

    def test_get_color(self) -> None:
        cm = plot.ColorMap(num_colors=self.num_colors)
        for i in range(self.num_colors):
            self.assertTrue(cm.get_color(i))

    def test_get_smap(self) -> None:
        cm = plot.ColorMap()
        smap = cm.get_smap()
        self.assertTrue(isinstance(smap, cmx.ScalarMappable))

    def test_set_colors(self) -> None:
        cm = plot.ColorMap(num_colors=5)
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        cm.set_colors(ax)
        ax.plot(0, 0)
        self.assertTrue(True)

    def test_set_color_failure(self) -> None:
        cm = plot.ColorMap()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.plot(0, 0)
        with self.assertRaises(ValueError):
            cm.set_colors(ax)

    def test_existing_colormap(self) -> None:
        colormap = colors.ListedColormap(['r', 'g', 'b'])
        cm = plot.ColorMap(colormap, num_colors=3)
        red = cm.get_color(0)
        green = cm.get_color(1)
        blue = cm.get_color(2)
        self.assertEqual(red, (1., 0, 0, 1))
        self.assertEqual(green, (0., 0.5, 0, 1))
        self.assertEqual(blue, (0., 0, 1, 1))

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)

#%% plotting.close_all
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_close_all(unittest.TestCase):
    r"""
    Tests the plotting.close_all function with the following cases:
        Nominal
        Specified list
    """
    def test_nominal(self) -> None:
        fig1 = plt.figure()
        fig2 = plt.figure()
        self.assertTrue(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        plot.close_all()
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertFalse(plt.fignum_exists(fig2.number))

    def test_list(self) -> None:
        fig1 = plt.figure()
        fig2 = plt.figure()
        self.assertTrue(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        plot.close_all([fig1])
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertTrue(plt.fignum_exists(fig2.number))
        plt.close(fig2)
        self.assertFalse(plt.fignum_exists(fig1.number))
        self.assertFalse(plt.fignum_exists(fig2.number))

#%% plotting.get_color_lists
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_get_color_lists(unittest.TestCase):
    r"""
    Tests the plotting.get_color_lists function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.keys = ['default', 'single', 'double', 'vec', 'quat', 'dbl_diff', 'vec_diff', 'quat_diff']

    def test_nominal(self) -> None:
        color_lists: Dict[str, colors.ListedColormap] = plot.get_color_lists()
        for key in self.keys:
            self.assertIn(key, color_lists)
        colormap = color_lists['quat_diff']
        self.assertEqual(colormap.N, 8)
        self.assertEqual(colormap.colors[0], 'xkcd:fuchsia')
        self.assertEqual(colormap.colors[7], 'xkcd:chocolate')

#%% plotting.ignore_plot_data
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_plotting_ignore_plot_data(unittest.TestCase):
    r"""
    Tests the plotting.ignore_plot_data function with the following cases:
        None
        Not ignoring
        Ignoring full data
        Ignoring specific columns
    """
    def setUp(self) -> None:
        self.data = np.zeros((3, 10), dtype=float)
        self.ignore_empties = True

    def test_none(self) -> None:
        ignore = plot.ignore_plot_data(None, True)
        self.assertTrue(ignore)
        ignore = plot.ignore_plot_data(None, False)
        self.assertTrue(ignore)
        ignore = plot.ignore_plot_data(None, False, 0)
        self.assertTrue(ignore)
        ignore = plot.ignore_plot_data(None, True, 0)
        self.assertTrue(ignore)

    def test_not_ignoring(self) -> None:
        ignore = plot.ignore_plot_data(self.data, False)
        self.assertFalse(ignore)
        ignore = plot.ignore_plot_data(self.data, False, 0)
        self.assertFalse(ignore)

    def test_ignoring_no_col(self) -> None:
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties)
        self.assertTrue(ignore)
        self.data[1, 2] = np.nan
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties)
        self.assertTrue(ignore)
        self.data[2, 5] = 0.1
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties)
        self.assertFalse(ignore)

    def test_ignoring_col(self) -> None:
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties, 1)
        self.assertTrue(ignore)
        self.data[1, 2] = np.nan
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties, 0)
        self.assertTrue(ignore)
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties, 1)
        self.assertTrue(ignore)
        self.data[2, 5] = 0.1
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties, 0)
        self.assertTrue(ignore)
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties, 1)
        self.assertTrue(ignore)
        ignore = plot.ignore_plot_data(self.data, self.ignore_empties, 5)
        self.assertFalse(ignore)

#%% plotting.whitten
class Test_plotting_whitten(unittest.TestCase):
    r"""
    Tests the plotting.whitten function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.color = (1, 0.4, 0)
        self.whittened_color = (1.0, 0.58, 0.3)

    def test_nominal(self) -> None:
        new_color = plot.whitten(self.color)
        self.assertEqual(new_color, self.whittened_color)

#%% plotting.resolve_name
class Test_plotting_resolve_name(unittest.TestCase):
    r"""
    Tests the plotting.resolve_name function with the following cases:
        Nominal (Windows)
        Nominal (Unix)
        No bad chars
        Different replacements (x3)
    """
    def setUp(self) -> None:
        self.bad_name = r'Bad name /\ <>!.png'
        self.exp_win  = 'Bad name __ __!.png'
        self.exp_unix = r'Bad name _\ <>!.png'

    def test_nominal(self) -> None:
        new_name = plot.resolve_name(self.bad_name)
        if IS_WINDOWS:
            self.assertEqual(new_name, self.exp_win)
        else:
            self.assertEqual(new_name, self.exp_unix)

    def test_nominal_win(self) -> None:
        new_name = plot.resolve_name(self.bad_name, force_win=True)
        self.assertEqual(new_name, self.exp_win)

    def test_nominal_unix(self) -> None:
        new_name = plot.resolve_name(self.bad_name, force_win=False)
        self.assertEqual(new_name, self.exp_unix)

    def test_no_bad_chars(self) -> None:
        good_name = 'Good name - Nice job.txt'
        new_name  = plot.resolve_name(good_name)
        self.assertEqual(new_name, good_name)

    def test_different_replacements(self) -> None:
        bad_name = 'new <>:"/\\|?*text'
        new_name = plot.resolve_name(bad_name, force_win=True, rep_token='X')
        self.assertEqual(new_name, 'new XXXXXXXXXtext')
        new_name = plot.resolve_name(bad_name, force_win=True, rep_token='')
        self.assertEqual(new_name, 'new text')
        new_name = plot.resolve_name(bad_name, force_win=True, rep_token='YY')
        self.assertEqual(new_name, 'new YYYYYYYYYYYYYYYYYYtext')

#%% plotting.storefig
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_storefig(unittest.TestCase):
    r"""
    Tests the plotting.storefig function with the following cases:
        saving one plot to disk
        saving one plot to multiple plot types
        saving multiple plots to one plot type
        saving to a bad folder location (should raise error)
        specifying a bad plot type (should raise error)
    """
    time: np.ndarray[float, 1]
    data: np.ndarray[float, 1]
    title: str
    folder: str
    plot_type: str
    fig: plt.Figure
    this_filename: str

    @classmethod
    def setUpClass(cls) -> None:
        # create data
        cls.time = np.arange(0, 10, 0.1)
        cls.data = np.sin(cls.time)
        cls.title = 'Test Plot'
        cls.folder = get_tests_dir()
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
        cls.this_filename = ''

    def test_saving(self) -> None:
        plot.storefig(self.fig, self.folder, self.plot_type)
        # assert that file exists
        self.this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type)
        self.assertTrue(os.path.isfile(self.this_filename))

    def test_multiple_plot_types(self) -> None:
        plot_types = ['png', 'svg']
        plot.storefig(self.fig, self.folder, plot_types)
        # assert that files exist
        for this_type in plot_types:
            self.this_filename = os.path.join(self.folder, self.title + '.' + this_type)
            self.assertTrue(os.path.isfile(self.this_filename))

    def test_save_as_jpg(self) -> None:
        # Note: this test case can fail if PIL is not installed, try "pip install Pillow"
        plot.storefig(self.fig, self.folder, 'jpg')
        # assert that files exist
        self.this_filename = os.path.join(self.folder, self.title + '.jpg')
        self.assertTrue(os.path.isfile(self.this_filename))

    def test_multiple_figures(self) -> None:
        plot.storefig([self.fig, self.fig], self.folder, self.plot_type)
        # assert that file exists
        self.this_filename = os.path.join(self.folder, self.title + '.' + self.plot_type)
        self.assertTrue(os.path.isfile(self.this_filename))

    def test_bad_folder(self) -> None:
        with self.assertRaises(ValueError):
            plot.storefig(self.fig, 'ZZ:\\non_existant_path')
        # TODO:
        pass

    def test_bad_plot_type(self) -> None:
        # TODO:
        pass

    def test_bad_characters(self) -> None:
        # change to bad name
        self.fig.canvas.set_window_title('Bad < > / names')
        # save file
        plot.storefig(self.fig, self.folder, self.plot_type)
        # restore filename
        self.fig.canvas.set_window_title(self.title)
        # assert that file exists
        if platform.system() == 'Windows':
            self.this_filename = os.path.join(self.folder, 'Bad _ _ _ names' + '.' + self.plot_type)
        else:
            self.this_filename = os.path.join(self.folder, 'Bad < > _ names' + '.' + self.plot_type)
        self.assertTrue(os.path.isfile(self.this_filename))

    def tearDown(self) -> None:
        # remove file
        if self.this_filename:
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.this_filename)

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close(cls.fig)

#%% plotting.titleprefix
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_titleprefix(unittest.TestCase):
    r"""
    Tests the plotting.titleprefix function with the following cases:
        normal use
        null prefix
        multiple figures
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.title = 'Figure Title'
        self.prefix = 'Prefix'
        self.fig.canvas.set_window_title(self.title)
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        ax = self.fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title('X vs Y')

    def test_normal(self) -> None:
        plot.titleprefix(self.fig, self.prefix)

    def test_null_prefix(self) -> None:
        plot.titleprefix(self.fig)

    def test_multiple_figs(self) -> None:
        plot.titleprefix([self.fig, self.fig], self.prefix)

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.disp_xlimits
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_disp_xlimits(unittest.TestCase):
    r"""
    Tests the plotting.disp_xlimits function with the following cases:
        Normal use
        Null action
        Only xmin
        Only xmax
        Multiple figures
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.xmin = 2
        self.xmax = 5
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        ax = self.fig.add_subplot(111)
        ax.plot(x, y)

    def test_normal(self) -> None:
        plot.disp_xlimits(self.fig, self.xmin, self.xmax)

    def test_null_action(self) -> None:
        plot.disp_xlimits(self.fig)

    def test_just_xmin(self) -> None:
        plot.disp_xlimits(self.fig, xmin=self.xmin)

    def test_just_xmax(self) -> None:
        plot.disp_xlimits(self.fig, xmax=self.xmax)

    def test_multiple_figs(self) -> None:
        plot.disp_xlimits([self.fig, self.fig], self.xmin, self.xmax)

    def test_inf(self) -> None:
        plot.disp_xlimits(self.fig, xmin=-np.inf)
        plot.disp_xlimits(self.fig, xmax=np.inf)

    def test_nat(self) -> None:
        plot.disp_xlimits(self.fig, xmin=np.datetime64('nat'), xmax=self.xmax)
        plot.disp_xlimits(self.fig, xmax=np.datetime64('nat'), xmin=self.xmin)

    def test_datetime(self) -> None:
        plot.disp_xlimits(self.fig, xmin=np.inf, xmax=datetime.datetime(2020, 4, 15, 0, 0, 0))

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.zoom_ylim
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_zoom_ylim(unittest.TestCase):
    r"""
    Tests the plotting.zoom_ylim function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Figure Title')
        self.ax = self.fig.add_subplot(111)
        self.time = np.arange(1, 10, 0.1)
        self.data = self.time ** 2
        self.ax.plot(self.time, self.data)
        self.ax.set_title('X vs Y')
        self.t_start = 3
        self.t_final = 5.0000001

    def test_nominal(self) -> None:
        plot.disp_xlimits(self.fig, self.t_start, self.t_final)
        (old_ymin, old_ymax) = self.ax.get_ylim()
        plot.zoom_ylim(self.ax, self.time, self.data, t_start=self.t_start, t_final=self.t_final)
        (new_ymin, new_ymax) = self.ax.get_ylim()
        self.assertGreater(old_ymax, new_ymax)
        self.assertLess(old_ymin, new_ymin)

    def test_no_zoom(self) -> None:
        (old_ymin, old_ymax) = self.ax.get_ylim()
        plot.zoom_ylim(self.ax, self.time, self.data, pad=2.0)
        (new_ymin, new_ymax) = self.ax.get_ylim()
        self.assertEqual(old_ymax, new_ymax)
        self.assertEqual(old_ymin, new_ymin)

    def test_bad_pad(self) -> None:
        with self.assertRaises(ValueError):
            plot.zoom_ylim(self.ax, self.time, self.data, pad=-10)

    def test_no_pad(self) -> None:
        plot.disp_xlimits(self.fig, self.t_start, self.t_final)
        (old_ymin, old_ymax) = self.ax.get_ylim()
        plot.zoom_ylim(self.ax, self.time, self.data, t_start=self.t_start, t_final=self.t_final, pad=0)
        (new_ymin, new_ymax) = self.ax.get_ylim()
        self.assertGreater(old_ymax, new_ymax)
        self.assertLess(old_ymin, new_ymin)
        self.assertAlmostEqual(new_ymin, self.t_start**2)
        self.assertAlmostEqual(new_ymax, self.t_final**2, places=4)

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.figmenu
@unittest.skipIf(not HAVE_MPL or not _HAVE_QT, 'Skipping due to missing matplotlib/PyQt5 dependency.')
class Test_plotting_figmenu(unittest.TestCase):
    r"""
    Tests the plotting.figmenu function with the following cases:
        One input
        List input
    """
    def setUp(self) -> None:
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()

    def test_one_input(self) -> None:
        plot.figmenu(self.fig1)

    def test_list_input(self) -> None:
        plot.figmenu([self.fig1, self.fig2])

    def tearDown(self) -> None:
        plt.close(self.fig1)
        plt.close(self.fig2)

#%% plotting.rgb_ints_to_hex
class Test_plotting_rgb_ints_to_hex(unittest.TestCase):
    r"""
    Tests the plotting.rgb_ints_to_hex function with the following cases:
        Nominal
        Out of range
    """
    def setUp(self) -> None:
        self.colors    = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (79, 129, 189), (0, 0, 0)]
        self.hex_codes = ['#ff0000', '#00ff00', '#0000ff', '#4f81bd', '#000000']

    def test_nominal(self) -> None:
        for (ix, this_color) in enumerate(self.colors):
            hex_codes = plot.rgb_ints_to_hex(this_color)
            self.assertEqual(hex_codes, self.hex_codes[ix])

    def test_bad_range(self) -> None:
        hex_code = plot.rgb_ints_to_hex((-100, 500, 9))
        self.assertEqual(hex_code, '#00ff09')

#%% plotting.get_screen_resolution
@unittest.skipIf(not _HAVE_QT, 'Skipping due to missing PyQt5 dependency.')
class Test_plotting_get_screen_resolution(unittest.TestCase):
    r"""
    Tests the plotting.get_screen_resolution function with the following cases:
        Nominal
    """
    def test_nominal(self) -> None:
        (screen_width, screen_height) = plot.get_screen_resolution()
        self.assertGreater(screen_width, 0)
        self.assertGreater(screen_height, 0)

#%% plotting.show_zero_ylim
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_show_zero_ylim(unittest.TestCase):
    r"""
    Tests the plotting.show_zero_ylim function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def test_no_change(self) -> None:
        self.ax.plot([1, 5, 10], [200, -150, 240], '.-')
        plot.show_zero_ylim(self.ax)

    def test_all_positive(self) -> None:
        self.ax.plot([1, 5, 10], [200, 250, 240], '.-')
        plot.show_zero_ylim(self.ax)

    def test_all_negative(self) -> None:
        self.ax.plot([1, 5, 10], [-200, -250, -240], '.-')
        plot.show_zero_ylim(self.ax)

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.plot_second_units_wrapper
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_plot_second_units_wrapper(unittest.TestCase):
    r"""
    Tests the plotting.plot_second_units_wrapper function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.description = 'Values over time'
        self.ylabel = 'Value [rad]'
        self.second_yscale: Union[None, int, float, Dict[str, float]] = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], '.-')
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.description)

    def test_none(self) -> None:
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)

    def test_int(self) -> None:
        self.second_yscale = 100
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), '')

    def test_float(self) -> None:
        self.second_yscale = 100.
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), '')

    def test_zero(self) -> None:
        self.second_yscale = 0.
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)
        self.second_yscale = {'new': 0}
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)

    def test_nan(self) -> None:
        self.second_yscale = np.nan
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)
        self.second_yscale = {'new': np.nan}
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)

    def test_full_replace(self) -> None:
        self.second_yscale = {u'Better Units [µrad]': 1e6}
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), u'Better Units [µrad]')

    def test_units_only(self) -> None:
        self.second_yscale = {'mrad': 1e3}
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), 'Value [mrad]')

    def test_no_units(self) -> None:
        self.ax.set_ylabel('Value')
        self.second_yscale = {'New Value': 1e3}
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_yscale)
        self.assertEqual(self.ax.get_ylabel(), 'Value')
        self.assertEqual(ax2.get_ylabel(), 'New Value')

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.plot_second_yunits
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_plot_second_yunits(unittest.TestCase):
    r"""
    Tests the plotting.plot_second_yunits function with the following cases:
        TBD
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], '.-')
        self.ax.set_ylabel('Value [rad]')
        self.ylab = u'Value [µrad]'
        self.multiplier = 1e6

    def test_nominal(self) -> None:
        ax2 = plot.plot_second_yunits(self.ax, self.ylab, self.multiplier)
        self.assertEqual(self.ax.get_ylabel(), 'Value [rad]')
        self.assertEqual(ax2.get_ylabel(), self.ylab)

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.get_rms_indices
@unittest.skipIf(not HAVE_NUMPY, 'Skipping due to missing numpy dependency.')
class Test_plotting_get_rms_indices(unittest.TestCase):
    r"""
    Tests the plotting.get_rms_indices function with the following cases:
        Nominal
        TBD
    """
    def setUp(self) -> None:
        self.time_one       = np.arange(11)
        self.time_two       = np.arange(2, 13)
        self.time_overlap   = np.arange(2, 11)
        self.xmin           = 1
        self.xmax           = 8
        self.exp            = dict()
        self.exp['one']     = np.array([False,  True,  True,  True,  True,  True,  True,  True,  True, False, False], dtype=bool)
        self.exp['two']     = np.array([ True,  True,  True,  True,  True,  True,  True, False, False, False, False], dtype=bool)
        self.exp['overlap'] = np.array([ True,  True,  True,  True,  True,  True,  True, False, False], dtype=bool)
        self.exp['pts']     = [1, 8]

    def test_nominal(self) -> None:
        ix = plot.get_rms_indices(self.time_one, self.time_two, self.time_overlap, xmin=self.xmin, xmax=self.xmax)
        for key in ix.keys():
            np.testing.assert_array_equal(ix[key], self.exp[key])

    def test_only_time_one(self) -> None:
        self.exp['two'] = np.array([])
        self.exp['overlap'] = np.array([])
        ix = plot.get_rms_indices(self.time_one, None, None, xmin=self.xmin, xmax=self.xmax)
        for key in ix.keys():
            np.testing.assert_array_equal(ix[key], self.exp[key])

    def test_no_bounds(self) -> None:
        self.exp['one'].fill(True)
        self.exp['two'].fill(True)
        self.exp['overlap'].fill(True)
        self.exp['pts'] = [0, 12]
        ix = plot.get_rms_indices(self.time_one, self.time_two, self.time_overlap)
        for key in ix.keys():
            np.testing.assert_array_equal(ix[key], self.exp[key])

    def test_datetime64(self) -> None:
        pass # TODO: write this

    def test_datetime(self) -> None:
        pass # TODO: write this

#%% plotting.plot_vert_lines
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_plot_vert_lines(unittest.TestCase):
    r"""
    Tests the plotting.plot_vert_lines function with the following cases:
        Nominal
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(np.arange(10), np.arange(10), label='Data')
        self.x = (2, 5)

    def test_nominal(self) -> None:
        plot.plot_vert_lines(self.ax, self.x, show_in_legend=False)
        self.ax.legend()

    def test_no_legend(self) -> None:
        plot.plot_vert_lines(self.ax, self.x, show_in_legend=True)
        self.ax.legend()

    def test_multiple_lines(self) -> None:
        labels = ['Line 1', 'Line 2', 'Line 3', 'Line 4']
        colormap = colors.ListedColormap(['r', 'g', 'b', 'k'])
        plot.plot_vert_lines(self.ax, [1, 2.5, 3.5, 8], show_in_legend=True, labels=labels, colormap=colormap)
        self.ax.legend()

    def test_multiple_unlabeled(self) -> None:
        plot.plot_vert_lines(self.ax, np.arange(0.5, 7.5, 1.), show_in_legend=False)
        self.ax.legend()

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.plot_phases
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_plot_phases(unittest.TestCase):
    r"""
    Tests the plotting.plot_phases function with the following cases:
        Single time
        End times
        No labels
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Sine Wave')
        self.ax = self.fig.add_subplot(111)
        time = np.arange(101)
        data = np.cos(time / 10)
        self.times = np.array([5, 20, 60, 90])
        self.times2 = np.array([[5, 20, 60, 90], [10, 60, 90, 95]])
        self.ax.plot(time, data, '.-')
        self.colormap = 'tab10'
        self.labels = ['Part 1', 'Phase 2', 'Watch Out', 'Final']

    def test_single_time(self) -> None:
        plot.plot_phases(self.ax, self.times, self.colormap, self.labels)

    def test_with_end_times(self) -> None:
        plot.plot_phases(self.ax, self.times2, self.colormap, self.labels)

    def test_no_labels(self) -> None:
        plot.plot_phases(self.ax, self.times, colormap=self.colormap)

    def test_no_colormap(self) -> None:
        plot.plot_phases(self.ax, self.times, labels=self.labels)

    def test_group_all_defaults(self) -> None:
        plot.plot_phases(self.ax, self.times, group_all=True)

    def test_group_all(self) -> None:
        plot.plot_phases(self.ax, self.times, group_all=True, labels=self.labels[0], colormap='red')

    def tearDown(self) -> None:
        if self.fig:
            plt.close(self.fig)

#%% plotting.plot_classification
@unittest.skipIf(not HAVE_MPL, 'Skipping due to missing matplotlib dependency.')
class Test_plotting_plot_classification(unittest.TestCase):
    r"""
    Tests the plotting.plot_classification function with the following cases:
        Inside axes
        Outside axes
        Classified options with test banner
        Bad option (should error)
    """
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([0, 10], [0, 10], '.-b')

    def test_inside(self) -> None:
        plot.plot_classification(self.ax, 'U', test=False, location='axis')

    def test_outside(self) -> None:
        plot.plot_classification(self.ax, 'U', test=False, location='figure')

    def test_caveat(self) -> None:
        plot.plot_classification(self.ax, 'U', caveat='//TEXT STR')

    def test_options(self) -> None:
        for opt in {'C', 'S', 'T', 'TS'}:
            plot.plot_classification(self.ax, opt, test=True, location='figure')

    def test_bad_option(self) -> None:
        with self.assertRaises(ValueError):
            plot.plot_classification(self.ax, 'BadOption')

    def tearDown(self) -> None:
        plt.close(self.fig)

#%% plotting.align_plots
class Test_plotting_align_plots(unittest.TestCase):
    r"""
    Tests the plotting.align_plots function with the following cases:
        TBD
    """
    pass # TODO: write this

#%% plotting.z_from_ci
@unittest.skipIf(not HAVE_SCIPY, 'Skipping due to missing scipy dependency.')
class Test_plotting_z_from_ci(unittest.TestCase):
    r"""
    Tests the plotting.z_from_ci function with the following cases:
        Nominal with 4 common values found online
    """
    def setUp(self) -> None:
        self.cis = [0.90,  0.95,  0.98,  0.99]
        self.zs  = [1.645, 1.96, 2.326, 2.576]

    def test_nominal(self) -> None:
        for (ci, exp_z) in zip(self.cis, self.zs):
            z = plot.z_from_ci(ci)
            self.assertTrue(abs(z - exp_z) < 0.001, '{} and {} are more than 0.001 from each other.'.format(z, exp_z))

#%% Unit test execution
if __name__ == '__main__':
    if HAVE_MPL:
        plt.ioff()
    unittest.main(exit=False)
