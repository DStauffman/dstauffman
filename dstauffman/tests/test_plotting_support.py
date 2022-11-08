r"""
Test file for the `support` module of the "dstauffman.plotting" library.

Notes
-----
#.  Written by David C. Stauffer in December 2018.
"""

#%% Imports
from __future__ import annotations

import datetime
import os
import pathlib
import platform
from typing import Dict, List, Optional, Tuple, Union
import unittest

from dstauffman import get_tests_dir, HAVE_DS, HAVE_MPL, HAVE_NUMPY, HAVE_SCIPY, IS_WINDOWS
import dstauffman.plotting as plot

if HAVE_MPL:
    import matplotlib as mpl
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
if HAVE_NUMPY:
    import numpy as np
try:
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest

    _HAVE_QT = True
except ModuleNotFoundError:
    _HAVE_QT = False
except ImportError:
    _HAVE_QT = False

_HAVE_DISPLAY = IS_WINDOWS or bool(os.environ.get("DISPLAY", None))

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


#%% plotting.COLOR_LISTS
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_COLOR_LISTS(unittest.TestCase):
    r"""
    Tests the plotting.COLOR_LISTS dictionary with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.keys = ["default", "single", "double", "vec", "quat", "dbl_diff", "vec_diff", "quat_diff"]

    def test_nominal(self) -> None:
        for key in self.keys:
            self.assertIn(key, plot.COLOR_LISTS)
        colormap = plot.COLOR_LISTS["quat_diff"]
        self.assertEqual(colormap.N, 8)
        self.assertEqual(colormap.colors[0], "xkcd:fuchsia")
        self.assertEqual(colormap.colors[7], "xkcd:chocolate")


#%% plotting._HoverButton
class Test_plotting__HoverButton(unittest.TestCase):
    r"""
    Tests the plotting._HoverButton class with the following cases:
        TBD
    """
    pass  # TODO: write this


#%% plotting.MyCustomToolbar
@unittest.skipIf(not HAVE_MPL or not _HAVE_QT or not _HAVE_DISPLAY, "Skipping due to missing matplotlib/Qt/DISPLAY dependency.")
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
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
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
        # fmt: off
        self.colormap   = "Paired"
        self.low        = 0
        self.high       = 1
        self.num_colors = 5
        self.fig: Optional[plt.Figure] = None
        # fmt: on

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
        colormap = colors.ListedColormap(["r", "g", "b"])
        cm = plot.ColorMap(colormap, num_colors=3)
        red = cm.get_color(0)
        green = cm.get_color(1)
        blue = cm.get_color(2)
        self.assertEqual(red, (1.0, 0, 0, 1))
        self.assertEqual(green, (0.0, 0.5, 0, 1))
        self.assertEqual(blue, (0.0, 0, 1, 1))

    def tearDown(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)


#%% plotting.close_all
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
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


#%% plotting.get_nondeg_colorlists
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_get_nondeg_colorlists(unittest.TestCase):
    r"""
    Tests the plotting.get_nondeg_colorlists function with the following cases:
        One
        Two
        Three
        Four
        5-10
        More than 10
    """

    def test_one(self) -> None:
        clist = plot.get_nondeg_colorlists(1)
        self.assertEqual(clist.N, 3)
        self.assertEqual(clist.colors[0], "#1f77b4")
        self.assertEqual(clist.colors[1], "xkcd:blue")
        self.assertEqual(clist.colors[2], "#1f77b4")

    def test_two(self) -> None:
        clist = plot.get_nondeg_colorlists(2)
        self.assertEqual(clist.N, 6)
        self.assertEqual(clist.colors[0], "xkcd:red")
        self.assertEqual(clist.colors[2], "xkcd:fuchsia")
        self.assertEqual(clist.colors[4], "xkcd:red")

    def test_three(self) -> None:
        clist = plot.get_nondeg_colorlists(3)
        self.assertEqual(clist.N, 9)
        self.assertEqual(clist.colors[0], "xkcd:red")
        self.assertEqual(clist.colors[3], "xkcd:fuchsia")
        self.assertEqual(clist.colors[6], "xkcd:red")

    def test_four(self) -> None:
        clist = plot.get_nondeg_colorlists(4)
        self.assertEqual(clist.N, 12)
        self.assertEqual(clist.colors[0], "xkcd:red")
        self.assertEqual(clist.colors[4], "xkcd:fuchsia")
        self.assertEqual(clist.colors[8], "xkcd:red")

    def test_five_to_ten(self) -> None:
        cmap = mpl.colormaps["tab20"]
        exp1 = cmap.colors[0]
        exp2 = cmap.colors[1]
        for i in [5, 8, 10]:
            clist = plot.get_nondeg_colorlists(i)
            self.assertEqual(clist.N, 3 * i)
            self.assertEqual(clist.colors[0], exp1)
            self.assertEqual(clist.colors[i], exp2)
            self.assertEqual(clist.colors[2 * i], exp1)

    def test_lots(self) -> None:
        cmap = mpl.colormaps["tab20"]
        exp1 = cmap.colors[0]
        exp2 = cmap.colors[1]
        for i in [11, 15, 20, 50]:
            clist = plot.get_nondeg_colorlists(i)
            self.assertEqual(clist.N, 3 * i)
            self.assertEqual(clist.colors[0], exp1)
            self.assertEqual(clist.colors[i], exp2)
            self.assertEqual(clist.colors[2 * i], exp1)


#%% plotting.ignore_plot_data
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_plotting_ignore_plot_data(unittest.TestCase):
    r"""
    Tests the plotting.ignore_plot_data function with the following cases:
        None
        Not ignoring
        Ignoring full data
        Ignoring specific columns
    """

    def setUp(self) -> None:
        self.data = np.zeros((3, 10))
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

    def test_nominal(self) -> None:
        new_color = plot.whitten((1.0, 0.4, 0.0))
        self.assertEqual(new_color, (1.0, 0.58, 0.3))

    def test_blacken(self) -> None:
        new_color = plot.whitten((1.0, 0.4, 0.0, 0.5), white=(0.0, 0.0, 0.0, 0.0), dt=0.5)
        self.assertEqual(new_color, (0.5, 0.2, 0.0, 0.25))


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
        self.bad_name = r"Bad name /\ <>!.png"
        self.exp_win = "Bad name __ __!.png"
        self.exp_unix = r"Bad name _\ <>!.png"

    def test_nominal(self) -> None:
        new_name = plot.resolve_name(self.bad_name)
        if IS_WINDOWS:
            self.assertEqual(new_name, self.exp_win)  # pragma: noc unix
        else:
            self.assertEqual(new_name, self.exp_unix)  # pragma: noc windows

    def test_nominal_win(self) -> None:
        new_name = plot.resolve_name(self.bad_name, force_win=True)
        self.assertEqual(new_name, self.exp_win)

    def test_nominal_unix(self) -> None:
        new_name = plot.resolve_name(self.bad_name, force_win=False)
        self.assertEqual(new_name, self.exp_unix)

    def test_no_bad_chars(self) -> None:
        good_name = "Good name - Nice job.txt"
        new_name = plot.resolve_name(good_name)
        self.assertEqual(new_name, good_name)

    def test_different_replacements(self) -> None:
        bad_name = 'new <>:"/\\|?*text'
        new_name = plot.resolve_name(bad_name, force_win=True, rep_token="X")
        self.assertEqual(new_name, "new XXXXXXXXXtext")
        new_name = plot.resolve_name(bad_name, force_win=True, rep_token="")
        self.assertEqual(new_name, "new text")
        new_name = plot.resolve_name(bad_name, force_win=True, rep_token="YY")
        self.assertEqual(new_name, "new YYYYYYYYYYYYYYYYYYtext")

    def test_newlines(self) -> None:
        bad_name = "Hello\nWorld.jpg"
        new_name = plot.resolve_name(bad_name)
        self.assertEqual(new_name, "Hello_World.jpg")


#%% plotting.storefig
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_storefig(unittest.TestCase):
    r"""
    Tests the plotting.storefig function with the following cases:
        saving one plot to disk
        saving one plot to multiple plot types
        saving multiple plots to one plot type
        saving to a bad folder location (should raise error)
        specifying a bad plot type (should raise error)
    """
    time: np.ndarray
    data: np.ndarray
    title: str
    folder: pathlib.Path
    plot_type: str
    fig: plt.Figure
    this_filename: Optional[pathlib.Path]
    show_warn: bool

    @classmethod
    def setUpClass(cls) -> None:
        # create data
        cls.time = np.arange(0, 10, 0.1)
        cls.data = np.sin(cls.time)
        cls.title = "Test Plot"
        cls.folder = get_tests_dir()
        cls.plot_type = "png"
        # create the figure and set the title
        cls.fig = plt.figure()
        cls.fig.canvas.manager.set_window_title(cls.title)
        # add an axis and plot the data
        ax = cls.fig.add_subplot(111)
        ax.plot(cls.time, cls.data)
        # add labels and legends
        ax.set_xlabel("Time [year]")
        ax.set_ylabel("Value [units]")
        ax.set_title(cls.title)
        # show a grid
        ax.grid(True)
        cls.this_filename = None
        # suppress warnings for no display
        cls.show_warn = _HAVE_DISPLAY

    def test_saving(self) -> None:
        plot.storefig(self.fig, self.folder, self.plot_type, self.show_warn)
        # assert that file exists
        self.this_filename = self.folder.joinpath(self.title + "." + self.plot_type)
        self.assertTrue(self.this_filename.is_file())

    def test_multiple_plot_types(self) -> None:
        plot_types = ["png", "svg"]
        plot.storefig(self.fig, self.folder, plot_types, self.show_warn)
        # assert that files exist
        for this_type in plot_types:
            self.this_filename = self.folder.joinpath(self.title + "." + this_type)
            self.assertTrue(self.this_filename.is_file())

    def test_save_as_jpg(self) -> None:
        # Note: this test case can fail if PIL is not installed, try "pip install Pillow"
        plot.storefig(self.fig, self.folder, "jpg", self.show_warn)
        # assert that files exist
        self.this_filename = self.folder.joinpath(self.title + ".jpg")
        self.assertTrue(self.this_filename.is_file())

    def test_multiple_figures(self) -> None:
        plot.storefig([self.fig, self.fig], self.folder, self.plot_type, self.show_warn)
        # assert that file exists
        self.this_filename = self.folder.joinpath(self.title + "." + self.plot_type)
        self.assertTrue(self.this_filename.is_file())

    def test_bad_folder(self) -> None:
        with self.assertRaises(ValueError):
            plot.storefig(self.fig, "X:\\non_existant_path", show_warn=self.show_warn)
        # TODO:
        pass

    def test_bad_plot_type(self) -> None:
        # TODO:
        pass

    def test_bad_characters(self) -> None:
        # change to bad name
        self.fig.canvas.manager.set_window_title("Bad < > / names")
        if not _HAVE_DISPLAY:  # pragma: no cover
            self.fig.axes[0].set_title("Bad < > / names")
        # save file
        plot.storefig(self.fig, self.folder, self.plot_type, self.show_warn)
        # restore filename
        self.fig.canvas.manager.set_window_title(self.title)
        if not _HAVE_DISPLAY:  # pragma: no cover
            self.fig.axes[0].set_title(self.title)
        # assert that file exists
        if platform.system() == "Windows":
            self.this_filename = self.folder.joinpath("Bad _ _ _ names" + "." + self.plot_type)  # pragma: noc unix
        else:
            self.this_filename = self.folder.joinpath("Bad < > _ names" + "." + self.plot_type)  # pragma: noc windows
        self.assertTrue(self.this_filename.is_file())

    def tearDown(self) -> None:
        # remove file
        if self.this_filename is not None:
            self.this_filename.unlink(missing_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close(cls.fig)


#%% plotting.titleprefix
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_titleprefix(unittest.TestCase):
    r"""
    Tests the plotting.titleprefix function with the following cases:
        normal use
        null prefix
        multiple figures
    """

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.title = "Figure Title"
        self.prefix = "Prefix"
        self.fig.canvas.manager.set_window_title(self.title)
        x = np.arange(0, 10, 0.1)
        y = np.sin(x)
        ax = self.fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title("X vs Y")

    def test_normal(self) -> None:
        plot.titleprefix(self.fig, self.prefix)

    def test_null_prefix(self) -> None:
        plot.titleprefix(self.fig)

    def test_multiple_figs(self) -> None:
        plot.titleprefix([self.fig, self.fig], self.prefix)

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.disp_xlimits
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
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
        plot.disp_xlimits(self.fig, xmin=np.datetime64("nat"), xmax=self.xmax)
        plot.disp_xlimits(self.fig, xmax=np.datetime64("nat"), xmin=self.xmin)

    def test_datetime(self) -> None:
        plot.disp_xlimits(self.fig, xmin=np.inf, xmax=datetime.datetime(2020, 4, 15, 0, 0, 0))

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.zoom_ylim
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_zoom_ylim(unittest.TestCase):
    r"""
    Tests the plotting.zoom_ylim function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title("Figure Title")
        self.ax = self.fig.add_subplot(111)
        self.time = np.arange(1, 10, 0.1)
        self.data = self.time**2
        self.ax.plot(self.time, self.data)
        self.ax.set_title("X vs Y")
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
@unittest.skipIf(not HAVE_MPL or not _HAVE_QT, "Skipping due to missing matplotlib/Qt dependency.")
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
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (79, 129, 189), (0, 0, 0)]
        self.hex_codes = ["#ff0000", "#00ff00", "#0000ff", "#4f81bd", "#000000"]

    def test_nominal(self) -> None:
        for (ix, this_color) in enumerate(self.colors):
            hex_codes = plot.rgb_ints_to_hex(this_color)
            self.assertEqual(hex_codes, self.hex_codes[ix])

    def test_bad_range(self) -> None:
        hex_code = plot.rgb_ints_to_hex((-100, 500, 9))
        self.assertEqual(hex_code, "#00ff09")


#%% plotting.get_screen_resolution
@unittest.skipIf(not _HAVE_QT, "Skipping due to missing Qt dependency.")
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
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_show_zero_ylim(unittest.TestCase):
    r"""
    Tests the plotting.show_zero_ylim function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def test_no_change(self) -> None:
        self.ax.plot([1, 5, 10], [200, -150, 240], ".-")
        plot.show_zero_ylim(self.ax)

    def test_all_positive(self) -> None:
        self.ax.plot([1, 5, 10], [200, 250, 240], ".-")
        plot.show_zero_ylim(self.ax)

    def test_all_negative(self) -> None:
        self.ax.plot([1, 5, 10], [-200, -250, -240], ".-")
        plot.show_zero_ylim(self.ax)

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.plot_second_units_wrapper
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_plot_second_units_wrapper(unittest.TestCase):
    r"""
    Tests the plotting.plot_second_units_wrapper function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.description = "Values over time"
        self.ylabel = "Value [rad]"
        self.second_units: Union[None, int, float, Tuple[str, float]] = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], ".-")
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.description)

    def test_none(self) -> None:
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)

    def test_int(self) -> None:
        self.second_units = 100
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), "")

    def test_float(self) -> None:
        self.second_units = 100.0
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), "")

    def test_zero(self) -> None:
        self.second_units = 0.0
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)
        self.second_units = ("new", 0)
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)

    def test_nan(self) -> None:
        self.second_units = np.nan
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)
        self.second_units = ("new", np.nan)
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertIsNone(ax2)

    def test_full_replace(self) -> None:
        self.second_units = ("Better Units [µrad]", 1e6)
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), "Better Units [µrad]")

    def test_units_only(self) -> None:
        self.second_units = ("mrad", 1e3)
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), self.ylabel)
        self.assertEqual(ax2.get_ylabel(), "Value [mrad]")

    def test_no_units(self) -> None:
        self.ax.set_ylabel("Value")
        self.second_units = ("New Value", 1e3)
        ax2 = plot.plot_second_units_wrapper(self.ax, self.second_units)
        self.assertEqual(self.ax.get_ylabel(), "Value")
        self.assertEqual(ax2.get_ylabel(), "New Value")

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.plot_second_yunits
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_plot_second_yunits(unittest.TestCase):
    r"""
    Tests the plotting.plot_second_yunits function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], ".-")
        self.ax.set_ylabel("Value [rad]")
        self.ylab = "Value [µrad]"
        self.multiplier = 1e6

    def test_nominal(self) -> None:
        ax2 = plot.plot_second_yunits(self.ax, self.ylab, self.multiplier)
        self.assertEqual(self.ax.get_ylabel(), "Value [rad]")
        self.assertEqual(ax2.get_ylabel(), self.ylab)

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.get_rms_indices
@unittest.skipIf(not HAVE_NUMPY, "Skipping due to missing numpy dependency.")
class Test_plotting_get_rms_indices(unittest.TestCase):
    r"""
    Tests the plotting.get_rms_indices function with the following cases:
        Nominal
        TBD
    """

    def setUp(self) -> None:
        # fmt: off
        self.time_one       = np.arange(11)
        self.time_two       = np.arange(2, 13)
        self.time_overlap   = np.arange(2, 11)
        self.xmin           = 1
        self.xmax           = 8
        self.exp: Dict[str, Union[np.ndarray, List[int]]] = dict()
        self.exp["one"]     = np.array([False,  True,  True,  True,  True,  True,  True,  True,  True, False, False], dtype=bool)
        self.exp["two"]     = np.array([ True,  True,  True,  True,  True,  True,  True, False, False, False, False], dtype=bool)
        self.exp["overlap"] = np.array([ True,  True,  True,  True,  True,  True,  True, False, False], dtype=bool)
        self.exp["pts"]     = [1, 8]
        # fmt: on

    def test_nominal(self) -> None:
        ix = plot.get_rms_indices(self.time_one, self.time_two, self.time_overlap, xmin=self.xmin, xmax=self.xmax)
        for key in ix.keys():
            np.testing.assert_array_equal(ix[key], self.exp[key])

    def test_only_time_one(self) -> None:
        self.exp["two"] = np.array([])
        self.exp["overlap"] = np.array([])
        ix = plot.get_rms_indices(self.time_one, None, None, xmin=self.xmin, xmax=self.xmax)
        for key in ix.keys():
            np.testing.assert_array_equal(ix[key], self.exp[key])

    def test_no_bounds(self) -> None:
        self.exp["one"].fill(True)  # type: ignore[union-attr]
        self.exp["two"].fill(True)  # type: ignore[union-attr]
        self.exp["overlap"].fill(True)  # type: ignore[union-attr]
        self.exp["pts"] = [0, 12]
        ix = plot.get_rms_indices(self.time_one, self.time_two, self.time_overlap)
        for key in ix.keys():
            np.testing.assert_array_equal(ix[key], self.exp[key])

    def test_datetime64(self) -> None:
        pass  # TODO: write this

    def test_datetime(self) -> None:
        pass  # TODO: write this


#%% plotting.plot_vert_lines
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_plot_vert_lines(unittest.TestCase):
    r"""
    Tests the plotting.plot_vert_lines function with the following cases:
        Nominal
    """

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(np.arange(10), np.arange(10), label="Data")
        self.x = (2, 5)

    def test_nominal(self) -> None:
        plot.plot_vert_lines(self.ax, self.x, show_in_legend=False)
        self.ax.legend()

    def test_no_legend(self) -> None:
        plot.plot_vert_lines(self.ax, self.x, show_in_legend=True)
        self.ax.legend()

    def test_multiple_lines(self) -> None:
        labels = ["Line 1", "Line 2", "Line 3", "Line 4"]
        colormap = colors.ListedColormap(["r", "g", "b", "k"])
        plot.plot_vert_lines(self.ax, [1, 2.5, 3.5, 8], show_in_legend=True, labels=labels, colormap=colormap)
        self.ax.legend()

    def test_multiple_unlabeled(self) -> None:
        plot.plot_vert_lines(self.ax, np.arange(0.5, 7.5, 1.0), show_in_legend=False)
        self.ax.legend()

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.plot_phases
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_plot_phases(unittest.TestCase):
    r"""
    Tests the plotting.plot_phases function with the following cases:
        Single time
        End times
        No labels
    """

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title("Sine Wave")
        self.ax = self.fig.add_subplot(111)
        time = np.arange(101)
        data = np.cos(time / 10)
        self.times = np.array([5, 20, 60, 90])
        self.times2 = np.array([[5, 20, 60, 90], [10, 60, 90, 95]])
        self.ax.plot(time, data, ".-")
        self.colormap = "tab10"
        self.labels = ["Part 1", "Phase 2", "Watch Out", "Final"]

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
        plot.plot_phases(self.ax, self.times, group_all=True, labels=self.labels[0], colormap="red")

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.plot_classification
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
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
        self.ax.plot([0, 10], [0, 10], ".-b")

    def test_inside(self) -> None:
        plot.plot_classification(self.ax, "U", test=False, location="axis")

    def test_outside(self) -> None:
        plot.plot_classification(self.ax, "U", test=False, location="figure")

    def test_caveat(self) -> None:
        plot.plot_classification(self.ax, "U", caveat="//TEXT STR")

    def test_options(self) -> None:
        for opt in {"C", "S", "T", "TS"}:
            plot.plot_classification(self.ax, opt, test=True, location="figure")

    def test_bad_option(self) -> None:
        with self.assertRaises(ValueError):
            plot.plot_classification(self.ax, "BadOption")

    def tearDown(self) -> None:
        plt.close(self.fig)


#%% plotting.align_plots
class Test_plotting_align_plots(unittest.TestCase):
    r"""
    Tests the plotting.align_plots function with the following cases:
        TBD
    """
    pass  # TODO: write this


#%% plotting.z_from_ci
@unittest.skipIf(not HAVE_SCIPY, "Skipping due to missing scipy dependency.")
class Test_plotting_z_from_ci(unittest.TestCase):
    r"""
    Tests the plotting.z_from_ci function with the following cases:
        Nominal with 4 common values found online
    """

    def setUp(self) -> None:
        # fmt: off
        self.cis = [0.90,  0.95,  0.98,  0.99]
        self.zs  = [1.645, 1.96, 2.326, 2.576]
        # fmt: on

    def test_nominal(self) -> None:
        for (ci, exp_z) in zip(self.cis, self.zs):
            z = plot.z_from_ci(ci)
            self.assertTrue(abs(z - exp_z) < 0.001, "{} and {} are more than 0.001 from each other.".format(z, exp_z))


#%% plotting.save_figs_to_pdf
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_plotting_save_figs_to_pdf(unittest.TestCase):
    r"""
    Tests the plotting.save_figs_to_pdf function with the following cases:
        TBD
    """
    fig1: plt.Figure
    fig2: plt.Figure
    filename: pathlib.Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.fig1 = plt.figure()
        ax1 = cls.fig1.add_subplot(1, 1, 1)
        ax1.plot(0, 0)
        cls.fig2 = plt.figure()
        ax2 = cls.fig2.add_subplot(1, 1, 1)
        ax2.plot([0, 1, 2], [2, 4, 6])
        cls.filename = get_tests_dir() / "pdf_figures.pdf"

    def test_nominal(self) -> None:
        plot.save_figs_to_pdf([self.fig1, self.fig2], self.filename)

    def test_none(self) -> None:
        plot.save_figs_to_pdf(None, self.filename)

    def test_single(self) -> None:
        plot.save_figs_to_pdf(self.fig1, self.filename)

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close(cls.fig1)
        plt.close(cls.fig2)
        cls.filename.unlink(missing_ok=True)


#%% plotting.add_datashaders
@unittest.skipIf(not HAVE_DS or not HAVE_MPL, "Skipping due to missing datashader dependency.")
class Test_plotting_add_datashaders(unittest.TestCase):
    r"""
    Tests the plotting.add_datashaders function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.plot([0, 0], [1, 1], ".-")
        self.points = 0.5 + 0.25 * np.random.rand(2, 1000)
        self.datashaders = [{"time": self.points[0, :], "data": self.points[1, :], "ax": self.ax}]

    def test_nominal(self) -> None:
        self.datashaders[0]["color"] = "xkcd:black"
        plot.add_datashaders(self.datashaders)

    def test_datetime(self) -> None:
        self.datashaders[0]["color"] = "xkcd:black"
        self.datashaders.append({"color": "xkcd:blue", "ax": self.ax})
        self.datashaders[1]["time"] = np.arange(np.datetime64("now", "ns"), np.timedelta64(1, "m"), np.timedelta64(1, "s"))
        self.datashaders[1]["data"] = np.random.rand(*self.datashaders[1]["time"].shape)
        self.datashaders[1]["time_units"] = "numpy"
        plot.add_datashaders(self.datashaders)

    def test_min_max(self) -> None:
        self.datashaders[0]["colormap"] = "autumn_r"
        self.datashaders[0]["vmin"] = 0.2
        self.datashaders[0]["vmax"] = 0.6
        plot.add_datashaders(self.datashaders)

    def test_colormap(self) -> None:
        self.datashaders[0]["colormap"] = "autumn_r"
        plot.add_datashaders(self.datashaders)


#%% plotting.fig_ax_factory
@unittest.skipIf(not HAVE_MPL, "Skipping due to missing matplotlib dependency.")
class Test_fig_ax_factory(unittest.TestCase):
    r"""
    Tests the plotting.fig_ax_factory function with the following cases:
        TBD
    """

    def setUp(self) -> None:
        self.fig_ax: Union[Tuple[None, ...], Tuple[Tuple[plt.Figure, plt.Axes], ...]] = (None,)

    def test_1d_rows(self) -> None:
        self.fig_ax = plot.fig_ax_factory(num_axes=4, layout="rows", sharex=True)  # type: ignore[call-overload]
        self.assertEqual(len(self.fig_ax), 4)
        # TODO: figure out how to test rows and sharex

    def test_1d_cols(self) -> None:
        self.fig_ax = plot.fig_ax_factory(num_axes=4, layout="cols", sharex=False)  # type: ignore[call-overload]
        self.assertEqual(len(self.fig_ax), 4)
        # TODO: figure out how to test rows and sharex

    def test_1d_bad_layout(self) -> None:
        with self.assertRaises(ValueError):
            plot.fig_ax_factory(num_axes=4, layout="rowwise")  # type: ignore[call-overload]

    def test_multi_figures(self) -> None:
        self.fig_ax = plot.fig_ax_factory(2, 1)  # type: ignore[call-overload]
        self.assertEqual(len(self.fig_ax), 2)
        self.assertIsNot(self.fig_ax[0], self.fig_ax[1])

    def test_2d(self) -> None:
        self.fig_ax = plot.fig_ax_factory(num_axes=[2, 3], layout="rowwise", sharex=True)  # type: ignore[call-overload]
        self.assertEqual(len(self.fig_ax), 6)

    def test_2d_colwise(self) -> None:
        self.fig_ax = plot.fig_ax_factory(num_axes=[3, 2], layout="colwise", sharex=False)  # type: ignore[call-overload]
        self.assertEqual(len(self.fig_ax), 6)

    def test_2d_bad_layout(self) -> None:
        with self.assertRaises(ValueError):
            plot.fig_ax_factory(num_axes=[4, 4], layout="rows")  # type: ignore[call-overload]

    def test_suptitle(self) -> None:
        self.fig_ax = plot.fig_ax_factory(num_axes=1, suptitle="Test Title")  # type: ignore[call-overload]
        self.assertEqual(len(self.fig_ax), 1)
        this_fig = self.fig_ax[0][0]  # type: ignore[index]
        if _HAVE_DISPLAY:  # pragma: no cover
            self.assertEqual(this_fig.canvas.manager.get_window_title(), "Test Title")
        self.assertEqual(this_fig._suptitle.get_text(), "Test Title")

    def test_passthrough(self) -> None:
        fig_ax = plot.fig_ax_factory(num_axes=3, passthrough=True)  # type: ignore[call-overload]
        self.assertEqual(len(fig_ax), 3)
        for i in range(3):
            self.assertIsNone(fig_ax[i])

    def tearDown(self) -> None:
        last_fig = None
        for fig_ax in self.fig_ax:
            if fig_ax is not None:
                this_fig = fig_ax[0]
                if this_fig is not last_fig:
                    plt.close(this_fig)
                last_fig = this_fig


#%% Unit test execution
if __name__ == "__main__":
    plot.suppress_plots()
    unittest.main(exit=False)
