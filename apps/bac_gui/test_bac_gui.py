# -*- coding: utf-8 -*-
r"""
Test file for the `dstauffman.apps.bac_gui.bac_gui` module.  It is intented to contain test cases to
demonstrate functionaliy and correct outcomes for all the classes and functions within the module.

Notes
-----
#.  Written by David C. Stauffer in June 2016.
"""

#%% Imports
# normal imports
import copy
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest
# Qt imports
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
# model imports
import dstauffman.apps.bac_gui as bac

#%% Constants
class Test_Constants(unittest.TestCase):
    r"""
    Tests the defined constants to verify that they can be used.
    """
    def test_init(self):
        self.assertEqual(bac.GUI_TOKEN, -1)
        self.assertAlmostEqual(bac.LEGAL_LIMIT, 0.0008)
        self.assertAlmostEqual(bac.BMI_CONV, 703.0704)

    def test_enums(self):
        m = bac.Gender.male
        f = bac.Gender.female
        self.assertNotEqual(m, f)

#%% GuiSettings
class Test_GuiSettings(unittest.TestCase):
    r"""
    Tests the GuiSettings class with the following cases:
        Nominal
        Text fields
        Printing
        Saving and loading
    """
    def setUp(self):
        self.gui_settings = bac.GuiSettings()
        self.filename = os.path.join(bac.get_root_dir(), 'TestGuiSettingsFile.pkl')

    def test_nominal(self):
        self.assertTrue(isinstance(self.gui_settings, bac.GuiSettings))

    def test_text_field(self):
        fields = bac.GuiSettings.get_text_fields()
        self.assertEqual(fields, ['height', 'weight', 'age', 'bmi', 'hr1', 'hr2', 'hr3', 'hr4', 'hr5', 'hr6'])

    def test_printing(self):
        text = str(self.gui_settings)
        self.assertTrue(text.startswith('GuiSettings:\n    age: -1\n'))
        self.assertTrue(text.endswith('\n    weight: -1'))

    def test_save_and_load(self):
        self.gui_settings.save(self.filename)
        gui_settings2 = bac.GuiSettings.load(self.filename)
        for key in vars(self.gui_settings).keys():
            self.assertEqual(getattr(self.gui_settings, key), getattr(gui_settings2, key))

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

#%% BacGui
class Test_BacGui(unittest.TestCase):
    r"""
    Tests the BacGui with the following cases:
        TBD
    """
    def _reset(self):
        self.gui.gui_settings = bac.GuiSettings()
        self.gui.wrapper()

    def _compare(self, gui_settings):
        equal = True
        for key in vars(self.gui_settings).keys():
            if getattr(self.gui_settings, key) != getattr(gui_settings, key):
                equal = False
        return equal

    def test_sequence(self):
        # instantiate the GUI
        self.gui = bac.BacGui()
        # copy the original state for reference
        self.gui_settings = copy.deepcopy(self.gui.gui_settings)
        if False:
            # Create a new profile
            self.gui.onActivated(self.gui.popup_profile.count()-1)
            # Enter a name for the new profile
            #QTest.keyClicks(widget, "Temp 1") TODO: make this work!
            #QtCore.QTimer.singleShot(0, button.clicked)

        # Update the height
        self.gui.lne_height.clear()
        QTest.keyClicks(self.gui.lne_height, "69")
        QTest.keyClick(self.gui.lne_height, QtCore.Qt.Key_Enter)
        self.assertAlmostEqual(self.gui.gui_settings.height, 69)

        # Check that BMI was not updated
        self.assertEqual(self.gui.gui_settings.bmi, bac.GUI_TOKEN)

        # Update the weight
        QTest.keyClicks(self.gui.lne_weight, "165")
        QTest.keyClick(self.gui.lne_weight, QtCore.Qt.Key_Enter)
        self.assertAlmostEqual(self.gui.gui_settings.weight, 165)

        # Check that BMI was updated
        self.assertNotEqual(self.gui.gui_settings.bmi, bac.GUI_TOKEN)

        # Reupdate weight without changing BMI again
        old = self.gui.gui_settings.bmi
        self.gui.lne_weight.clear()
        QTest.keyClicks(self.gui.lne_weight, "161")
        QTest.keyClick(self.gui.lne_weight, QtCore.Qt.Key_Enter)
        self.assertAlmostEqual(self.gui.gui_settings.weight, 161)
        self.assertAlmostEqual(old, self.gui.gui_settings.bmi)

        # Change gender
        self.assertEqual(self.gui.gui_settings.gender, bac.Gender.female)
        QTest.mouseClick(self.gui.radio_male, QtCore.Qt.LeftButton)
        self.assertEqual(self.gui.gui_settings.gender, bac.Gender.male)

        # Update age
        self.assertEqual(self.gui.gui_settings.age, bac.GUI_TOKEN)
        self.gui.lne_age.clear()
        QTest.keyClicks(self.gui.lne_age, "34")
        QTest.keyClick(self.gui.lne_age, QtCore.Qt.Key_Enter)
        self.assertAlmostEqual(self.gui.gui_settings.age, 34)

        # Update to bad value
        self.gui.lne_age.clear()
        QTest.keyClicks(self.gui.lne_age, "34.abc")
        QTest.keyClick(self.gui.lne_age, QtCore.Qt.Key_Enter)
        self.assertAlmostEqual(self.gui.gui_settings.age, 34)

        # Update age back to empty
        self.assertEqual(self.gui.gui_settings.age, 34)
        self.gui.lne_age.clear()
        QTest.keyClick(self.gui.lne_age, QtCore.Qt.Key_Enter)
        self.assertAlmostEqual(self.gui.gui_settings.age, bac.GUI_TOKEN)

        # Click again for no change
        QTest.mouseClick(self.gui.radio_male, QtCore.Qt.LeftButton)
        self.assertEqual(self.gui.gui_settings.gender, bac.Gender.male)

        # Change gender back
        QTest.mouseClick(self.gui.radio_fmal, QtCore.Qt.LeftButton)
        self.assertEqual(self.gui.gui_settings.gender, bac.Gender.female)

        # Click again for no change
        QTest.mouseClick(self.gui.radio_fmal, QtCore.Qt.LeftButton)
        self.assertEqual(self.gui.gui_settings.gender, bac.Gender.female)

        # press save button
        QTest.mouseClick(self.gui.btn_save, QtCore.Qt.LeftButton)

        # press plot button
        QTest.mouseClick(self.gui.btn_plot, QtCore.Qt.LeftButton)

        # More popup tests
        if False:
            pass
            # Change again to "Temp 2", but choose cancel
            # TODO:

            # Change again, but enter empty text and choose ok
            # TODO:

            # CHange again, this time with "Temp 2"
            # TODO:

            # Change again, this time with "Temp 1" and it already exists, check that restores values
            # TODO: write this, and also capture the output or change the print statement

    def tearDown(self):
        QApplication.instance().closeAllWindows()
        folder = bac.get_root_dir()
        files = ['Temp 1.pkl', 'Temp 2.pkl', 'Default.pkl', 'BAC vs. Time for Temp 1.png']
        for file in files:
            filename = os.path.join(folder, file)
            if os.path.isfile(filename):
                os.remove(filename)

#%% get_root_dir
class Test_get_root_dir(unittest.TestCase):
    r"""
    Tests the get_root_dir function with these cases:
        call the function
    """
    def test_function(self):
        filepath      = inspect.getfile(bac.get_root_dir)
        expected_root = os.path.split(filepath)[0]
        folder = bac.get_root_dir()
        self.assertEqual(folder, expected_root)
        self.assertTrue(os.path.isdir(folder))

#%% calculate_bmi
class Test_calculate_bmi(unittest.TestCase):
    r"""
    Tests the calculate_bmi function the following cases:
        Default conv
        Specified conv
    """
    def setUp(self):
        self.height = 69
        self.weight = 161
        self.gender = bac.Gender.male
        self.bmi    = 23.77532753623188
        self.bmi2   = 0.033816425120772944

    def test_default_conv(self):
        bmi = bac.calculate_bmi(self.height, self.weight, self.gender)
        self.assertAlmostEqual(bmi, self.bmi)

    def test_specified_conv(self):
        bmi = bac.calculate_bmi(self.height, self.weight, self.gender, conv=1)
        self.assertAlmostEqual(bmi, self.bmi2)

#%% calculate_bac
class Test_calculate_bac(unittest.TestCase):
    r"""
    Tests the calculate_bac function with the following cases:
        Nominal
        Add time to beginning
        Add time to end
    """
    def setUp(self):
        self.time_drinks = np.array([1, 2, 3, 4, 5, 6])
        self.drinks = np.array([1, 1.5, 2.2, 0.5, 0, 0])
        self.time_out = self.time_drinks.copy()
        self.body_weight = 105
        self.bac_func = lambda drinks, weight, time: drinks.cumsum() / weight * 0.0375 - 0.00015*time
        self.bac_ = self.drinks.cumsum() / self.body_weight * 0.0375 - 0.00015*self.time_out

    def test_nominal(self):
        bac1 = bac.calculate_bac(self.time_drinks, self.drinks, self.time_out, self.body_weight)
        bac2 = self.bac_func(self.drinks, self.body_weight, self.time_drinks)
        np.testing.assert_array_almost_equal(bac1, bac2)

    def test_no_t_zero(self):
        time_out = np.array([0, 1, 2, 3, 4, 5, 6])
        bac1 = bac.calculate_bac(self.time_drinks, self.drinks, time_out, self.body_weight)
        bac2 = self.bac_func(self.drinks, self.body_weight, self.time_drinks)
        self.assertEqual(bac1[0], 0)
        np.testing.assert_array_almost_equal(bac1[1:], bac2)

    def test_no_t_final(self):
        time_out = np.array([1, 2, 3, 4, 5, 6, 7])
        bac1 = bac.calculate_bac(self.time_drinks, self.drinks, time_out, self.body_weight)
        bac2 = self.bac_func(self.drinks, self.body_weight, self.time_drinks)
        self.assertEqual(bac1[-1], self.bac_func(np.sum(self.drinks), self.body_weight, 7))
        np.testing.assert_array_almost_equal(bac1[:-1], bac2)

#%% plot_bac
class Test_plot_bac(unittest.TestCase):
    r"""
    Tests the plot_bac function with the following cases:
        Nominal
        With legal limit
    """
    def setUp(self):
        self.gui_settings = bac.GuiSettings()
        self.gui_settings.height = 69
        self.gui_settings.weight = 161
        self.gui_settings.age    = 34
        self.gui_settings.bmi    = 23.78
        self.gui_settings.gender = bac.Gender.male
        self.fig = None

    def test_nominal(self):
        self.fig = bac.plot_bac(self.gui_settings)

    def test_legal_limit(self):
        self.fig = bac.plot_bac(self.gui_settings, legal_limit=0.0008)

    def tearDown(self):
        plt.close(self.fig)

#%% Unit test execution
if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    unittest.main(exit=False)
    # close the qapp
    qapp.closeAllWindows()
