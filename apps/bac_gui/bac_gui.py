# -*- coding: utf-8 -*-
r"""
The main module file for the BAC GUI.  It defines the GUI and it's behavior and plotting.

Notes
-----
#.  Written by David C. Stauffer in June 2016.
"""

#%% Imports
# normal imports
import doctest
from enum import Enum, unique
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import unittest
# Qt imports
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QComboBox, QFormLayout, QGridLayout, QGroupBox, \
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QRadioButton, QToolTip, QWidget, \
    QVBoxLayout

#%% Constants
GUI_TOKEN   = -1
LEGAL_LIMIT = 0.08/100
BMI_CONV    = 703.0704

#%% Classes - Gender
@unique
class Gender(Enum):
    r"""
    Enumerator definitions for the possible gender conditions.
    """
    male   = 1 # uncircumcised male
    female = 2 # female

#%% Classes - GuiSettings
class GuiSettings(object):
    r"""
    Settings that capture the current state of the GUI.
    """
    def __init__(self):
        self.profile     = 'Default'
        self.height      = GUI_TOKEN
        self.weight      = GUI_TOKEN
        self.gender      = Gender.female
        self.age         = GUI_TOKEN
        self.bmi         = GUI_TOKEN
        self.hr1         = 0
        self.hr2         = 0
        self.hr3         = 0
        self.hr4         = 0
        self.hr5         = 0
        self.hr6         = 0

    def __str__(self):
        r"""Prints all the settings out."""
        text = ['GuiSettings:']
        for key in sorted(vars(self)):
            text.append('    {}: {}'.format(key, getattr(self, key)))
        return '\n'.join(text)

    @staticmethod
    def get_text_fields():
        r"""Returns the names of all the line edit widgets"""
        return ['height', 'weight', 'age', 'bmi', 'hr1', 'hr2', 'hr3', 'hr4', 'hr5', 'hr6']

    @staticmethod
    def load(filename):
        r"""Loads a instance of the class from a given filename."""
        with open(filename, 'rb') as file:
            gui_settings = pickle.load(file)
        assert isinstance(gui_settings, GuiSettings)
        return gui_settings

    def save(self, filename):
        r"""Saves an instance of the class to the given filename."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

#%% Classes - BacGui
class BacGui(QMainWindow):
    r"""
    The BAC GUI.
    """
    # Create GUI setting defaults for the class
    gui_settings = GuiSettings()

    def __init__(self):
        # call super method
        super(BacGui, self).__init__()
        # initialize the state data
        #self.initialize_state(filename, board, cur_move, cur_game, game_hist)
        # call init method to instantiate the GUI
        self.init()

    # GUI initialization
    def init(self):
        r"""Initializes the GUI."""
        # initialize timer
        self.timer = QtCore.QTimer(self)

        # properties
        QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        # Central Widget
        self.gui_widget = QWidget(self)
        self.setCentralWidget(self.gui_widget)

        # Panels (group boxes)
        self.grp_profile = QGroupBox('Profile')
        self.grp_consump = QGroupBox('Consumption')
        self.grp_plotter = QWidget()

        # Layouts
        layout_gui     = QHBoxLayout(self.gui_widget)
        layout_profile = QGridLayout(self.grp_profile)
        layout_consump = QFormLayout(self.grp_consump)
        layout_plotter = QVBoxLayout(self.grp_plotter)

        # Labels
        lbl_profile  = QLabel('Profile:')
        lbl_height   = QLabel('Height:')
        lbl_weight   = QLabel('Weight:')
        lbl_age      = QLabel('Age:')
        lbl_bmi      = QLabel('BMI:')
        lbl_gender   = QLabel('Gender:')

        lbl_hr1      = QLabel('Hour 1:')
        lbl_hr2      = QLabel('Hour 2:')
        lbl_hr3      = QLabel('Hour 3:')
        lbl_hr4      = QLabel('Hour 4:')
        lbl_hr5      = QLabel('Hour 5:')
        lbl_hr6      = QLabel('Hour 6:')

        lbl_drink = QLabel('One Drink is:\n1 oz of 100 proof\n5 oz of wine\n12 oz of regular beer')

        # Fields
        self.popup_profile = QComboBox()
        profiles = self.initialize_profiles()
        for this_profile in profiles:
            self.popup_profile.addItem(this_profile)
        self.popup_profile.setCurrentIndex(0)
        self.popup_profile.activated.connect(self.onActivated)

        self.lne_height = QLineEdit('')
        self.lne_weight = QLineEdit('')
        self.lne_age    = QLineEdit('')
        self.lne_bmi    = QLineEdit('')

        self.radio_gender = QWidget()
        layout_gender   = QHBoxLayout(self.radio_gender)
        self.radio_fmal = QRadioButton('Female')
        self.radio_fmal.setChecked(True)
        self.radio_fmal.toggled.connect(self.radio_toggle)
        self.radio_male = QRadioButton('Male')
        self.radio_male.toggled.connect(self.radio_toggle)
        layout_gender.addWidget(self.radio_fmal)
        layout_gender.addWidget(self.radio_male)

        self.lne_hr1 = QLineEdit('')
        self.lne_hr2 = QLineEdit('')
        self.lne_hr3 = QLineEdit('')
        self.lne_hr4 = QLineEdit('')
        self.lne_hr5 = QLineEdit('')
        self.lne_hr6 = QLineEdit('')

        lnes = [getattr(self, 'lne_' + field) for field in self.gui_settings.get_text_fields()]
        for this_lne in lnes:
            this_lne.setAlignment(QtCore.Qt.AlignCenter)
            this_lne.editingFinished.connect(self.text_finished)

        # Buttons - Save Profile button
        self.btn_save = QPushButton('Save Profile')
        self.btn_save.setToolTip('Saves the current profile to disk.')
        self.btn_save.setMaximumWidth(120)
        self.btn_save.setStyleSheet('color: black; background-color: #00bfbf; font: bold;')
        self.btn_save.clicked.connect(self.btn_save_function)
        # Buttons - Plot button
        self.btn_plot = QPushButton('Plot')
        self.btn_plot.setToolTip('Plots the BAC over time with the given information.')
        self.btn_plot.setMaximumWidth(200)
        self.btn_plot.setStyleSheet('color: black; background-color: #009900; font: bold;')
        self.btn_plot.clicked.connect(self.btn_plot_function)

        # Populate widgets - profile
        layout_profile.addWidget(lbl_profile, 0, 0)
        layout_profile.addWidget(lbl_height, 1, 0)
        layout_profile.addWidget(lbl_weight, 2, 0)
        layout_profile.addWidget(lbl_age, 3, 0)
        layout_profile.addWidget(lbl_bmi, 4, 0)
        layout_profile.addWidget(lbl_gender, 5, 0)
        layout_profile.addWidget(self.popup_profile, 0, 1)
        layout_profile.addWidget(self.lne_height, 1, 1)
        layout_profile.addWidget(self.lne_weight, 2, 1)
        layout_profile.addWidget(self.lne_age, 3, 1)
        layout_profile.addWidget(self.lne_bmi, 4, 1)
        layout_profile.addWidget(self.radio_gender, 5, 1)
        layout_profile.addWidget(self.btn_save, 6, 0, 1, 2, QtCore.Qt.AlignCenter)

        # Populate widgets - consumption
        layout_consump.addRow(lbl_hr1, self.lne_hr1)
        layout_consump.addRow(lbl_hr2, self.lne_hr2)
        layout_consump.addRow(lbl_hr3, self.lne_hr3)
        layout_consump.addRow(lbl_hr4, self.lne_hr4)
        layout_consump.addRow(lbl_hr5, self.lne_hr5)
        layout_consump.addRow(lbl_hr6, self.lne_hr6)

        # Populate widgets - plotter
        layout_plotter.addWidget(lbl_drink)
        layout_plotter.addWidget(self.btn_plot)

        # Populate widgets - main GUI
        layout_gui.addWidget(self.grp_profile)
        layout_gui.addWidget(self.grp_consump)
        layout_gui.addWidget(self.grp_plotter)

        # Call wrapper to initialize GUI
        self.wrapper()

        # GUI final layout properties
        self.center()
        self.setWindowTitle('BAC GUI')
        self.setWindowIcon(QtGui.QIcon(os.path.join(get_root_dir(), 'bac_gui.png')))
        self.show()

    #%% Other initializations
    def initialize_profiles(self):
        r"""Gets the list of all current profiles that exist in the folder."""
        # Check to see if the Default profile exists, and if so load it, else create it
        folder = get_root_dir()
        filename = os.path.join(folder, 'Default.pkl')
        if os.path.isfile(filename): # pragma: no cover
            self.gui_settings = GuiSettings.load(filename)
        else: # pragma: no cover
            self.gui_settings.save(filename)
        # Find all the pickle files that exist, and make them into profiles
        profiles = glob.glob(os.path.join(folder, '*.pkl'))
        profiles = [os.path.normpath(x).split(os.path.sep)[-1][:-4] for x in profiles]
        profiles = set(profiles) ^ {'Default'}
        profiles = ['Default'] + sorted(profiles) + ['New+']
        return profiles

    #%% wrapper
    def wrapper(self):
        r"""
        Acts as a wrapper to everything the GUI needs to do.
        """
        # Note: nothing is done to update the profile field, it's assumed to correct already
        # loop through and update the text fields
        for field in self.gui_settings.get_text_fields():
            this_value = getattr(self.gui_settings, field)
            this_lne = getattr(self, 'lne_' + field)
            if this_value == GUI_TOKEN:
                this_lne.setText('')
            else:
                this_lne.setText('{:g}'.format(this_value))
        # update the gender button group
        if self.gui_settings.gender == Gender.female:
            if self.radio_male.isChecked():
                self.radio_fmal.setChecked(True)
        elif self.gui_settings.gender == Gender.male:
            if self.radio_fmal.isChecked():
                self.radio_male.setChecked(True)
        else: # pragma: no cover
            raise ValueError('Unexpected value for gender: "{}".'.format(self.gui_settings.gender))

    #%% Other callbacks - closing
    def closeEvent(self, event):
        r"""Things in here happen on GUI closing."""
        event.accept()

    #%% Other callbacks - center the GUI on the screen
    def center(self):
        r"""Makes the GUI centered on the active screen."""
        frame_gm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_gm.moveCenter(center_point)
        self.move(frame_gm.topLeft())

    #%% Other callbacks - dislaying an error for invalid edit box entries
    def display_text_error(self, field):
        r"""Displays a temporary message for invalid characters within the line edit boxes."""
        field.setStyleSheet('color: white; background-color: red; font: bold;')
        reset = lambda: field.setStyleSheet('color: black; background-color: white; font: normal;')
        self.timer.setInterval(300)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(reset)
        self.timer.start()
        self.wrapper()

    #%% Other callbacks - updating the selected profile name
    def onActivated(self, value):
        r"""Controls behavior when mode combobox is changed."""
        # check if "New+" was selected
        if value == len(self.popup_profile) - 1:
            # get the current items
            items = [self.popup_profile.itemText(i) for i in range(self.popup_profile.count())]
            # ask for a new profile
            text, ok = QtGui.QInputDialog.getText(self, 'Profile Name',  'Enter a new profile:')
            if not ok or not text:
                # put back to last choice
                ix = items.index(self.gui_settings.profile)
                self.popup_profile.setCurrentIndex(ix)
            else:
                if text in items:
                    # if already an existing profile, then load the old one
                    print('Profile "{}" already exists and is being loaded.'.format(text))
                    self.gui_settings = self.gui_settings.load(os.path.join(get_root_dir(), text + '.pkl'))
                else:
                    # create the new profile
                    gui_settings = GuiSettings()
                    gui_settings.profile = text
                    # if successful in saving, then update the working copy
                    gui_settings.save(os.path.join(get_root_dir(), text + '.pkl'))
                    self.gui_settings = gui_settings
                    # find where to insert in GUI and insert
                    i = 1
                    while i < len(items)-1 and items[i] < text:
                        i += 1
                    self.popup_profile.insertItem(i, text)
                    self.popup_profile.setCurrentIndex(i)
        else:
            # changed to the desired existing profile
            text = self.popup_profile.currentText()
            if text != self.gui_settings.profile:
                self.gui_settings = self.gui_settings.load(os.path.join(get_root_dir(), text + '.pkl'))
        # update the GUI to reflect any new settings
        self.wrapper()

    #%% Other callbacks - update the line edit boxes
    def text_finished(self):
        r"""Updates gui_settings for LineEdit text changes that happen when you leave the box."""
        sender  = self.sender()
        fields  = self.gui_settings.get_text_fields()
        senders = [getattr(self, 'lne_' + field) for field in fields]
        for ix in range(len(senders)):
            if sender == senders[ix]:
                text = sender.text()
                if text == '':
                    setattr(self.gui_settings, fields[ix], GUI_TOKEN)
                    break
                try:
                    value = float(text)
                except ValueError:
                    self.display_text_error(senders[ix])
                    break
                else:
                    setattr(self.gui_settings, fields[ix], value)
                    break
        else: # pragma: no cover
            raise ValueError('Unexpected field went into this method.')

        # check for conditions to update the BMI
        if sender == self.lne_height or sender == self.lne_weight:
            if self.gui_settings.height != GUI_TOKEN and self.gui_settings.weight != GUI_TOKEN:
                if self.gui_settings.bmi == GUI_TOKEN:
                    self.gui_settings.bmi = calculate_bmi(self.gui_settings.height, self.gui_settings.weight, \
                        self.gui_settings.gender, BMI_CONV)
        # call the wrapper to update all the possible field changes
        self.wrapper()

    #%% Other callbacks - Updating the gender button group
    def radio_toggle(self):
        r"""Controls the gender radio button group."""
        # assert that only one of the button group is checked
        assert self.radio_fmal.isChecked() ^ self.radio_male.isChecked(), 'Only one button may be checked.'
        # determine which button is checked and update the settings accordingly
        if self.radio_fmal.isChecked():
            self.gui_settings.gender = Gender.female
        elif self.radio_male.isChecked(): # pragma: no branch
            self.gui_settings.gender = Gender.male
        self.wrapper()

    #%% Other callbacks - Save button
    def btn_save_function(self):
        r"""Saves the current settings to the specified profile."""
        # save the profile
        self.gui_settings.save(os.path.join(get_root_dir(), self.gui_settings.profile + '.pkl'))

    #%% Other callbacks - Plot button
    def btn_plot_function(self):
        r"""Plots the results and saves to a .png file."""
        # call the plotting function
        fig = plot_bac(self.gui_settings, LEGAL_LIMIT)
        # save the figure
        filename = os.path.join(get_root_dir(), fig.canvas.get_window_title() + '.png')
        fig.savefig(filename, dpi=160, bbox_inches='tight')

#%% Functions - get_root_dir
def get_root_dir():
    r"""
    Returns the folder that contains this source file and thus the root folder for the whole code.

    Returns
    -------
    folder : str
        Location of the folder that contains all the source files for the code.

    Examples
    --------

    >>> from dstauffman.apps.bac_gui import get_root_dir
    >>> folder = get_root_dir()

    """
    # this folder is the root directory based on the location of this file (utils.py)
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return folder

#%% Functions - calculate_bmi
def calculate_bmi(height, weight, gender, conv=BMI_CONV):
    r"""
    Calculates the BMI (Body Mass Index) for someone based on their height and weight (and maybe
    eventually gender).

    Parameters
    ----------
    height : float
        Height in inches
    weight : float
        Weight in pounds
    gender : class Gender
        Gender
    conv : float, optional
        Unit conversion factor

    Returns
    -------
    bmi : float
        Body mass index

    Examples
    --------

    >>> from dstauffman.apps.bac_gui import calculate_bmi, Gender
    >>> height = 69
    >>> weight = 161
    >>> gender = Gender.male
    >>> bmi = calculate_bmi(height, weight, gender)
    >>> print('{:.2f}'.format(bmi))
    23.78

    """
    # calculate the BMI using a simple formula (could be expanded later)
    bmi = weight / height**2 * conv
    return bmi

#%% Functions - calculate_bac
def calculate_bac(time_drinks, drinks, time_out, body_weight):
    r"""
    Calculates a BAC (Blood Alcohol Content) over time.

    Examples
    --------

    >>> from dstauffman.apps.bac_gui import calculate_bac
    >>> import numpy as np
    >>> time_drinks = np.array([1, 2, 3, 4, 5, 6])
    >>> drinks = np.array([1, 1.5, 2.2, 0.5, 0, 0])
    >>> time_out = time_drinks.copy()
    >>> body_weight = 105
    >>> bac = calculate_bac(time_drinks, drinks, time_out, body_weight)
    >>> print(bac) # doctest: +NORMALIZE_WHITESPACE
    [ 0.00020714  0.00059286  0.00122857  0.00125714  0.00110714  0.00095714]

    """
    # hard-coded values
    drink_weight_conv = 0.0375 # converts standard drinks consumed per pound to BAC
    burn_up = 0.00015 # alcohol content burned up per hour

    # potentially expand time and data vectors
    if time_drinks[0] > time_out[0]:
        time_drinks = np.append(time_out[0], time_drinks)
        drinks = np.append(0, drinks)
    if time_drinks[-1] < time_out[-1]:
        time_drinks = np.append(time_drinks, np.inf)
        drinks = np.append(drinks, drinks[-1])

    # find the cumulative amount of drinks consumed
    cum_drinks = np.cumsum(drinks)

    ## find the BAC assuming no alcohol was converted
    bac_init = cum_drinks / body_weight * drink_weight_conv

    # interpolate the BAC to the desired time, still assuming no alcohol was converted
    bac_interp = np.interp(time_out, time_drinks, bac_init)

    # subtract off the amount that was converted by the body in the given time
    bac = np.maximum(bac_interp - burn_up * time_out, 0)

    return bac

#%% Function - plot_bac
def plot_bac(gui_settings, legal_limit=None):
    r"""
    Plots the BAC over time.

    Parameters
    ----------
    gui_settings : class GuiSettings
        GUI settings
    legal_limit : float, optional
        Legal limit for BAC before considered impaired and unable to drive

    Returns
    -------
    fig : class matplotlib.Figure
        Figure handle

    Examples
    --------

    >>> from dstauffman.apps.bac_gui import GuiSettings, plot_bac, Gender
    >>> import matplotlib.pyplot as plt
    >>> gui_settings = GuiSettings()
    >>> gui_settings.height = 69
    >>> gui_settings.weight = 161
    >>> gui_settings.age    = 34
    >>> gui_settings.bmi    = 23.78
    >>> gui_settings.gender = Gender.male
    >>> fig = plot_bac(gui_settings)

    Close the figure
    >>> plt.close(fig)

    """
    #% hard-coded values
    time_drinks = np.array([1, 2, 3, 4, 5, 6])
    time_out    = np.linspace(0, 12, 1000)
    ratio2per   = 100

    # check inputs
    assert isinstance(gui_settings, GuiSettings)

    # pull out information from gui_settings
    drinks      = np.array([gui_settings.hr1, gui_settings.hr2, gui_settings.hr3, \
        gui_settings.hr4, gui_settings.hr5, gui_settings.hr6])
    body_weight = gui_settings.weight
    name        = gui_settings.profile

    # calculate the BAC
    bac = ratio2per * calculate_bac(time_drinks, drinks, time_out, body_weight)

    # turn interactive plotting off
    plt.ioff()

    # create the figure and axis
    fig = plt.figure(facecolor='w')
    this_title = 'BAC vs. Time for {}'.format(name)
    fig.canvas.set_window_title(this_title)
    ax = fig.add_subplot(111)

    # plot the data
    ax.plot(time_out, bac, '.-', label='BAC')
    if legal_limit is not None:
        ax.plot(np.array([time_out[0], time_out[-1]]), ratio2per*legal_limit*np.ones(2), '--', \
            label='Legal Limit', color='red', linewidth=2)

    # add some labels and such
    plt.title(this_title)
    plt.xlabel('Time [hr]')
    plt.ylabel('BAC [%]')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    return fig

#%% Unit Test
if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    unittest.main(module='dstauffman.apps.bac_gui.test_bac_gui', exit=False)
    doctest.testmod(verbose=False)
    # close the qapp
    qapp.closeAllWindows()
