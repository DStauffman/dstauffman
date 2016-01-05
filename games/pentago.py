# -*- coding: utf-8 -*-
r"""
Pentago board game as a Python GUI.

Notes
-----
#.  Written by David C. Stauffer in MATLAB in January 2010, translated to Python in December 2015.
"""

#%% Imports
import doctest
from matplotlib.figure import Figure, SubplotParams
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np
import os
import pickle
import sys
import unittest
try:
    from PyQt5 import QtGui, QtCore
    from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QLabel, QMessageBox
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError: # pragma: no cover
    from PyQt4 import QtGui, QtCore
    from PyQt4.QtGui import QApplication, QWidget, QToolTip, QPushButton, QLabel, QMessageBox
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from dstauffman     import get_images_dir, get_data_dir, Frozen

#%% Constants
# color definitions
COLOR             = {}
COLOR['board']    = (1., 1., 0.)
COLOR['win']      = (1., 0., 0.)
COLOR['white']    = (1., 1., 1.)
COLOR['black']    = (0., 0., 0.)
COLOR['maj_edge'] = (0., 0., 0.)
COLOR['min_edge'] = (0., 0., 1.)
COLOR['button']   = (1., 1., 0.75)
COLOR['gui_bkgd'] = (225./255, 224./255, 228./255)
COLOR['next_wht'] = (0.6, 0.6, 1.0) # (0.8, 0.8, 0.8)
COLOR['next_blk'] = (0.0, 0.0, 0.4) # (0.4, 0.4, 0.4)
COLOR['redo']     = (1.0, 0.0, 1.0)
COLOR['win_wht']  = (1.0, 0.9, 0.9) # (1.0, 0.9, 0.9)
COLOR['win_blk']  = (0.2, 0.0, 0.0) # (0.6, 0.0, 0.0)
COLOR['win_tie']  = (1.0, 0.0, 0.6)
COLOR['rot_wht']  = (0.0, 1.0, 1.0) # COLOR['next_wht']
COLOR['rot_blk']  = (0.0, 0.0, 0.5) # COLOR['next_blk']
COLOR['rot_grey'] = (0.6, 0.6, 0.6)
COLOR['rot_move'] = (0.0, 0.0, 1.0)
COLOR['rot_undo'] = (0.6, 0.6, 1.0)

# player enumerations
PLAYER          = {}
PLAYER['white'] = 1
PLAYER['black'] = -1
PLAYER['none']  = 0
PLAYER['draw']  = 2

# sizes of the different pieces and squares
RADIUS           = {}
RADIUS['piece']  = 0.45;
RADIUS['next']   = 0.35;
RADIUS['move']   = 0.30;
RADIUS['win']    = 0.25;
RADIUS['square'] = 0.5;

# Gameplay options
OPTIONS = {}
OPTIONS['load_previous_game'] = 'Yes' # from ['Yes','No','Ask']
OPTIONS['name_white'] = 'Player 1'
OPTIONS['name_black'] = 'Player 2'
OPTIONS['plot_winning_moves'] = True

# all possible winning combinations
WIN = np.array([\
    [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],\
    [1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],\
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],\
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0],\
    \
    [0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0],\
    [0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0],\
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0],\
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0],\
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0],\
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],\
    \
    [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],\
    [0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],\
    [0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0],\
    [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,0],\
    [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],\
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],\
    \
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],\
    [0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0],\
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0],\
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1],\
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0],\
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],\
    \
    [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],\
    [0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0],\
    [0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],\
    [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],\
    [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0],\
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0],\
    \
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],\
    [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],\
    [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],\
    [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],\
    [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],\
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],\
], dtype=bool)

# boolean flag for whether to log extra information or not
LOGGING = True

#%% Classes - Moves
class Move(Frozen):
    r"""
    Class that keeps track of each individual move.
    """
    def __init__(self, row, column, quadrant, direction):
        self.row       = row
        self.column    = column
        self.quadrant  = quadrant
        self.direction = direction

    def __str__(self):
        return 'row: {}, col: {}, quad: {}, dir: {}'.format(self.row, self.column, self.quadrant, self.direction)

#%% Classes - GameStats
class GameStats(Frozen):
    r"""
    Class that keeps track of all the moves in a game.
    """
    def __init__(self, number, first_move, winner=PLAYER['none'], move_list=None):
        self.number     = number
        self.first_move = first_move
        self.winner     = winner
        if move_list is None:
            self.move_list = []
        else:
            self.move_list = move_list

    def add_move(self, move):
        r"""Adds the given move to the game move history."""
        assert isinstance(move, Move), 'The specified move must be an instance of class Move.'
        self.move_list.append(move)

    def remove_moves(self, cur_move=None):
        r"""Removes the moves from the current move number to the end of the list."""
        if cur_move is None:
            self.move_list.pop()
        else:
            del(self.move_list[cur_move:])

    @property
    def num_moves(self):
        return len(self.move_list)

    @staticmethod
    def save(filename, game_hist):
        r"""Saves a list of GameStats objects to disk."""
        with open(filename, 'wb') as file:
            pickle.dump(game_hist, file)

    @staticmethod
    def load(filename):
        r"""Loads a list of GameStats objects to disk."""
        with open(filename, 'rb') as file:
            game_hist = pickle.load(file)
        return game_hist

#%% Dynamic globals
cur_move    = 0
cur_game    = 0
board    = PLAYER['none'] * np.ones((6, 6), dtype=int)
move_status = {'ok': False, 'pos': None, 'patch_object': None}
game_hist   = []
game_hist.append(GameStats(number=cur_game, first_move=PLAYER['white']))

#%% Classes - PentagoGui
class PentagoGui(QWidget):
    r"""
    The Pentago GUI.
    """
    def __init__(self, **kwargs):
        super(PentagoGui, self).__init__(**kwargs)
        self.init()

    def init(self):
        r"""Creates the actual GUI."""

        #%% properties
        QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        #%% Icons
        icon1R = QtGui.QIcon(os.path.join(get_images_dir(), 'right1.png'))
        icon2R = QtGui.QIcon(os.path.join(get_images_dir(), 'right2.png'))
        icon3R = QtGui.QIcon(os.path.join(get_images_dir(), 'right3.png'))
        icon4R = QtGui.QIcon(os.path.join(get_images_dir(), 'right4.png'))
        icon1L = QtGui.QIcon(os.path.join(get_images_dir(), 'left1.png'))
        icon2L = QtGui.QIcon(os.path.join(get_images_dir(), 'left2.png'))
        icon3L = QtGui.QIcon(os.path.join(get_images_dir(), 'left3.png'))
        icon4L = QtGui.QIcon(os.path.join(get_images_dir(), 'left4.png'))

        #%% Text
        # Pentago
        lbl_pentago = QLabel('Pentago', self)
        lbl_pentago.setGeometry(360, 51, 220, 40)
        lbl_pentago.setAlignment(QtCore.Qt.AlignCenter)
        lbl_pentago.setStyleSheet('font-size: 18pt; font: bold;')
        # Score
        lbl_score = QLabel('Score:', self)
        lbl_score.setGeometry(50, 220, 220, 40)
        lbl_score.setAlignment(QtCore.Qt.AlignCenter)
        lbl_score.setStyleSheet('font-size: 12pt; font: bold;')
        # Move
        lbl_move = QLabel('Move:', self)
        lbl_move.setGeometry(740, 220, 220, 40)
        lbl_move.setAlignment(QtCore.Qt.AlignCenter)
        lbl_move.setStyleSheet('font-size: 12pt; font: bold;')
        # White Wins
        lbl_white = QLabel('White Wins:', self)
        lbl_white.setGeometry(50, 280, 80, 20)
        # Black Wins
        lbl_black = QLabel('Black Wins:', self)
        lbl_black.setGeometry(50, 310, 80, 20)
        # Games Tied
        lbl_games = QLabel('Games Tied:', self)
        lbl_games.setGeometry(50, 340, 80, 20)
        # Changeable labels
        self.lbl_white_wins = QLabel('0', self)
        self.lbl_white_wins.setGeometry(140, 280, 60, 20)
        self.lbl_white_wins.setAlignment(QtCore.Qt.AlignRight)
        self.lbl_black_wins = QLabel('0', self)
        self.lbl_black_wins.setGeometry(140, 310, 60, 20)
        self.lbl_black_wins.setAlignment(QtCore.Qt.AlignRight)
        self.lbl_games_tied = QLabel('0', self)
        self.lbl_games_tied.setGeometry(140, 340, 60, 20)
        self.lbl_games_tied.setAlignment(QtCore.Qt.AlignRight)

        #%% Axes
        self.wid_board = QWidget(self)
        self.wid_board.setGeometry(260, 140, 420, 420)
        # TODO: figure out how to kill the fucking frame
        fig = Figure(figsize=(4.2, 4.2), dpi=100, frameon=True, subplotpars=SubplotParams(left=0, \
            right=1, wspace=0, hspace=0))
        self.board_canvas = FigureCanvas(fig)
        self.board_canvas.setParent(self.wid_board)
        self.board_canvas.mpl_connect('button_release_event', lambda event: _mouse_click_callback(self, event))
        self.board_axes = fig.add_subplot(111, aspect='equal')
        self.board_axes.invert_yaxis()

        self.wid_move = QWidget(self)
        self.wid_move.setGeometry(780, 700-350-71, 70, 70)
        fig = Figure(figsize=(.7, .7), dpi=100)
        self.move_canvas = FigureCanvas(fig)
        self.move_canvas.setParent(self.wid_move)
        self.move_axes = fig.add_subplot(111, aspect='equal')

        #%% Buttons
        # Undo button
        self.btn_undo = QPushButton('Undo', self)
        self.btn_undo.setToolTip('Undoes the last move.')
        self.btn_undo.setGeometry(350, 600, 60, 30)
        self.btn_undo.setStyleSheet('color: yellow; background-color: #990000; font: bold;')
        self.btn_undo.clicked.connect(self.btn_undo_function)
        # New Game button
        self.btn_new = QPushButton('New Game', self)
        self.btn_new.setToolTip('Starts a new game.')
        self.btn_new.setGeometry(430, 600, 80, 50)
        self.btn_new.setStyleSheet('color: yellow; background-color: #006633; font: bold;')
        self.btn_new.clicked.connect(self.btn_new_function)
        # Redo button
        self.btn_redo = QPushButton('Redo', self)
        self.btn_redo.setToolTip('Redoes the last move.')
        self.btn_redo.setGeometry(530, 600, 60, 30)
        self.btn_redo.setStyleSheet('color: yellow; background-color: #000099; font: bold;')
        self.btn_redo.clicked.connect(self.btn_redo_function)

        # 1R button
        self.btn_1R = QPushButton('', self)
        self.btn_1R.setToolTip('Rotates quadrant 1 to the right 90 degrees.')
        self.btn_1R.setIcon(icon1R)
        self.btn_1R.setIconSize(QtCore.QSize(71, 71))
        self.btn_1R.setGeometry(260, 49, 71, 71)
        self.btn_1R.clicked.connect(self.btn_1R_function)
        # 2R button
        self.btn_2R = QPushButton('', self)
        self.btn_2R.setToolTip('Rotates quadrant 2 to the right 90 degrees.')
        self.btn_2R.setIcon(icon2R)
        self.btn_2R.setIconSize(QtCore.QSize(71, 71))
        self.btn_2R.setGeometry(700, 139, 71, 71)
        self.btn_2R.clicked.connect(self.btn_2R_function)
        # 3R button
        self.btn_3R = QPushButton('', self)
        self.btn_3R.setToolTip('Rotates quadrant 3 to the right 90 degrees.')
        self.btn_3R.setIcon(icon3R)
        self.btn_3R.setIconSize(QtCore.QSize(71, 71))
        self.btn_3R.setGeometry(170, 489, 71, 71)
        self.btn_3R.clicked.connect(self.btn_3R_function)
        # 4R button
        self.btn_4R = QPushButton('', self)
        self.btn_4R.setToolTip('Rotates quadrant 4 to the right 90 degrees.')
        self.btn_4R.setIcon(icon4R)
        self.btn_4R.setIconSize(QtCore.QSize(71, 71))
        self.btn_4R.setGeometry(610, 579, 71, 71)
        self.btn_4R.clicked.connect(self.btn_4R_function)
        # 1L button
        self.btn_1L = QPushButton('', self)
        self.btn_1L.setToolTip('Rotates quadrant 1 to the left 90 degrees.')
        self.btn_1L.setIcon(icon1L)
        self.btn_1L.setIconSize(QtCore.QSize(71, 71))
        self.btn_1L.setGeometry(170, 139, 71, 71)
        self.btn_1L.clicked.connect(self.btn_1L_function)
        # 2L button
        self.btn_2L = QPushButton('', self)
        self.btn_2L.setToolTip('Rotates quadrant 2 to the left 90 degrees.')
        self.btn_2L.setIcon(icon2L)
        self.btn_2L.setIconSize(QtCore.QSize(71, 71))
        self.btn_2L.setGeometry(610, 49, 71, 71)
        self.btn_2L.clicked.connect(self.btn_2L_function)
        # 3L button
        self.btn_3L = QPushButton('', self)
        self.btn_3L.setToolTip('Rotates quadrant 3 to the left 90 degrees.')
        self.btn_3L.setIcon(icon3L)
        self.btn_3L.setIconSize(QtCore.QSize(71, 71))
        self.btn_3L.setGeometry(260, 579, 71, 71)
        self.btn_3L.clicked.connect(self.btn_3L_function)
        # 4L button
        self.btn_4L = QPushButton('', self)
        self.btn_4L.setToolTip('Rotates quadrant 4 to the left 90 degrees.')
        self.btn_4L.setIcon(icon4L)
        self.btn_4L.setIconSize(QtCore.QSize(71, 71))
        self.btn_4L.setGeometry(700, 489, 71, 71)
        self.btn_4L.clicked.connect(self.btn_4L_function)

        #%% Call wrapper to initialize GUI
        wrapper(self)

        #%% GUI properties
        self.setGeometry(520, 380, 1000, 700)
        self.setWindowTitle('Pentago')
        self.setWindowIcon(QtGui.QIcon(os.path.join(get_images_dir(),'pentago.png')))
        self.show()

    #%% Other callbacks
    def closeEvent(self, event):
        """Things in here happen on GUI closing."""
        close_immediately = True
        filename = os.path.join(get_data_dir(), 'pentago.p')
        if close_immediately:
            GameStats.save(filename, game_hist)
            event.accept()
        else:
            # Alternative with user choice
            reply = QMessageBox.question(self, 'Message', \
                "Are you sure to quit?", QMessageBox.Yes | \
                QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                GameStats.save(filename, game_hist)
                event.accept()
            else:
                event.ignore()

    #%% Button callbacks
    def btn_undo_function(self):
        """Functions that executes on undo button press."""
        # declare globals
        global cur_move, cur_game, board
        # get last move
        last_move = game_hist[cur_game].move_list[cur_move-1]
        if LOGGING:
            print('Undoing move = {}'.format(last_move))
        # undo rotation
        _rotate_board(board, last_move.quadrant, -last_move.direction)
        # delete piece
        board[last_move.row, last_move.column] = PLAYER['none']
        # update current move
        cur_move -= 1
        # call GUI wrapper
        wrapper(self)

    def btn_new_function(self):
        """Functions that executes on new game button press."""
        # declare globals
        global cur_move, cur_game, board
        # update values
        last_lead = game_hist[cur_game].first_move
        next_lead = PLAYER['black'] if last_lead == PLAYER['white'] else PLAYER['white']
        assert len(game_hist) == cur_game + 1
        cur_game += 1
        cur_move = 0
        game_hist.append(GameStats(number=cur_game, first_move=next_lead, winner=PLAYER['none']))
        board = PLAYER['none'] * np.ones((6, 6), dtype=int)
        # call GUI wrapper
        wrapper(self)

    def btn_redo_function(self):
        """Functions that executes on redo button press."""
        # declare globals
        global cur_move, cur_game, board
        # get next move
        redo_move = game_hist[cur_game].move_list[cur_move]
        if LOGGING:
            print('Redoing move = {}'.format(redo_move))
        # place piece
        board[redo_move.row, redo_move.column] = _calc_cur_move(cur_move, cur_game)
        # redo rotation
        _rotate_board(board, redo_move.quadrant, redo_move.direction)
        # update current move
        cur_move += 1
        # call GUI wrapper
        wrapper(self)

    def btn_1R_function(self):
        """Functions that executes on 1R button press."""
        _execute_move(quadrant=1, direction=1)
        wrapper(self)

    def btn_2R_function(self):
        """Functions that executes on 2R button press."""
        _execute_move(quadrant=2, direction=1)
        wrapper(self)

    def btn_3R_function(self):
        """Functions that executes on 3R button press."""
        _execute_move(quadrant=3, direction=1)
        wrapper(self)

    def btn_4R_function(self):
        """Functions that executes on 4R button press."""
        _execute_move(quadrant=4, direction=1)
        wrapper(self)

    def btn_1L_function(self):
        """Functions that executes on 1L button press."""
        _execute_move(quadrant=1, direction=-1)
        wrapper(self)

    def btn_2L_function(self):
        """Functions that executes on 2L button press."""
        _execute_move(quadrant=2, direction=-1)
        wrapper(self)

    def btn_3L_function(self):
        """Functions that executes on 3L button press."""
        _execute_move(quadrant=3, direction=-1)
        wrapper(self)

    def btn_4L_function(self):
        """Functions that executes on 4L button press."""
        _execute_move(quadrant=4, direction=-1)
        wrapper(self)

#%% _mouse_click_callback
def _mouse_click_callback(self, event):
    r"""
    Function that executes on mouse click on the board axes.  Ends up placing a piece on the board.
    """
    # ignore events that are outside the axes
    if event.xdata is None or event.ydata is None:
        if LOGGING:
            print('Click is off the board.')
        return
    # test for a game that has already been concluded
    if game_hist[cur_game].winner != PLAYER['none']:
        if LOGGING:
            print('Game is over.')
        move_status['ok'] = False
        move_status['pos'] = None
        return
    # alias the rounded values of the mouse click location
    x = np.round(event.ydata).astype(int)
    y = np.round(event.xdata).astype(int)
    if LOGGING:
        print('Clicked on (x,y) = ({}, {})'.format(x, y))
    # get axes limits
    (m, n) = board.shape
    # ignore values that are outside the board
    if x < 0 or y < 0 or x >= m or y >= n:
        if LOGGING:
            print('Click is outside playable board.')
        return
    if board[x, y] == PLAYER['none']:
        # check for previous good move
        if move_status['ok']:
            if LOGGING:
                print('removing previous piece.')
            move_status['patch_object'].remove()
        move_status['ok'] = True
        move_status['pos'] = (x, y)
        if LOGGING:
            print('Placing current piece.')
        if _calc_cur_move(cur_move, cur_game) == PLAYER['white']:
            move_status['patch_object'] = _plot_piece(self.board_axes, x, y, RADIUS['piece'], COLOR['next_wht'])
        elif _calc_cur_move(cur_move, cur_game) == PLAYER['black']:
            move_status['patch_object'] = _plot_piece(self.board_axes, x, y, RADIUS['piece'], COLOR['next_blk'])
        else:
            raise ValueError('Unexpected player to move next')
    else:
        # delete a previously placed piece
        if move_status['ok']:
            move_status['patch_object'].remove()
        move_status['ok'] = False
        move_status['pos'] = None
    # redraw the board
    self.board_canvas.draw()

#%% _calc_cur_move
def _calc_cur_move(cur_move, cur_game):
    r"""
    Calculates whose move it is based on the turn and game number.

    Parameters
    ----------
    cur_move : int
        Current move number
    cur_game : int
        Current game number

    Returns
    -------
    move : int
        Current move, from {1=white, -1=black}

    Examples
    --------
    >>> from dstauffman.games.pentago import _calc_cur_move
    >>> move = _calc_cur_move(0, 0)
    >>> print(move)
    1

    """
    if np.mod(cur_move + cur_game, 2) == 0:
        move = PLAYER['white']
    else:
        move = PLAYER['black']
    return move

#%% _execute_move
def _execute_move(*, quadrant, direction):
    r"""
    Tests and then executes a move.
    """
    global board, cur_move
    if move_status['ok']:
        if LOGGING:
            print('Rotating Quadrant {} in Direction {}.'.format(quadrant, direction))
        # delete gray piece
        move_status['patch_object'].remove()
        move_status['patch_object'] = None
        # add new piece to board
        board[move_status['pos'][0], move_status['pos'][1]] = _calc_cur_move(cur_move, cur_game)
        # rotate board
        _rotate_board(board, quadrant, direction)
        # increment move list
        assert game_hist[cur_game].num_moves >= cur_move, \
            'Number of moves = {}, Current Move = {}'.format(game_hist[cur_game].num_moves, cur_move)
        this_move = Move(move_status['pos'][0], move_status['pos'][1], quadrant, direction)
        if game_hist[cur_game].num_moves == cur_move:
            game_hist[cur_game].add_move(this_move)
        else:
            game_hist[cur_game].move_list[cur_move] = this_move
            game_hist[cur_game].remove_moves(cur_move+1)
        # increment current move
        cur_move += 1
        # reset status for next move
        move_status['ok'] = False
        move_status['pos'] = None
    else:
        if LOGGING:
            print('No move to execute.')

#%% _rotate_board
def _rotate_board(board, quadrant, direction):
    r"""
    Rotates the specified board position.

    Notes
    -----
    #.  Modifies board in-place.
    """
    # determine if 6x6 board or 36xN
    (r, c) = board.shape

    if r == 6 and c == 6:
        # get quad
        if quadrant == 1:
            old_sub = board[0:3, 0:3]
        elif quadrant == 2:
            old_sub = board[0:3, 3:6]
        elif quadrant == 3:
            old_sub = board[3:6, 0:3]
        elif quadrant == 4:
            old_sub = board[3:6, 3:6]
        else:
            raise ValueError('Unexpected value for quadrant.')

        # rotate quad
        if direction == -1:
            new_sub = np.rot90(old_sub)
        elif direction == 1:
            new_sub = np.rot90(old_sub, 3)
        else:
            raise ValueError('Unexpected value for dir')

        # update rotated quad
        if quadrant == 1:
            board[0:3, 0:3] = new_sub
        elif quadrant == 2:
            board[0:3, 3:6] = new_sub
        elif quadrant == 3:
            board[3:6, 0:3] = new_sub
        elif quadrant == 4:
            board[3:6, 3:6] = new_sub
    elif r == 36: # TODO: fix this case
        # get quad
        if quadrant == 1:
            ix_old = np.array([ 1,  2,  3,  7,  8,  9, 13, 14, 15]) # TODO: update these numbers
        elif quadrant == 2:
            ix_old = np.array([19, 20, 21, 25, 26, 27, 31, 32, 33])
        elif quadrant == 3:
            ix_old = np.array([ 4,  5,  6, 10, 11, 12, 16, 17, 18])
        elif quadrant == 4:
            ix_old = np.array([22, 23, 24, 28, 29, 30, 34, 35, 36])
        else:
            raise ValueError('Unexpected value for quad')
        # rotate quad
        if direction == -1:
            ix_new = ix_old([7, 4, 1, 8, 5, 2, 9, 6, 3]);
        elif direction == 1:
            ix_new = ix_old([3, 6, 9, 2, 5, 8, 1, 4, 7]);
        else:
            raise ValueError('Unexpected value for dir')
        # update placements
        board[ix_old, :] = board[ix_new, :]
    else:
        raise ValueError('Unexpected size of board.')

#%% _check_for_win
def _check_for_win(board):
    r"""
    Checks for a win.
    """
    # find white and black wins
    white = np.nonzero(np.sum(np.expand_dims(board.ravel() == PLAYER['white'], axis=1) * WIN, axis=0) == 5)[0]
    black = np.nonzero(np.sum(np.expand_dims(board.ravel() == PLAYER['black'], axis=1) * WIN, axis=0) == 5)[0]

    # determine winner
    if len(white) == 0:
        if len(black) == 0:
            winner = PLAYER['none']
        else:
            winner = PLAYER['black']
    else:
        if len(black) == 0:
            winner = PLAYER['white']
        else:
            winner = PLAYER['draw']

    # check for a full game board after determining no other win was found
    if winner == PLAYER['none'] and not np.any(board == PLAYER['none']):
        winner = PLAYER['draw']

    # find winning pieces on the board
    if winner == PLAYER['none']:
        win_mask = np.zeros((6,6), dtype=bool)
    else:
        if LOGGING:
            print('Win detected.  Winner is {}.'.format(list(PLAYER.keys())[list(PLAYER.values()).index(winner)]))
        win_mask = np.reshape(np.sum(WIN[:, white], axis=1) + np.sum(WIN[:, black], axis=1), (6, 6)) != 0

    # update statistics
    game_hist[cur_game].winner = winner

    return (winner, win_mask)

#%% _update_game_stats
def _update_game_stats(self, results):
    r"""
    Updates the game stats on the left of the GUI.
    """
    # calculate the number of wins
    white_wins = np.sum(results == PLAYER['white'])
    black_wins = np.sum(results == PLAYER['black'])
    games_tied = np.sum(results == PLAYER['draw'])

    # update the gui
    self.lbl_white_wins.setText("{}".format(white_wins))
    self.lbl_black_wins.setText("{}".format(black_wins))
    self.lbl_games_tied.setText("{}".format(games_tied))

#%% _display_controls
def _display_controls(self):
    r"""
    Determines what controls to display on the GUI.
    """
    # show/hide New Game Button
    if game_hist[cur_game].winner == PLAYER['none']:
        self.btn_new.hide()
    else:
        self.btn_new.show()

    # show/hide Undo Button
    if cur_move > 0:
        self.btn_undo.show()
    else:
        self.btn_undo.hide()

    # show/hide Redo Button
    if game_hist[cur_game].num_moves > cur_move:
        self.btn_redo.show()
    else:
        self.btn_redo.hide()

#%% _plot_cur_move
def _plot_cur_move(ax, move):
    r"""
    Plots the piece corresponding the current players move.
    """
    # local alias
    half_box_size = RADIUS['square']

    # fill background
    ax.add_patch(Rectangle((-half_box_size, -half_box_size), 2*half_box_size, 2*half_box_size, \
        color=COLOR['board']))

    # draw the piece
    if move == PLAYER['white']:
        _plot_piece(ax, 0, 0, RADIUS['piece'], COLOR['white'])
    elif move == PLAYER['black']:
        _plot_piece(ax, 0, 0, RADIUS['piece'], COLOR['white'])
    elif move == PLAYER['none']:
        pass
    else:
        raise ValueError('Unexpected player.')

    # set axis limits (TODO: can these be moved into just the initialization?)
    ax.set_xlim(-half_box_size, half_box_size)
    ax.set_ylim(-half_box_size, half_box_size)
    ax.set_axis_off()

#%% _plot_picee
def _plot_piece(ax, vc, hc, r, c, half=False):
    r"""
    Plots a piece on the board.

    Parameters
    ----------
    ax, object
        Axis to plot on
    vc, float
        Vertical center (Y-axis or board row)
    hc, float
        Horizontal center (X-axis or board column)
    r, float
        radius
    c, 3-tuple
        RGB triplet color
    half, bool optional, default is false
        flag for plotting half a piece

    Returns
    -------
    fill_handle, object
        handle to the piece

    Examples
    --------
    >>> from dstauffman.games.pentago import _plot_piece
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.set_xlim(0.5, 1.5)
    >>> _ = ax.set_ylim(0.5, 1.5)
    >>> _plot_piece(ax, 1, 1, 0.45, (0, 0, 1))
    >>> plt.show(block=False)

    >>> plt.close()

    """

    # theta angle to sweep out 2*pi
    if half:
        piece = Wedge((hc, vc), r, 180, 360, facecolor=c, edgecolor='k')
    else:
        piece = Circle((hc, vc), r, facecolor=c, edgecolor='k')

    # plot piece
    ax.add_patch(piece)
    return piece

#%% _plot_board
def _plot_board(ax):
    r"""
    Plots the board (and the current player move).
    """

    # delete previous children
    # TODO: write this

    # get axes limits
    (m, n) = board.shape
    s = RADIUS['square']
    xmin = 0 - s
    xmax = m - 1 + s
    ymin = 0 - s
    ymax = n - 1 + s

    # fill background
    ax.add_patch(Rectangle((-xmin-1, -ymin-1), xmax-xmin, ymax-ymin, facecolor=COLOR['board'], \
        edgecolor=COLOR['maj_edge']))

    # draw minor horizontal lines
    ax.plot([1-s, 1-s], [ymin, ymax], color=COLOR['min_edge'])
    ax.plot([2-s, 2-s], [ymin, ymax], color=COLOR['min_edge'])
    ax.plot([4-s, 4-s], [ymin, ymax], color=COLOR['min_edge'])
    ax.plot([5-s, 5-s], [ymin, ymax], color=COLOR['min_edge'])
    # draw minor vertical lines
    ax.plot([xmin, xmax], [1-s, 1-s], color=COLOR['min_edge'])
    ax.plot([xmin, xmax], [2-s, 2-s], color=COLOR['min_edge'])
    ax.plot([xmin, xmax], [4-s, 4-s], color=COLOR['min_edge'])
    ax.plot([xmin, xmax], [5-s, 5-s], color=COLOR['min_edge'])
    # draw major quadrant lines
    ax.plot([3-s, 3-s], [ymin, ymax], color=COLOR['maj_edge'], linewidth=2)
    ax.plot([xmin, xmax], [3-s, 3-s], color=COLOR['maj_edge'], linewidth=2)

    # loop through and place marbles
    for i in range(m):
        for j in range(n):
            if board[i, j] == PLAYER['none']:
                pass
            elif board[i, j] == PLAYER['white']:
                _plot_piece(ax, i, j, RADIUS['piece'], COLOR['white'])
            elif board[i, j] == PLAYER['black']:
                _plot_piece(ax, i, j, RADIUS['piece'], COLOR['black'])
            else:
                raise ValueError('Bad board position.')

#%% _plot_win
def _plot_win(ax, mask):
    r"""
    Plots the winning pieces in red.

    Parameters
    ----------
    ax : object
        Axis to plot on
    mask : 2D bool ndarray
        Mask for which squares to plot the win

    Examples
    --------
    >>> from dstauffman.games.pentago import _plot_win
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 5.5)
    >>> _ = ax.set_ylim(-0.5, 5.5)
    >>> ax.invert_yaxis()
    >>> mask = np.zeros((6, 6), dtype=bool)
    >>> mask[0, 0:5] = True
    >>> _plot_win(ax, mask)
    >>> plt.show(block=False)

    >>> plt.close()

    """
    (m, n) = mask.shape
    for i in range(m):
        for j in range(n):
            if mask[i, j]:
                _plot_piece(ax, i, j, RADIUS['win'], COLOR['win'])

#%% move_wrapper
def wrapper(self):
    r"""
    Acts as a wrapper to everything the GUI needs to do.
    """
    # clean up an existing artifacts
    self.board_axes.clear()
    self.move_axes.clear()

    # plot the current move
    _plot_cur_move(self.move_axes, 1)
    self.move_canvas.draw()

    # draw turn arrows
    # TODO: write this

    # draw the board
    _plot_board(self.board_axes)

    # check for win
    (winner, win_mask) = _check_for_win(board)

    # plot win
    _plot_win(self.board_axes, win_mask)

    # display relevant controls
    _display_controls(self)

    # plot possible winning moves
    # TODO:

    # redraw with the final board
    #self.board_axes.set_axis_off() # Turn this off for debugging
    self.board_canvas.draw()

    # update game stats on GUI
    results = [x.winner for x in game_hist]
    _update_game_stats(self, results=np.array(results)) # TODO: make better?
    self.update()

#%% Unit Test
def _main():
    """Unit test function."""
    pass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'null'
    if mode == 'run':
        # Runs the GUI application
        app = QApplication(sys.argv)
        gui = PentagoGui()
        gui.show()
        sys.exit(app.exec_())
    elif mode == 'test':
        _main()
        # open a qapp
        if QApplication.instance() is None:
            qapp = QApplication(sys.argv)
        else:
            qapp = QApplication.instance()
        # run the tests
        unittest.main(module='dstauffman.games.test_pentago', exit=False)
        doctest.testmod(verbose=False)
        # close the qapp
        qapp.closeAllWindows()
    elif mode == 'null':
        pass
    else:
        raise ValueError('Unexpected mode of "{}".'.format(mode))
