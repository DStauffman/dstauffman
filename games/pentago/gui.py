# -*- coding: utf-8 -*-
r"""
GUI module file for the "pentago" game.  It defines the GUI.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
import logging
from matplotlib.pyplot import Axes
from matplotlib.figure import Figure
import numpy as np
import os
import sys
import unittest
try: # pragma: no cover
    from PyQt5 import QtGui, QtCore
    from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QLabel, QMessageBox
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError: # pragma: no cover
    from PyQt4 import QtGui, QtCore
    from PyQt4.QtGui import QApplication, QWidget, QToolTip, QPushButton, QLabel, QMessageBox
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from dstauffman import get_images_dir, get_output_dir, Counter
from dstauffman.games.pentago.classes   import Move, GameStats, State
from dstauffman.games.pentago.constants import COLOR, PLAYER, OPTIONS, SIZES
from dstauffman.games.pentago.plotting  import plot_board, plot_cur_move, plot_piece, \
                                               plot_possible_win, plot_win
from dstauffman.games.pentago.utils     import calc_cur_move, check_for_win, create_board_from_moves, \
                                               find_moves, rotate_board

# TODO: add boxes for flipping settings

#%% Classes - RotationButton
class RotationButton(QPushButton):
    r"""
    Custom QPushButton to allow drawing multiple images on the buttons for plotting possible winning rotations.
    """
    def __init__(self, text, parent, quadrant, direction):
        super(RotationButton, self).__init__(text, parent)
        self.quadrant  = quadrant
        self.direction = direction
        self.overlay   = None

    def paintEvent(self, event):
        r"""Custom paint event to update the buttons."""
        # call super method
        QPushButton.paintEvent(self, event)
        # create painter and load base image
        painter = QtGui.QPainter(self)
        pixmap_key = str(self.quadrant) + ('L' if self.direction == -1 else 'R')
        images = self.parent().images
        pixmap = images[pixmap_key].pixmap(QtCore.QSize(SIZES['button'], SIZES['button']))
        if self.overlay is None:
            painter.drawPixmap(0, 0, pixmap)
        else:
            # optionally load the overlaid image
            overlay_pixmap = images[self.overlay].pixmap(QtCore.QSize(SIZES['button'], SIZES['button']))
            painter.drawPixmap(0, 0, overlay_pixmap)
            painter.setCompositionMode(painter.CompositionMode_SourceOver)
            painter.drawPixmap(0, 0, pixmap)
        painter.end()

#%% Classes - PentagoGui
class PentagoGui(QWidget):
    r"""
    The Pentago GUI.
    """
    def __init__(self, **kwargs):
        # call super method
        super(PentagoGui, self).__init__(**kwargs)
        # initialized the state data
        self.initialize_state()
        # load the image data
        self.load_images()
        # call init method to instantiate the GUI
        self.init()

    #%% State initialization
    def initialize_state(self):
        r"""
        Loads the previous game based on settings and whether the file exists.
        """
        # preallocate to not load
        load_game = False
        if OPTIONS['load_previous_game'] == 'No':
            pass
        elif OPTIONS['load_previous_game'] == 'Yes':
            load_game = True
        # ask if loading
        elif OPTIONS['load_previous_game'] == 'Ask':
            widget = QWidget()
            reply = QMessageBox.question(widget, 'Message', \
                "Do you want to load the previous game?", QMessageBox.Yes | \
                QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                load_game = True
        else:
            raise ValueError('Unexpected value for the load_previous_game option.')
        # initialize outputs
        self.state = State()
        if load_game:
            filename  = os.path.join(get_output_dir(), 'pentago.pkl')
            if os.path.isfile(filename):
                self.state.game_hist   = GameStats.load(filename)
                self.state.cur_game    = Counter(len(self.state.game_hist)-1)
                self.state.cur_move    = Counter(len(self.state.game_hist[-1].move_list))
                self.state.board       = create_board_from_moves(self.state.game_hist[-1].move_list, \
                    self.state.game_hist[-1].first_move)
                self.state.move_status = {'ok': False, 'pos': None, 'patch_object': None}

    #%% load_images
    def load_images(self):
        r"""
        Loads the images for use later on.

        Returns
        -------
        images : dict
            Images for use on the rotation buttons.

        Notes
        -----
        #.  Written by David C. Stauffer in January 2016.
        #.  TODO: needs a QApplication to exist first.  Play around with making this earlier.

        """
        # get the directory for the images
        images_dir = get_images_dir()
        # create a dictionary for saving all the images in
        self.images        = {}
        self.images['1R']  = QtGui.QIcon(os.path.join(images_dir, 'right1.png'))
        self.images['2R']  = QtGui.QIcon(os.path.join(images_dir, 'right2.png'))
        self.images['3R']  = QtGui.QIcon(os.path.join(images_dir, 'right3.png'))
        self.images['4R']  = QtGui.QIcon(os.path.join(images_dir, 'right4.png'))
        self.images['1L']  = QtGui.QIcon(os.path.join(images_dir, 'left1.png'))
        self.images['2L']  = QtGui.QIcon(os.path.join(images_dir, 'left2.png'))
        self.images['3L']  = QtGui.QIcon(os.path.join(images_dir, 'left3.png'))
        self.images['4L']  = QtGui.QIcon(os.path.join(images_dir, 'left4.png'))
        self.images['wht'] = QtGui.QIcon(os.path.join(images_dir, 'blue_button.png'))
        self.images['blk'] = QtGui.QIcon(os.path.join(images_dir, 'cyan_button.png'))
        self.images['w_b'] = QtGui.QIcon(os.path.join(images_dir, 'blue_cyan_button.png'))
        self.images['b_w'] = QtGui.QIcon(os.path.join(images_dir, 'cyan_blue_button.png'))

    #%% GUI initialization
    def init(self):
        r"""Initializes the GUI."""
        #%% properties
        QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        #%% Text
        # Pentago
        lbl_pentago = QLabel('Pentago', self)
        lbl_pentago.setGeometry(390, 50, 220, 40)
        lbl_pentago.setAlignment(QtCore.Qt.AlignCenter)
        lbl_pentago.setStyleSheet('font-size: 18pt; font: bold;')
        # Score
        lbl_score = QLabel('Score:', self)
        lbl_score.setGeometry(35, 220, 220, 40)
        lbl_score.setAlignment(QtCore.Qt.AlignCenter)
        lbl_score.setStyleSheet('font-size: 12pt; font: bold;')
        # Move
        lbl_move = QLabel('Move:', self)
        lbl_move.setGeometry(725, 220, 220, 40)
        lbl_move.setAlignment(QtCore.Qt.AlignCenter)
        lbl_move.setStyleSheet('font-size: 12pt; font: bold;')
        # White Wins
        lbl_white = QLabel('White Wins:', self)
        lbl_white.setGeometry(80, 280, 80, 20)
        # Black Wins
        lbl_black = QLabel('Black Wins:', self)
        lbl_black.setGeometry(80, 310, 80, 20)
        # Games Tied
        lbl_games = QLabel('Games Tied:', self)
        lbl_games.setGeometry(80, 340, 80, 20)
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
        # board
        self.wid_board = QWidget(self)
        self.wid_board.setGeometry(290, 140, 420, 420)
        fig = Figure(figsize=(4.2, 4.2), dpi=100, frameon=False)
        self.board_canvas = FigureCanvas(fig)
        self.board_canvas.setParent(self.wid_board)
        self.board_canvas.mpl_connect('button_release_event', lambda event: self.mouse_click_callback(event))
        self.board_axes = Axes(fig, [0., 0., 1., 1.])
        self.board_axes.invert_yaxis()
        self.board_axes.set_axis_off()
        fig.add_axes(self.board_axes)

        # current move
        self.wid_move = QWidget(self)
        self.wid_move.setGeometry(800, 280, 70, 70)
        fig = Figure(figsize=(.7, .7), dpi=100, frameon=False)
        self.move_canvas = FigureCanvas(fig)
        self.move_canvas.setParent(self.wid_move)
        self.move_axes = Axes(fig, [0., 0., 1., 1.])
        self.move_axes.set_xlim(-SIZES['square']/2, SIZES['square']/2)
        self.move_axes.set_ylim(-SIZES['square']/2, SIZES['square']/2)
        self.move_axes.set_axis_off()
        fig.add_axes(self.move_axes)

        #%% Buttons
        button_size = QtCore.QSize(SIZES['button'], SIZES['button'])
        # Undo button
        self.btn_undo = QPushButton('Undo', self)
        self.btn_undo.setToolTip('Undoes the last move.')
        self.btn_undo.setGeometry(380, 600, 60, 30)
        self.btn_undo.setStyleSheet('color: yellow; background-color: #990000; font: bold;')
        self.btn_undo.clicked.connect(self.btn_undo_function)
        # New Game button
        self.btn_new = QPushButton('New Game', self)
        self.btn_new.setToolTip('Starts a new game.')
        self.btn_new.setGeometry(460, 600, 80, 50)
        self.btn_new.setStyleSheet('color: yellow; background-color: #006633; font: bold;')
        self.btn_new.clicked.connect(self.btn_new_function)
        # Redo button
        self.btn_redo = QPushButton('Redo', self)
        self.btn_redo.setToolTip('Redoes the last move.')
        self.btn_redo.setGeometry(560, 600, 60, 30)
        self.btn_redo.setStyleSheet('color: yellow; background-color: #000099; font: bold;')
        self.btn_redo.clicked.connect(self.btn_redo_function)

        # 1R button
        self.btn_1R = RotationButton('', self, quadrant=1, direction=1)
        self.btn_1R.setToolTip('Rotates quadrant 1 to the right 90 degrees.')
        self.btn_1R.setIconSize(button_size)
        self.btn_1R.setGeometry(290, 49, SIZES['button'], SIZES['button'])
        self.btn_1R.clicked.connect(self.btn_rot_function)
        # 2R button
        self.btn_2R = RotationButton('', self, quadrant=2, direction=1)
        self.btn_2R.setToolTip('Rotates quadrant 2 to the right 90 degrees.')
        self.btn_2R.setIconSize(button_size)
        self.btn_2R.setGeometry(730, 139, SIZES['button'], SIZES['button'])
        self.btn_2R.clicked.connect(self.btn_rot_function)
        # 3R button
        self.btn_3R = RotationButton('', self, quadrant=3, direction=1)
        self.btn_3R.setToolTip('Rotates quadrant 3 to the right 90 degrees.')
        self.btn_3R.setIconSize(button_size)
        self.btn_3R.setGeometry(200, 489, SIZES['button'], SIZES['button'])
        self.btn_3R.clicked.connect(self.btn_rot_function)
        # 4R button
        self.btn_4R = RotationButton('', self, quadrant=4, direction=1)
        self.btn_4R.setToolTip('Rotates quadrant 4 to the right 90 degrees.')
        self.btn_4R.setIconSize(button_size)
        self.btn_4R.setGeometry(640, 579, SIZES['button'], SIZES['button'])
        self.btn_4R.clicked.connect(self.btn_rot_function)
        # 1L button
        self.btn_1L = RotationButton('', self, quadrant=1, direction=-1)
        self.btn_1L.setToolTip('Rotates quadrant 1 to the left 90 degrees.')
        self.btn_1L.setIconSize(button_size)
        self.btn_1L.setGeometry(200, 139, SIZES['button'], SIZES['button'])
        self.btn_1L.clicked.connect(self.btn_rot_function)
        # 2L button
        self.btn_2L = RotationButton('', self, quadrant=2, direction=-1)
        self.btn_2L.setToolTip('Rotates quadrant 2 to the left 90 degrees.')
        self.btn_2L.setIconSize(button_size)
        self.btn_2L.setGeometry(640, 49, SIZES['button'], SIZES['button'])
        self.btn_2L.clicked.connect(self.btn_rot_function)
        # 3L button
        self.btn_3L = RotationButton('', self, quadrant=3, direction=-1)
        self.btn_3L.setToolTip('Rotates quadrant 3 to the left 90 degrees.')
        self.btn_3L.setIconSize(button_size)
        self.btn_3L.setGeometry(290, 579, SIZES['button'], SIZES['button'])
        self.btn_3L.clicked.connect(self.btn_rot_function)
        # 4L button
        self.btn_4L = RotationButton('', self, quadrant=4, direction=-1)
        self.btn_4L.setToolTip('Rotates quadrant 4 to the left 90 degrees.')
        self.btn_4L.setIconSize(button_size)
        self.btn_4L.setGeometry(730, 489, SIZES['button'], SIZES['button'])
        self.btn_4L.clicked.connect(self.btn_rot_function)
        # buttons dictionary for use later
        self.rot_buttons = {'1L':self.btn_1L, '2L':self.btn_2L, '3L':self.btn_3L, '4L':self.btn_4L, \
            '1R':self.btn_1R, '2R':self.btn_2R, '3R':self.btn_3R, '4R':self.btn_4R}

        #%% Call wrapper to initialize GUI
        self.wrapper()

        #%% GUI properties
        self.resize(1000, 700)
        self.center()
        self.setWindowTitle('Pentago')
        self.setWindowIcon(QtGui.QIcon(os.path.join(get_images_dir(),'pentago.png')))
        self.show()

    #%% Other callbacks - closing
    def closeEvent(self, event):
        """Things in here happen on GUI closing."""
        close_immediately = True
        filename = os.path.join(get_output_dir(), 'pentago.pkl')
        if close_immediately:
            GameStats.save(filename, self.state.game_hist)
            event.accept()
        else:
            # Alternative with user choice
            reply = QMessageBox.question(self, 'Message', \
                "Are you sure to quit?", QMessageBox.Yes | \
                QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                GameStats.save(filename, self.state.game_hist)
                event.accept()
            else:
                event.ignore()

    #%% Other callbacks - center the GUI on the screen
    def center(self):
        r"""Makes the GUI centered on the active screen."""
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    #%% Other callbacks - display_controls
    def display_controls(self):
        r"""
        Determines what controls to display on the GUI.
        """
        # show/hide New Game Button
        if self.state.game_hist[self.state.cur_game].winner == PLAYER['none']:
            self.btn_new.hide()
        else:
            self.btn_new.show()

        # show/hide Undo Button
        if self.state.cur_move > 0:
            self.btn_undo.show()
        else:
            self.btn_undo.hide()

        # show/hide Redo Button
        if self.state.game_hist[self.state.cur_game].num_moves > self.state.cur_move:
            self.btn_redo.show()
        else:
            self.btn_redo.hide()

    #%% update_game_stats
    def update_game_stats(self, results):
        r"""
        Updates the game stats on the left of the GUI.
        """
        # calculate the number of wins
        white_wins = np.count_nonzero(results == PLAYER['white'])
        black_wins = np.count_nonzero(results == PLAYER['black'])
        games_tied = np.count_nonzero(results == PLAYER['draw'])

        # update the gui
        self.lbl_white_wins.setText("{}".format(white_wins))
        self.lbl_black_wins.setText("{}".format(black_wins))
        self.lbl_games_tied.setText("{}".format(games_tied))

    #%% Button callbacks
    def btn_undo_function(self):
        r"""Functions that executes on undo button press."""
        # get last move
        last_move = self.state.game_hist[self.state.cur_game].move_list[self.state.cur_move-1]
        logging.debug('Undoing move = {}'.format(last_move))
        # undo rotation
        rotate_board(self.state.board, last_move.quadrant, -last_move.direction)
        # delete piece
        self.state.board[last_move.row, last_move.column] = PLAYER['none']
        # update current move
        self.state.cur_move -= 1
        # call GUI wrapper
        self.wrapper()

    def btn_new_function(self):
        r"""Functions that executes on new game button press."""
        # update values
        last_lead = self.state.game_hist[self.state.cur_game].first_move
        next_lead = PLAYER['black'] if last_lead == PLAYER['white'] else PLAYER['white']
        assert len(self.state.game_hist) == self.state.cur_game + 1
        self.state.cur_game += 1
        self.state.cur_move = Counter(0)
        self.state.game_hist.append(GameStats(number=self.state.cur_game, first_move=next_lead, winner=PLAYER['none']))
        self.state.board = PLAYER['none'] * np.ones((SIZES['board'], SIZES['board']), dtype=int)
        # call GUI wrapper
        self.wrapper()

    def btn_redo_function(self):
        r"""Functions that executes on redo button press."""
        # get next move
        redo_move = self.state.game_hist[self.state.cur_game].move_list[self.state.cur_move]
        logging.debug('Redoing move = {}'.format(redo_move))
        # place piece
        self.state.board[redo_move.row, redo_move.column] = calc_cur_move(self.state.cur_move, self.state.cur_game)
        # redo rotation
        rotate_board(self.state.board, redo_move.quadrant, redo_move.direction)
        # update current move
        self.state.cur_move += 1
        # call GUI wrapper
        self.wrapper()

    def btn_rot_function(self):
        r"""Functions that executes on rotation button press."""
        # determine sending button
        button = self.sender()
        # execute the move
        self.execute_move(quadrant=button.quadrant, direction=button.direction)
        # call GUI wrapper
        self.wrapper()

    #%% mouse_click_callback
    def mouse_click_callback(self, event):
        r"""
        Function that executes on mouse click on the board axes.  Ends up placing a piece on the board.
        """
        # ignore events that are outside the axes
        if event.xdata is None or event.ydata is None:
            logging.debug('Click is off the board.')
            return
        # test for a game that has already been concluded
        if self.state.game_hist[self.state.cur_game].winner != PLAYER['none']:
            logging.debug('Game is over.')
            self.state.move_status['ok'] = False
            self.state.move_status['pos'] = None
            return
        # alias the rounded values of the mouse click location
        x = np.round(event.ydata).astype(int)
        y = np.round(event.xdata).astype(int)
        logging.debug('Clicked on (x,y) = ({}, {})'.format(x, y))
        # get axes limits
        (m, n) = self.state.board.shape
        # ignore values that are outside the board
        if x < 0 or y < 0 or x >= m or y >= n:
            logging.debug('Click is outside playable board.')
            return
        if self.state.board[x, y] == PLAYER['none']:
            # check for previous good move
            if self.state.move_status['ok']:
                logging.debug('removing previous piece.')
                self.state.move_status['patch_object'].remove()
            self.state.move_status['ok'] = True
            self.state.move_status['pos'] = (x, y)
            logging.debug('Placing current piece.')
            current_player = calc_cur_move(self.state.cur_move, self.state.cur_game)
            if current_player == PLAYER['white']:
                self.state.move_status['patch_object'] = plot_piece(self.board_axes, x, y, SIZES['piece'], COLOR['next_wht'])
            elif current_player == PLAYER['black']:
                self.state.move_status['patch_object'] = plot_piece(self.board_axes, x, y, SIZES['piece'], COLOR['next_blk'])
            else:
                raise ValueError('Unexpected player to move next.')
        else:
            # delete a previously placed piece
            if self.state.move_status['ok']:
                self.state.move_status['patch_object'].remove()
            self.state.move_status['ok'] = False
            self.state.move_status['pos'] = None
        # redraw the board
        self.board_canvas.draw()

    #%% execute_move
    def execute_move(self, *, quadrant, direction):
        r"""
        Tests and then executes a move.
        """
        if self.state.move_status['ok']:
            logging.debug('Rotating Quadrant {} in Direction {}.'.format(quadrant, direction))
            # delete gray piece
            self.state.move_status['patch_object'].remove()
            self.state.move_status['patch_object'] = None
            # add new piece to board
            self.state.board[self.state.move_status['pos'][0], self.state.move_status['pos'][1]] = \
                calc_cur_move(self.state.cur_move, self.state.cur_game)
            # rotate board
            rotate_board(self.state.board, quadrant, direction)
            # increment move list
            assert self.state.game_hist[self.state.cur_game].num_moves >= self.state.cur_move, \
                'Number of moves = {}, Current Move = {}'.format(self.state.game_hist[self.state.cur_game].num_moves, self.state.cur_move)
            this_move = Move(self.state.move_status['pos'][0], self.state.move_status['pos'][1], quadrant, direction)
            if self.state.game_hist[self.state.cur_game].num_moves == self.state.cur_move:
                self.state.game_hist[self.state.cur_game].add_move(this_move)
            else:
                self.state.game_hist[self.state.cur_game].move_list[self.state.cur_move] = this_move
                self.state.game_hist[self.state.cur_game].remove_moves(self.state.cur_move+1)
            # increment current move
            self.state.cur_move += 1
            # reset status for next move
            self.state.move_status['ok'] = False
            self.state.move_status['pos'] = None
        else:
            logging.debug('No move to execute.')

    #%% wrapper
    def wrapper(self):
        r"""
        Acts as a wrapper to everything the GUI needs to do.
        """
        # clean up an existing artifacts
        self.board_axes.clear()
        self.move_axes.clear()

        # plot the current move
        plot_cur_move(self.move_axes, calc_cur_move(self.state.cur_move, self.state.cur_game))
        self.move_canvas.draw()

        # draw turn arrows in default colors
        for button in self.rot_buttons.values():
            button.overlay = None

        # draw the board
        plot_board(self.board_axes, self.state.board)

        # check for win
        (winner, win_mask) = check_for_win(self.state.board)
        # update winner
        self.state.game_hist[self.state.cur_game].winner = winner

        # plot win
        plot_win(self.board_axes, win_mask)

        # display relevant controls
        self.display_controls()

        # plot possible winning moves (includes updating turn arrows)
        if winner == PLAYER['none'] and OPTIONS['plot_winning_moves']:
            (white_moves, black_moves) = find_moves(self.state.board)
            plot_possible_win(self.board_axes, self.rot_buttons, white_moves, black_moves, self.state.cur_move, self.state.cur_game)

        # redraw with the final board
        self.board_axes.set_axis_off()
        self.board_canvas.draw()

        # update game stats on GUI
        self.update_game_stats(results=GameStats.get_results(self.state.game_hist))
        self.update()

#%% Unit Test
if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    unittest.main(module='dstauffman.games.pentago.tests.test_gui', exit=False)
    doctest.testmod(verbose=False)
    # close the qapp
    qapp.closeAllWindows()
