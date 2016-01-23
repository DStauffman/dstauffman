# -*- coding: utf-8 -*-
r"""
GUI module file for the "tictactoe" game.  It defines the GUI.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
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
from dstauffman.games.tictactoe.classes   import GameStats, State
from dstauffman.games.tictactoe.constants import LOGGING, PLAYER, OPTIONS, SIZES
from dstauffman.games.tictactoe.plotting  import plot_board, plot_cur_move, plot_possible_win, \
                                                 plot_powers, plot_win
from dstauffman.games.tictactoe.utils     import calc_cur_move, check_for_win, create_board_from_moves, \
                                                 find_moves, make_move, play_ai_game

# TODO: make into grid layout
# TODO: add boxes for flipping settings

#%% Classes - TicTacToeGui
class TicTacToeGui(QWidget):
    r"""
    The Tic Tac Toe GUI.
    """
    def __init__(self, filename=None, board=None, cur_move=None, cur_game=None, game_hist=None):
        # call super method
        super(TicTacToeGui, self).__init__()
        # initialize the state data
        self.initialize_state()
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
        # load previous game
        if load_game:
            filename  = os.path.join(get_output_dir(), 'tictactoe.p')
            if os.path.isfile(filename):
                self.state.game_hist   = GameStats.load(filename)
                self.state.cur_game    = Counter(len(self.state.game_hist)-1)
                self.state.cur_move    = Counter(len(self.state.game_hist[-1].move_list))
                self.state.board       = create_board_from_moves(self.state.game_hist[-1].move_list, \
                    self.state.game_hist[-1].first_move)
            else:
                raise ValueError('Could not find file: "{}"'.format(filename))

    #%% GUI initialization
    def init(self):
        r"""Initializes the GUI."""
        #%% properties
        QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        #%% Text
        # Tic Tac Toe
        lbl_tictactoe = QLabel('Tic Tac Toe', self)
        lbl_tictactoe.setGeometry(360, 51, 220, 40)
        lbl_tictactoe.setAlignment(QtCore.Qt.AlignCenter)
        lbl_tictactoe.setStyleSheet('font-size: 18pt; font: bold;')
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
        # O Wins
        lbl_o = QLabel('O Wins:', self)
        lbl_o.setGeometry(50, 280, 80, 20)
        # X Wins
        lbl_x = QLabel('X Wins:', self)
        lbl_x.setGeometry(50, 310, 80, 20)
        # Games Tied
        lbl_games = QLabel('Games Tied:', self)
        lbl_games.setGeometry(50, 340, 80, 20)
        # Changeable labels
        self.lbl_o_wins = QLabel('0', self)
        self.lbl_o_wins.setGeometry(140, 280, 60, 20)
        self.lbl_o_wins.setAlignment(QtCore.Qt.AlignRight)
        self.lbl_x_wins = QLabel('0', self)
        self.lbl_x_wins.setGeometry(140, 310, 60, 20)
        self.lbl_x_wins.setAlignment(QtCore.Qt.AlignRight)
        self.lbl_games_tied = QLabel('0', self)
        self.lbl_games_tied.setGeometry(140, 340, 60, 20)
        self.lbl_games_tied.setAlignment(QtCore.Qt.AlignRight)

        #%% Axes
        # board
        self.wid_board = QWidget(self)
        self.wid_board.setGeometry(260, 140, 420, 420)
        fig = Figure(figsize=(4.2, 4.2), dpi=100, frameon=False)
        self.board_canvas = FigureCanvas(fig)
        self.board_canvas.setParent(self.wid_board)
        self.board_canvas.mpl_connect('button_release_event', lambda event: self.mouse_click_callback(event))
        self.board_axes = Axes(fig, [0., 0., 1., 1.])
        self.board_axes.invert_yaxis()
        fig.add_axes(self.board_axes)

        # current move
        self.wid_move = QWidget(self)
        self.wid_move.setGeometry(780, 279, 70, 70)
        fig = Figure(figsize=(.7, .7), dpi=100, frameon=False)
        self.move_canvas = FigureCanvas(fig)
        self.move_canvas.setParent(self.wid_move)
        self.move_axes = Axes(fig, [0., 0., 1., 1.])
        self.move_axes.set_xlim(-SIZES['square']/2, SIZES['square']/2)
        self.move_axes.set_ylim(-SIZES['square']/2, SIZES['square']/2)
        self.move_axes.set_axis_off()
        fig.add_axes(self.move_axes)

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

        #%% Call wrapper to initialize GUI
        self.wrapper()

        #%% GUI properties
        self.resize(1000, 700)
        self.center()
        self.setWindowTitle('Tic Tac Toe')
        self.setWindowIcon(QtGui.QIcon(os.path.join(get_images_dir(),'tictactoe.png')))
        self.show()

    #%% Other callbacks - closing
    def closeEvent(self, event):
        r"""Things in here happen on GUI closing."""
        filename = os.path.join(get_output_dir(), 'tictactoe.p')
        GameStats.save(filename, self.state.game_hist)
        event.accept()

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

    #%% Other callbacks - update_game_stats
    def update_game_stats(self, results):
        r"""
        Updates the game stats on the left of the GUI.
        """
        # calculate the number of wins
        o_wins     = np.sum(results == PLAYER['o'])
        x_wins     = np.sum(results == PLAYER['x'])
        games_tied = np.sum(results == PLAYER['draw'])

        # update the gui
        self.lbl_o_wins.setText("{}".format(o_wins))
        self.lbl_x_wins.setText("{}".format(x_wins))
        self.lbl_games_tied.setText("{}".format(games_tied))

    #%% Button callbacks
    def btn_undo_function(self): # TODO: deal with AI moves, too
        r"""Functions that executes on undo button press."""
        # get last move
        last_move = self.state.game_hist[self.state.cur_game].move_list[self.state.cur_move-1]
        if LOGGING:
            print('Undoing move = {}'.format(last_move))
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
        next_lead = PLAYER['x'] if last_lead == PLAYER['o'] else PLAYER['o']
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
        if LOGGING:
            print('Redoing move = {}'.format(redo_move))
        # place piece
        self.state.board[redo_move.row, redo_move.column] = calc_cur_move(self.state.cur_move, self.state.cur_game)
        # update current move
        self.state.cur_move += 1
        # call GUI wrapper
        self.wrapper()

    #%% mouse_click_callback
    def mouse_click_callback(self, event):
        r"""
        Function that executes on mouse click on the board axes.  Ends up placing a piece on the board.
        """
        # ignore events that are outside the axes
        if event.xdata is None or event.ydata is None:
            if LOGGING:
                print('Click is off the board.')
            return
        # test for a game that has already been concluded
        if self.state.game_hist[self.state.cur_game].winner != PLAYER['none']:
            if LOGGING:
                print('Game is over.')
            return
        # alias the rounded values of the mouse click location
        x = np.round(event.ydata).astype(int)
        y = np.round(event.xdata).astype(int)
        if LOGGING:
            print('Clicked on (x,y) = ({}, {})'.format(x, y))
        # get axes limits
        (m, n) = self.state.board.shape
        # ignore values that are outside the board
        if x < 0 or y < 0 or x >= m or y >= n:
            if LOGGING:
                print('Click is outside playable board.')
            return
        # check that move is on a free square
        if self.state.board[x, y] == PLAYER['none']:
            # make the move
            make_move(self.board_axes, self.state.board, x, y, self.state.cur_move, \
                self.state.cur_game, self.state.game_hist)
        # redraw the game/board
        self.wrapper()

    #%% wrapper
    def wrapper(self):
        r"""
        Acts as a wrapper to everything the GUI needs to do.
        """
        def sub_wrapper(self):
            r"""
            Sub-wrapper so that the wrapper can call itself for making AI moves.
            """
            # clean up an existing artifacts
            self.board_axes.clear()
            self.move_axes.clear()

            # plot the current move
            current_player = calc_cur_move(self.state.cur_move, self.state.cur_game)
            plot_cur_move(self.move_axes, current_player)
            self.move_canvas.draw()

            # draw the board
            plot_board(self.board_axes, self.state.board)

            # check for win
            (winner, win_mask) = check_for_win(self.state.board)
            # update winner
            self.state.game_hist[self.state.cur_game].winner = winner

            # plot win
            plot_win(self.board_axes, win_mask, self.state.board)

            # display relevant controls
            self.display_controls()

            # plot possible winning moves (includes updating turn arrows)
            if winner == PLAYER['none'] and OPTIONS['plot_best_moves']:
                (o_moves, x_moves) = find_moves(self.state.board)
                plot_possible_win(self.board_axes, o_moves, x_moves)

            # plot the move power
            if winner == PLAYER['none'] and OPTIONS['plot_move_power']:
                plot_powers(self.board_axes, self.state.board, o_moves, x_moves)

            # redraw with the final board
            self.board_axes.set_axis_off()
            self.board_canvas.draw()

            # update game stats on GUI
            self.update_game_stats(results=GameStats.get_results(self.state.game_hist))
            self.update()

            return (winner, current_player)

        # call the wrapper
        (winner, current_player) = sub_wrapper(self)

        # make computer AI move
        while winner == PLAYER['none'] and (\
            (OPTIONS['o_is_computer'] and current_player == PLAYER['o']) or \
            (OPTIONS['x_is_computer'] and current_player == PLAYER['x'])):
            play_ai_game(self.board_axes, self.state.board, self.state.cur_move, self.state.cur_game, \
                self.state.game_hist)
            (winner, current_player) = sub_wrapper(self)

#%% Unit Test
if __name__ == '__main__':
    # open a qapp
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run the tests
    unittest.main(module='dstauffman.games.tests.test_gui', exit=False)
    doctest.testmod(verbose=False)
    # close the qapp
    qapp.closeAllWindows()
