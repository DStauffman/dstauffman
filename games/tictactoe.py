# -*- coding: utf-8 -*-
r"""
Tic Tac Toe board game as a Python GUI.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
from matplotlib.pyplot import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Wedge, Polygon
import numpy as np
import os
import pickle
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
from dstauffman import get_images_dir, get_output_dir, Frozen

#%% Constants
# color definitions
COLOR             = {}
COLOR['board']    = (1., 1., 0.)
COLOR['win']      = (1., 0., 0.)
COLOR['o']        = (1., 1., 1.)
COLOR['x']        = (0., 0., 0.)
COLOR['edge']     = (0., 0., 0.)
COLOR['win_o']    = (1.0, 0.9, 0.9)
COLOR['win_x']    = (0.2, 0.0, 0.0)
COLOR['win_ox']   = (0.8, 0.0, 0.0)

# player enumerations
PLAYER          = {}
PLAYER['o']     = 1
PLAYER['x']     = -1
PLAYER['none']  = 0
PLAYER['draw']  = 2

# piece radius
SIZES           = {}
SIZES['piece']  = 0.9
SIZES['square'] = 1.0
SIZES['board']  = 3

# Gameplay options
OPTIONS                       = {}
OPTIONS['load_previous_game'] = 'Ask' # from ['Yes','No','Ask'] # TODO: change back to Ask eventually
OPTIONS['plot_winning_moves'] = False
OPTIONS['plot_move_power']    = False
OPTIONS['o_is_computer']      = False # TODO: needs fixing
OPTIONS['x_is_computer']      = False

# all possible winning combinations
WIN = np.array([\
[1,0,0,1,0,0,1,0],\
[1,0,0,0,1,0,0,0],\
[1,0,0,0,0,1,0,1],\
\
[0,1,0,1,0,0,0,0],\
[0,1,0,0,1,0,1,1],\
[0,1,0,0,0,1,0,0],\
\
[0,0,1,1,0,0,0,1],\
[0,0,1,0,1,0,0,0],\
[0,0,1,0,0,1,1,0],\
], dtype=bool)

# for debugging
LOGGING = True

#%% Classes - Moves
class Move(Frozen):
    r"""
    Class that keeps track of each individual move.
    """
    def __init__(self, row, column, power=None):
        self.row       = row
        self.column    = column
        self.power     = power

    def __eq__(self, other):
        r"""Equality is based on row and column."""
        return (self.row == other.row and self.column == other.column)

    def __ne__(self, other):
        r"""Inequality is based on row and column."""
        return not self.__eq__(other)

    def __lt__(self, other):
        r"""Sorts by power, then row, then column."""
        if self.power is None:
            if other.power is not None:
                return True
        else:
            if other.power is None:
                return False
            else:
                if self.power < other.power:
                    return True
                elif self.power > other.power:
                    return False
        if self.row < other.row:
            return True
        elif self.row > other.row:
            return False
        if self.column < other.column:
            return True
        elif self.column > other.column:
            return False
        return False

    def __hash__(self):
        r"""Hash uses str instead of repr, and thus power does not distinguish values."""
        return hash(self.__str__())

    def __str__(self):
        r"""String returns values except for power."""
        return 'row: {}, col: {}'.format(self.row, self.column)

    def __repr__(self):
        r"""Repr returns all values, including power."""
        return '<' + self.__str__() + ', pwr: {}'.format(self.power) + '>'

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
        r"""Calculates the number of moves in a move list."""
        return len(self.move_list)

    @staticmethod
    def get_results(game_hist):
        r"""Pulls the results out of a list of game histories."""
        return np.array([x.winner for x in game_hist])

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
cur_move  = 0 # TODO: make mutable, explicitly pass and eliminate globals
cur_game  = 0
board     = PLAYER['none'] * np.ones((3, 3), dtype=int)
game_hist = []
game_hist.append(GameStats(number=cur_game, first_move=PLAYER['o']))

#%% Classes - TicTacToeGui
class TicTacToeGui(QWidget):
    r"""
    The Tic Tac Toe GUI.
    """
    def __init__(self, **kwargs):
        super(TicTacToeGui, self).__init__(**kwargs)
        self.init()

    def init(self):
        r"""Creates the actual GUI."""

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
        self.board_canvas.mpl_connect('button_release_event', lambda event: _mouse_click_callback(self, event))
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
        wrapper(self)

        #%% GUI properties
        self.setGeometry(520, 380, 1000, 700)
        self.setWindowTitle('Tic Tac Toe')
        self.setWindowIcon(QtGui.QIcon(os.path.join(get_images_dir(),'tictactoe.png')))
        self.show()

    #%% Other callbacks
    def closeEvent(self, event):
        r"""Things in here happen on GUI closing."""
        filename = os.path.join(get_output_dir(), 'tictactoe.p')
        GameStats.save(filename, game_hist)
        event.accept()

    #%% Button callbacks
    def btn_undo_function(self):
        r"""Functions that executes on undo button press."""
        # declare globals
        global cur_move, cur_game, board
        # get last move
        last_move = game_hist[cur_game].move_list[cur_move-1]
        if LOGGING:
            print('Undoing move = {}'.format(last_move))
        # delete piece
        board[last_move.row, last_move.column] = PLAYER['none']
        # update current move
        cur_move -= 1
        # call GUI wrapper
        wrapper(self)

    def btn_new_function(self):
        r"""Functions that executes on new game button press."""
        # declare globals
        global cur_move, cur_game, board
        # update values
        last_lead = game_hist[cur_game].first_move
        next_lead = PLAYER['x'] if last_lead == PLAYER['o'] else PLAYER['o']
        assert len(game_hist) == cur_game + 1
        cur_game += 1
        cur_move = 0
        game_hist.append(GameStats(number=cur_game, first_move=next_lead, winner=PLAYER['none']))
        board = PLAYER['none'] * np.ones((SIZES['board'], SIZES['board']), dtype=int)
        # call GUI wrapper
        wrapper(self)

    def btn_redo_function(self):
        r"""Functions that executes on redo button press."""
        # declare globals
        global cur_move, cur_game, board
        # get next move
        redo_move = game_hist[cur_game].move_list[cur_move]
        if LOGGING:
            print('Redoing move = {}'.format(redo_move))
        # place piece
        board[redo_move.row, redo_move.column] = _calc_cur_move(cur_move, cur_game)
        # update current move
        cur_move += 1
        # call GUI wrapper
        wrapper(self)

#%% _mouse_click_callback
def _mouse_click_callback(self, event):
    r"""
    Function that executes on mouse click on the board axes.  Ends up placing a piece on the board.
    """
    global board, cur_move
    # ignore events that are outside the axes
    if event.xdata is None or event.ydata is None:
        if LOGGING:
            print('Click is off the board.')
        return
    # test for a game that has already been concluded
    if game_hist[cur_game].winner != PLAYER['none']:
        if LOGGING:
            print('Game is over.')
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
    # check that move is on a free square
    if board[x, y] == PLAYER['none']:
        # make the move
        _make_move(self.board_axes, board, x, y)
    # redraw the game/board
    wrapper(self)
    
#%% _make_move
def _make_move(ax, board, x, y):
    r"""
    Does the actual move.
    """
    global cur_move
    if LOGGING:
        print('Placing current piece.')
    current_player = _calc_cur_move(cur_move, cur_game)
    # update board position
    board[x, y] = current_player
    # plot the piece
    if current_player == PLAYER['o']:
        piece = _plot_piece(ax, x, y, SIZES['piece'], COLOR['o'], PLAYER['o'])
    elif current_player == PLAYER['x']:
        piece = _plot_piece(ax, x, y, SIZES['piece'], COLOR['x'], PLAYER['x'])
    else:
        raise ValueError('Unexpected player to move next.')
    assert piece
    # increment move list
    assert game_hist[cur_game].num_moves >= cur_move, \
        'Number of moves = {}, Current Move = {}'.format(game_hist[cur_game].num_moves, cur_move)
    this_move = Move(x, y)
    if game_hist[cur_game].num_moves == cur_move:
        game_hist[cur_game].add_move(this_move)
    else:
        game_hist[cur_game].move_list[cur_move] = this_move
        game_hist[cur_game].remove_moves(cur_move+1)
    # increment current move
    cur_move += 1

#%% _load_previous_game
def _load_previous_game():
    r"""
    Loads the previous game based on settings and whether it exists.
    """
    global cur_game, cur_move, board, game_hist
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
    if load_game:
        filename  = os.path.join(get_output_dir(), 'tictactoe.p')
        if os.path.isfile(filename):
            game_hist   = GameStats.load(filename)
            cur_game    = len(game_hist)-1
            cur_move    = len(game_hist[-1].move_list)
            board       = _create_board_from_moves(game_hist[-1].move_list, game_hist[-1].first_move)

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
        Current move, from {1=o, -1=x}

    Examples
    --------
    >>> from dstauffman.games.tictactoe import _calc_cur_move
    >>> move = _calc_cur_move(0, 0)
    >>> print(move)
    1

    """
    if np.mod(cur_move + cur_game, 2) == 0:
        move = PLAYER['o']
    else:
        move = PLAYER['x']
    return move

#%% _check_for_win
def _check_for_win(board):
    r"""
    Checks for a win.
    """
    # find wins
    o = np.nonzero(np.sum(np.expand_dims(board.ravel() == PLAYER['o'], axis=1) * WIN, axis=0) == 3)[0]
    x = np.nonzero(np.sum(np.expand_dims(board.ravel() == PLAYER['x'], axis=1) * WIN, axis=0) == 3)[0]

    # determine winner
    if len(o) == 0:
        if len(x) == 0:
            winner = PLAYER['none']
        else:
            winner = PLAYER['x']
    else:
        if len(x) == 0:
            winner = PLAYER['o']
        else:
            winner = PLAYER['draw']

    # check for a full game board after determining no other win was found
    if winner == PLAYER['none'] and not np.any(board == PLAYER['none']):
        winner = PLAYER['draw']

    # find winning pieces on the board
    if winner == PLAYER['none']:
        win_mask = np.zeros((3,3), dtype=bool)
    else:
        if LOGGING:
            print('Win detected.  Winner is {}.'.format(list(PLAYER.keys())[list(PLAYER.values()).index(winner)]))
        win_mask = np.reshape(np.sum(WIN[:, x], axis=1) + np.sum(WIN[:, o], axis=1), (SIZES['board'], SIZES['board'])) != 0

    # update statistics
    game_hist[cur_game].winner = winner

    return (winner, win_mask)

#%% _find_moves
def _find_moves(board):
    r"""
    Finds the best current move.

    Examples
    --------

    >>> from dstauffman.games.tictactoe import _find_moves, PLAYER, SIZES
    >>> import numpy as np
    >>> board = PLAYER['none'] * np.ones((SIZES['board'], SIZES['board']), dtype=int)
    >>> board[0, 0] = PLAYER['o']
    >>> board[0, 1] = PLAYER['o']
    >>> board[1, 1] = PLAYER['o']
    >>> (o_moves, x_moves) = _find_moves(board)

    """
    # calculate the number of total squares
    num_pieces = SIZES['board']*SIZES['board']

    # find all the available moves
    open_moves = np.arange(num_pieces)[board.ravel() == PLAYER['none']]

    # check that there are at least some available moves
    assert len(open_moves) > 0, 'At least one move must be available.'

    # expand the board to a linear 2D matrix
    big_board = np.expand_dims(board.ravel(), axis=1)

    # correlate the boards to the winning positions
    test = big_board * WIN

    # find the scores
    score = np.sum(test, axis=0)

    # test for already winning positions that shouldn't exist
    if np.any(np.abs(score) >= 3):
        raise ValueError('Board should not already be in a winning position.')

    # find the remaining possible wins
    rem_o_wins = WIN[:, np.sum((big_board == PLAYER['x']) * WIN, axis=0) == 0]
    rem_x_wins = WIN[:, np.sum((big_board == PLAYER['o']) * WIN, axis=0) == 0]

    # calculate a score for each possible move
    o_scores = np.sum(rem_o_wins, axis=1)
    x_scores = np.sum(rem_x_wins, axis=1)

    # find win in one moves
    o_win_in_one = rem_o_wins[:, np.sum(big_board * rem_o_wins, axis=0) == 2]
    x_win_in_one = rem_x_wins[:, np.sum(big_board * rem_x_wins, axis=0) == -2]
    o_pos = np.logical_and(np.logical_xor(np.abs(big_board), o_win_in_one), o_win_in_one)
    x_pos = np.logical_and(np.logical_xor(np.abs(big_board), x_win_in_one), x_win_in_one)
    o_win = np.nonzero(o_pos)[0]
    x_win = np.nonzero(x_pos)[0]

    # create the list of moves and incorporate the score
    o_moves = []
    x_moves = []
    for this_move in open_moves:
        row = this_move // SIZES['board']
        column = np.mod(this_move, SIZES['board'])
        o_score = 100 if this_move in o_win else 10 if this_move in x_win else o_scores[this_move] + x_scores[this_move]/10
        x_score = 100 if this_move in x_win else 10 if this_move in o_win else x_scores[this_move] + o_scores[this_move]/10
        o_moves.append(Move(row, column, o_score))
        x_moves.append(Move(row, column, x_score))

    # sort by best moves
    o_moves.sort(reverse=True)
    x_moves.sort(reverse=True)
    return (o_moves, x_moves)
    
#%% _play_ai_game
def _play_ai_game(ax, cur_game):
    r"""
    Computer AI based play.
    """
    global board, cur_move
    current_player = _calc_cur_move(cur_move, cur_game)
    if current_player == PLAYER['o'] and OPTIONS['o_is_computer']:
        (moves, _) = _find_moves(board)
    elif current_player == PLAYER['x'] and OPTIONS['x_is_computer']:
        (_, moves) = _find_moves(board)
    else:
        return
    this_move = moves[0]
    _make_move(ax, board, this_move.row, this_move.column)

#%% _create_board_from_moves
def _create_board_from_moves(moves, first_player):
    r"""
    Recreates a board from a move history.
    """
    # make sure the first player is valid
    assert first_player == PLAYER['x'] or first_player == PLAYER['o']
    # create the initial board
    board = PLAYER['none'] * np.ones((SIZES['board'], SIZES['board']), dtype=int)
    # alias this player
    this_player = first_player
    # loop through the move history
    for this_move in moves:
        # check that square is empty
        assert board[this_move.row, this_move.column] == PLAYER['none'], 'Invalid move encountered.'
        # place the piece
        board[this_move.row, this_move.column] = this_player
        # update the next player to move
        this_player = PLAYER['x'] if this_player == PLAYER['o'] else PLAYER['o']
    return board

#%% _update_game_stats
def _update_game_stats(self, results):
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

    Examples
    --------

    >>> from dstauffman.games.tictactoe import _plot_cur_move, PLAYER
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.set_xlim(-0.5, 0.5)
    >>> _ = ax.set_ylim(-0.5, 0.5)
    >>> _plot_cur_move(ax, PLAYER['x'])
    >>> plt.show(block=False)

    >>> plt.close()

    """
    # local alias
    box_size = SIZES['square']

    # fill background
    ax.add_patch(Rectangle((-box_size/2, -box_size/2), box_size, box_size, \
        facecolor=COLOR['board'], edgecolor='k'))

    # draw the piece
    if move == PLAYER['x']:
        _plot_piece(ax, 0, 0, SIZES['piece'], COLOR['x'], shape=PLAYER['x'])
    elif move == PLAYER['o']:
        _plot_piece(ax, 0, 0, SIZES['piece'], COLOR['o'], shape=PLAYER['o'])
    elif move == PLAYER['none']:
        pass
    else:
        raise ValueError('Unexpected player.')

    # turn the axes back off (they get reinitialized at some point)
    ax.set_axis_off()

#%% _plot_picee
def _plot_piece(ax, vc, hc, size, color, shape, thick=True):
    r"""
    Plots a piece on the board.

    Parameters
    ----------
    ax : object
        Axis to plot on
    vc : float
        Vertical center (Y-axis or board row)
    hc : float
        Horizontal center (X-axis or board column)
    size : float
        size
    color : 3-tuple
        RGB triplet color
    shape : int
        type of piece to plot

    Examples
    --------

    >>> from dstauffman.games.tictactoe import _plot_piece, PLAYER
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> _ = _plot_piece(ax, 1, 1, 0.9, (0, 0, 1), PLAYER['x'])
    >>> plt.show(block=False)

    >>> plt.close()

    """
    if thick:
        width = 0.2 # normalized units
    else:
        width = 0.1
    if shape != PLAYER['o']:
        coords1 = [(c[0]*size/2+hc, c[1]*size/2+vc) for c in [(1, 1), (-1+width, -1), (-1, -1), (1-width, 1), (1, 1)]]
        coords2 = [(c[0]*size/2+hc, c[1]*size/2+vc) for c in [(-1, 1), (-1+width, 1), (1, -1), (1-width, -1), (-1, 1)]]
    if shape == PLAYER['o']:
        # plot an O
        patch1 = Wedge((hc, vc), size/2, 0, 360, width=size*width/2, facecolor=color, edgecolor='k')
        piece = [patch1]
    elif shape == PLAYER['x']:
        # plot an X
        patch1 = Polygon(coords1, True, facecolor=color, edgecolor='k')
        patch2 = Polygon(coords2, True, facecolor=color, edgecolor='k')
        piece = [patch1, patch2]
        ax
    elif shape == PLAYER['draw']:
        # plot a combined O and X
        patch1 = Wedge((hc, vc), size/2, 0, 360, width=size*width/2, facecolor=color, edgecolor='k')
        patch2 = Polygon(coords1, True, facecolor=color, edgecolor='k')
        patch3 = Polygon(coords2, True, facecolor=color, edgecolor='k')
        piece = [patch1, patch2, patch3]
    else:
        raise ValueError('Unexpected shape.')

    # plot piece
    for this_patch in piece:
        ax.add_patch(this_patch)
    return piece

#%% _plot_board
def _plot_board(ax, board):
    r"""
    Plots the board (and the current player move).
    """
    # get axes limits
    (m, n) = board.shape
    s = SIZES['square']/2
    xmin = 0 - s
    xmax = m - 1 + s
    ymin = 0 - s
    ymax = n - 1 + s

    # fill background
    ax.add_patch(Rectangle((-xmin-1, -ymin-1), xmax-xmin, ymax-ymin, facecolor=COLOR['board'], \
        edgecolor=None))

    # draw minor horizontal lines
    ax.plot([1-s, 1-s], [ymin, ymax], color=COLOR['edge'], linewidth=2)
    ax.plot([2-s, 2-s], [ymin, ymax], color=COLOR['edge'], linewidth=2)
    # draw minor vertical lines
    ax.plot([xmin, xmax], [1-s, 1-s], color=COLOR['edge'], linewidth=2)
    ax.plot([xmin, xmax], [2-s, 2-s], color=COLOR['edge'], linewidth=2)

    # loop through and place pieces
    for i in range(m):
        for j in range(n):
            if board[i, j] == PLAYER['none']:
                pass
            elif board[i, j] == PLAYER['o']:
                _plot_piece(ax, i, j, SIZES['piece'], COLOR['o'], PLAYER['o'])
            elif board[i, j] == PLAYER['x']:
                _plot_piece(ax, i, j, SIZES['piece'], COLOR['x'], PLAYER['x'])
            else:
                raise ValueError('Bad board position.')

#%% _plot_win
def _plot_win(ax, mask, board):
    r"""
    Plots the winning pieces in red.

    Parameters
    ----------
    ax : object
        Axis to plot on
    mask : 2D bool ndarray
        Mask for which squares to plot the win
    board : 2D int ndarray
        Board position

    Examples
    --------
    >>> from dstauffman.games.tictactoe import _plot_win, PLAYER
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> ax.invert_yaxis()
    >>> mask = np.zeros((3, 3), dtype=bool)
    >>> mask[0, 0:2] = True
    >>> board = PLAYER['none'] * np.ones((3, 3), dtype=int)
    >>> board[0, 0:2] = PLAYER['x']
    >>> _plot_win(ax, mask, board)
    >>> plt.show(block=False)

    >>> plt.close()

    """
    (m, n) = mask.shape
    for i in range(m):
        for j in range(n):
            if mask[i, j]:
                _plot_piece(ax, i, j, SIZES['piece'], COLOR['win'], board[i, j], thick=False)

#%% _plot_possible_win
def _plot_possible_win(ax, o_moves, x_moves):
    r"""
    Plots the possible wins on the board.

    Examples
    --------

    >>> from dstauffman.games.tictactoe import _plot_possible_win, _find_moves
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> ax.invert_yaxis()
    >>> board = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    >>> (o_moves, x_moves) = _find_moves(board)
    >>> _plot_possible_win(ax, o_moves, x_moves) # doctest: +SKIP
    >>> plt.show(block=False)

    >>> plt.close()

    """
    # find set of winning positions to plot
    best_power = 3
    best_power = o_moves[0].power # TODO: temporary for best moves
    pos_o = set([move for move in o_moves if move.power >= best_power])
    best_power = x_moves[0].power # TODO: temporary for best moves
    pos_x = set([move for move in x_moves if move.power >= best_power])

    # find intersecting positions
    pos_both  = pos_o & pos_x

    # plot the whole pieces
    for pos in pos_o ^ pos_both:
        _plot_piece(ax, pos.row, pos.column, SIZES['piece'], COLOR['win_o'], PLAYER['o'], thick=False)
    for pos in pos_x ^ pos_both:
        _plot_piece(ax, pos.row, pos.column, SIZES['piece'], COLOR['win_x'], PLAYER['x'], thick=False)

    # plot the pieces that would win for either player
    for pos in pos_both:
        _plot_piece(ax, pos.row, pos.column, SIZES['piece'], COLOR['win_ox'], PLAYER['draw'], thick=False)
        
#%% _plot_powers
def _plot_powers(ax, board, o_moves, x_moves):
    r"""
    Plots the powers of each move visually on the board.
    
    Examples
    --------

    >>> from dstauffman.games.tictactoe import _plot_powers, _find_moves, _plot_board
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> ax.invert_yaxis()
    >>> board = np.array([[-1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=int)
    >>> _plot_board(ax, board)
    >>> (o_moves, x_moves) = _find_moves(board)
    >>> _plot_powers(ax, board, o_moves, x_moves)
    >>> plt.show(block=False)

    >>> plt.close()
    
    """
    for this_move in o_moves:
        ax.annotate('{}'.format(this_move.power), xy=(this_move.column-0.4, this_move.row-0.4), \
            xycoords='data', horizontalalignment='center', verticalalignment='center', fontsize=15, color='b')
    for this_move in x_moves:
        ax.annotate('{}'.format(this_move.power), xy=(this_move.column+0.4, this_move.row+0.4), \
            xycoords='data', horizontalalignment='center', verticalalignment='center', fontsize=15, color='k')

#%% move_wrapper
def wrapper(self):
    r"""
    Acts as a wrapper to everything the GUI needs to do.
    """
    # clean up an existing artifacts
    self.board_axes.clear()
    self.move_axes.clear()

    # plot the current move
    _plot_cur_move(self.move_axes, _calc_cur_move(cur_move, cur_game))
    self.move_canvas.draw()

    # draw the board
    _plot_board(self.board_axes, board)

    # check for win
    (winner, win_mask) = _check_for_win(board)

    # plot win
    _plot_win(self.board_axes, win_mask, board)

    # display relevant controls
    _display_controls(self)

    # plot possible winning moves (includes updating turn arrows)
    if winner == PLAYER['none'] and OPTIONS['plot_winning_moves']:
        (o_moves, x_moves) = _find_moves(board)
        _plot_possible_win(self.board_axes, o_moves, x_moves)
        
    # plot the move power
    if winner == PLAYER['none'] and OPTIONS['plot_move_power']:
        _plot_powers(self.board_axes, board, o_moves, x_moves)

    # redraw with the final board
    self.board_axes.set_axis_off()
    self.board_canvas.draw()

    # update game stats on GUI
    _update_game_stats(self, results=GameStats.get_results(game_hist))
    self.update()
    
    # make computer AI move
    if winner == PLAYER['none'] and (OPTIONS['o_is_computer'] or OPTIONS['x_is_computer']):
        _play_ai_game(self.board_axes, cur_game) # TODO: make this work
        self.update()

#%% Unit Test
if __name__ == '__main__':
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'test'
    if mode == 'run':
        # Runs the GUI application
        qapp = QApplication(sys.argv)
        # load the previous game content
        _load_previous_game()
        # instatiates the GUI
        gui = TicTacToeGui()
        gui.show()
        sys.exit(qapp.exec_())
    elif mode == 'test':
        # open a qapp
        if QApplication.instance() is None:
            qapp = QApplication(sys.argv)
        else:
            qapp = QApplication.instance()
        # run the tests
        unittest.main(module='dstauffman.games.test_tictactoe', exit=False)
        doctest.testmod(verbose=False)
        # close the qapp
        qapp.closeAllWindows()
        qapp.exit()
    elif mode == 'null':
        pass
    else:
        raise ValueError('Unexpected mode of "{}".'.format(mode))
