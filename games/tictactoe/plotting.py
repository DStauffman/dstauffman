# -*- coding: utf-8 -*-
r"""
Plotting module file for the "tictactoe" game.  It defines the plotting functions.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
from matplotlib.patches import Rectangle, Wedge, Polygon
import unittest
from dstauffman.games.tictactoe.constants import COLOR, PLAYER, SIZES

#%% plot_cur_move
def plot_cur_move(ax, move):
    r"""
    Plots the piece corresponding the current players move.

    Parameters
    ----------
    ax : object
        Axis to plot on
    move : int
        current player to move

    Examples
    --------

    >>> from dstauffman.games.tictactoe import plot_cur_move, PLAYER
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.set_xlim(-0.5, 0.5)
    >>> _ = ax.set_ylim(-0.5, 0.5)
    >>> plot_cur_move(ax, PLAYER['x'])
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
        plot_piece(ax, 0, 0, SIZES['piece'], COLOR['x'], shape=PLAYER['x'])
    elif move == PLAYER['o']:
        plot_piece(ax, 0, 0, SIZES['piece'], COLOR['o'], shape=PLAYER['o'])
    elif move == PLAYER['none']:
        pass
    else:
        raise ValueError('Unexpected player.')

    # turn the axes back off (they get reinitialized at some point)
    ax.set_axis_off()

#%% plot_piece
def plot_piece(ax, vc, hc, size, color, shape, thick=True):
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

    >>> from dstauffman.games.tictactoe import plot_piece, PLAYER
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> ax.invert_yaxis()
    >>> _ = plot_piece(ax, 1, 1, 0.9, (0, 0, 1), PLAYER['x'])
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

#%% plot_board
def plot_board(ax, board):
    r"""
    Plots the board (and the current player move).

    Parameters
    ----------
    ax : object
        Axis to plot on
    board : 2D int ndarray
        Board position

    Examples
    --------
    >>> from dstauffman.games.tictactoe import plot_board, PLAYER
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> ax.invert_yaxis()
    >>> board = PLAYER['none'] * np.ones((3, 3), dtype=int)
    >>> board[0, 0:2] = PLAYER['x']
    >>> plot_board(ax, board)
    >>> plt.show(block=False)

    >>> plt.close()

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
                plot_piece(ax, i, j, SIZES['piece'], COLOR['o'], PLAYER['o'])
            elif board[i, j] == PLAYER['x']:
                plot_piece(ax, i, j, SIZES['piece'], COLOR['x'], PLAYER['x'])
            else:
                raise ValueError('Bad board position.')

#%% plot_win
def plot_win(ax, mask, board):
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
    >>> from dstauffman.games.tictactoe import plot_win, PLAYER
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
    >>> plot_win(ax, mask, board)
    >>> plt.show(block=False)

    >>> plt.close()

    """
    (m, n) = mask.shape
    for i in range(m):
        for j in range(n):
            if mask[i, j]:
                plot_piece(ax, i, j, SIZES['piece'], COLOR['win'], board[i, j], thick=False)

#%% plot_possible_win
def plot_possible_win(ax, o_moves, x_moves):
    r"""
    Plots the possible wins on the board.

    Examples
    --------

    >>> from dstauffman.games.tictactoe import plot_possible_win, find_moves
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> ax.invert_yaxis()
    >>> board = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    >>> (o_moves, x_moves) = find_moves(board)
    >>> plot_possible_win(ax, o_moves, x_moves)
    >>> plt.show(block=False)

    >>> plt.close()

    """
    # find set of winning positions to plot
    best_power = o_moves[0].power
    pos_o = set([move for move in o_moves if move.power >= best_power])
    best_power = x_moves[0].power
    pos_x = set([move for move in x_moves if move.power >= best_power])

    # find intersecting positions
    pos_both  = pos_o & pos_x

    # plot the whole pieces
    for pos in pos_o ^ pos_both:
        plot_piece(ax, pos.row, pos.column, SIZES['piece'], COLOR['win_o'], PLAYER['o'], thick=False)
    for pos in pos_x ^ pos_both:
        plot_piece(ax, pos.row, pos.column, SIZES['piece'], COLOR['win_x'], PLAYER['x'], thick=False)

    # plot the pieces that would win for either player
    for pos in pos_both:
        plot_piece(ax, pos.row, pos.column, SIZES['piece'], COLOR['win_ox'], PLAYER['draw'], thick=False)

#%% plot_powers
def plot_powers(ax, board, o_moves, x_moves):
    r"""
    Plots the powers of each move visually on the board.

    Examples
    --------

    >>> from dstauffman.games.tictactoe import plot_powers, find_moves, plot_board
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 2.5)
    >>> _ = ax.set_ylim(-0.5, 2.5)
    >>> ax.invert_yaxis()
    >>> board = np.array([[-1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=int)
    >>> plot_board(ax, board)
    >>> (o_moves, x_moves) = find_moves(board)
    >>> plot_powers(ax, board, o_moves, x_moves)
    >>> plt.show(block=False)

    >>> plt.close()

    """
    for this_move in o_moves:
        ax.annotate('{}'.format(this_move.power), xy=(this_move.column-0.4, this_move.row-0.4), \
            xycoords='data', horizontalalignment='left', verticalalignment='center', fontsize=15, color='b')
    for this_move in x_moves:
        ax.annotate('{}'.format(this_move.power), xy=(this_move.column+0.4, this_move.row+0.4), \
            xycoords='data', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.games.tictactoe.tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
