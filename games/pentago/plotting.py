# -*- coding: utf-8 -*-
r"""
Plotting module file for the "pentago" game.  It defines the plotting functions.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
from matplotlib.patches import Circle, Rectangle, Wedge
import unittest
from dstauffman.games.pentago.constants import COLOR, PLAYER, SIZES
from dstauffman.games.pentago.utils     import calc_cur_move

#%% plot_cur_move
def plot_cur_move(ax, move):
    r"""
    Plots the piece corresponding the current players move.
    """
    # local alias
    box_size = SIZES['square']

    # fill background
    ax.add_patch(Rectangle((-box_size/2, -box_size/2), box_size, box_size, \
        facecolor=COLOR['board'], edgecolor='k'))

    # draw the piece
    if move == PLAYER['white']:
        plot_piece(ax, 0, 0, SIZES['piece'], COLOR['white'])
    elif move == PLAYER['black']:
        plot_piece(ax, 0, 0, SIZES['piece'], COLOR['black'])
    elif move == PLAYER['none']:
        pass
    else:
        raise ValueError('Unexpected player.')

    # turn the axes back off (they get reinitialized at some point)
    ax.set_axis_off()

#%% plot_piece
def plot_piece(ax, vc, hc, r, c, half=False):
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
    >>> from dstauffman.games.pentago import plot_piece
    >>> import matplotlib.pyplot as plt
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.set_xlim(0.5, 1.5)
    >>> _ = ax.set_ylim(0.5, 1.5)
    >>> obj = plot_piece(ax, 1, 1, 0.45, (0, 0, 1))
    >>> plt.show(block=False) # doctest: +SKIP

    >>> plt.close()

    """

    # theta angle to sweep out 2*pi
    if half:
        piece = Wedge((hc, vc), r, 270, 90, facecolor=c, edgecolor='k')
    else:
        piece = Circle((hc, vc), r, facecolor=c, edgecolor='k')

    # plot piece
    ax.add_patch(piece)
    return piece

#%% plot_board
def plot_board(ax, board):
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
                plot_piece(ax, i, j, SIZES['piece'], COLOR['white'])
            elif board[i, j] == PLAYER['black']:
                plot_piece(ax, i, j, SIZES['piece'], COLOR['black'])
            else:
                raise ValueError('Bad board position.')

#%% plot_win
def plot_win(ax, mask):
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
    >>> from dstauffman.games.pentago import plot_win
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 5.5)
    >>> _ = ax.set_ylim(-0.5, 5.5)
    >>> ax.invert_yaxis()
    >>> mask = np.zeros((6, 6), dtype=bool)
    >>> mask[0, 0:5] = True
    >>> plot_win(ax, mask)
    >>> plt.show(block=False) # doctest: +SKIP

    >>> plt.close()

    """
    (m, n) = mask.shape
    for i in range(m):
        for j in range(n):
            if mask[i, j]:
                plot_piece(ax, i, j, SIZES['win'], COLOR['win'])

#%% plot_possible_win
def plot_possible_win(ax, rot_buttons, white_moves, black_moves, cur_move, cur_game):
    r"""
    Plots the possible wins on the board.

    Examples
    --------

    >>> from dstauffman.games.pentago import plot_possible_win, find_moves
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> _ = ax.set_xlim(-0.5, 5.5)
    >>> _ = ax.set_ylim(-0.5, 5.5)
    >>> ax.invert_yaxis()
    >>> board = np.reshape(np.hstack((0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, np.zeros(24))), (6, 6))
    >>> (white_moves, black_moves) = find_moves(board)
    >>> rot_buttons = dict() # TODO: write this # doctest: +SKIP
    >>> cur_move = 0
    >>> cur_game = 0
    >>> plot_possible_win(ax, rot_buttons, white_moves, black_moves, cur_move, cur_game) # doctest: +SKIP
    >>> plt.show(block=False) # doctest: +SKIP

    >>> plt.close()

    """
    # find set of positions to plot
    pos_white = set(white_moves)
    pos_black = set(black_moves)
    # find intersecting positions
    pos_both  = pos_white & pos_black

    # plot the whole pieces
    for this_move in pos_white ^ pos_both:
        plot_piece(ax, this_move.row, this_move.column, SIZES['win'], COLOR['win_wht'])
        rot_buttons[this_move.rot_key].overlay = 'wht'
    for this_move in pos_black ^ pos_both:
        plot_piece(ax, this_move.row, this_move.column, SIZES['win'], COLOR['win_blk'])
        rot_buttons[this_move.rot_key].overlay = 'blk'

    # plot the half pieces, with the current players move as whole
    next_move = calc_cur_move(cur_move, cur_game)
    if next_move == PLAYER['white']:
        for this_move in pos_both:
            plot_piece(ax, this_move.row, this_move.column, SIZES['win'], COLOR['win_wht'])
            plot_piece(ax, this_move.row, this_move.column, SIZES['win'], COLOR['win_blk'], half=True)
            rot_buttons[this_move.rot_key].overlay = 'w_b'
    elif next_move == PLAYER['black']:
        for this_move in pos_both:
            plot_piece(ax, this_move.row, this_move.column, SIZES['win'], COLOR['win_blk'])
            plot_piece(ax, this_move.row, this_move.column, SIZES['win'], COLOR['win_wht'], half=True)
            rot_buttons[this_move.rot_key].overlay = 'b_w'
    else:
        raise ValueError('Unexpected next player.')

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.games.pentago.tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
