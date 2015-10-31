# -*- coding: utf-8 -*-
"""
The "fiver" file solves the geometric puzzle with twelve pieces made of all unique combinations of
five squares that share adjacent edges.  Then these 12 pieces are layed out into boards of sixty
possible places in different orientations.  I'm unaware of a generic name for this game.

Notes
-----
#.  Written by David C. Stauffer in October 2015 when he found the puzzle on his dresser while
    acquiring and rearranging some furniture.
"""
# pylint: disable=E1101, C0326, C0103

#%% Imports
# backwards compatibility
from __future__ import print_function
from __future__ import division
# regular imports
from datetime import datetime
import doctest
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import shutil
# model imports
from dstauffman import Opts, setup_plots, setup_dir, get_root_dir, ColorMap

#%% Hard-coded values
SIZE_PIECES = 5
NUM_PIECES  = 12
NUM_ORIENTS = 8
# build colormap
cm = ColorMap('Paired', 0, NUM_PIECES-1)
COLORS = ['w'] + [cm.get_color(i) for i in range(NUM_PIECES)] + ['k']
# make boards
BOARD1 = (NUM_PIECES+1)*np.ones((14, 18), dtype=int)
BOARD1[4:10,4:14] = 0
BOARD2 = (NUM_PIECES+1)*np.ones((16, 16), dtype=int)
BOARD2[4:12, 4:12] = 0
BOARD2[7:9, 7:9] = NUM_PIECES+1

#%% Functions - _pad_piece
def _pad_piece(piece, max_size, pad_value=0):
    if np.isscalar(max_size):
        max_size = [max_size, max_size]
    (i, j) = piece.shape
    new_piece = piece.copy()
    if j < max_size[0]:
        new_piece = np.hstack((new_piece, pad_value*np.ones((i, max_size[1]-j), dtype=int)))
    if i < max_size[1]:
        new_piece = np.vstack((new_piece, pad_value*np.ones((max_size[0]-i, max_size[1]), dtype=int)))
    return new_piece

#%% Functions - _shift_piece
def _shift_piece(piece):
    r"""
    TBD

    Examples
    --------

    >>> from dstauffman.games.fiver import _shift_piece
    >>> import numpy as np
    >>> x = np.zeros((5,5), dtype=int)
    >>> x[1, :] = 1
    >>> y = _shift_piece(x)
    >>> print(y)
    [[1 1 1 1 1]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]]

    """
    new_piece = piece.copy()
    ix = [1, 2, 3, 4, 0]
    while np.all(new_piece[0, :] == 0):
        new_piece = new_piece[ix, :]
    while np.all(new_piece[:, 0] == 0):
        new_piece = new_piece[:, ix]
    return new_piece

#%% Functions - _rotate_piece
def _rotate_piece(piece):
    r"""
    TBD

    Examples
    --------

    >>> from dstauffman.games.fiver import _rotate_piece
    >>> import numpy as np
    >>> x = np.arange(25).reshape((5, 5))
    >>> y = _rotate_piece(x)
    >>> print(y)
    [[20 15 10  5  0]
     [21 16 11  6  1]
     [22 17 12  7  2]
     [23 18 13  8  3]
     [24 19 14  9  4]]

    """
    # build the correct map based on the given axis
    map_ = np.array([20, 15, 10,  5,  0, 21, 16, 11,  6,  1, 22, 17, 12,  7,  2, \
        23, 18, 13,  8,  3, 24, 19, 14,  9,  4])
    # rotate the piece by using the map and return
    temp_piece = piece.ravel()[map_].reshape((SIZE_PIECES, SIZE_PIECES))
    # shift to upper left most position
    new_piece = _shift_piece(temp_piece)
    return new_piece

#%% Functions - _flip_piece
def _flip_piece(piece):
    r"""
    TBD

    Examples
    --------

    >>> from dstauffman.games.fiver import _flip_piece
    >>> import numpy as np
    >>> x = np.arange(25).reshape((5, 5))
    >>> y = _flip_piece(x)
    >>> print(y)
    [[20 21 22 23 24]
     [15 16 17 18 19]
     [10 11 12 13 14]
     [ 5  6  7  8  9]
     [ 0  1  2  3  4]]

    """
    ix = [4, 3, 2, 1, 0]
    temp_piece = piece[ix, :]
    # shift to upper left most position
    new_piece = _shift_piece(temp_piece)
    return new_piece

#%% Functions - _get_unique_pieces
def _get_unique_pieces(pieces):
    r"""
    TBD

    Examples
    --------

    >>> from dstauffman.games.fiver import _get_unique_pieces, _rotate_piece
    >>> import numpy as np
    >>> pieces = np.zeros((5, 5, 3), dtype=int)
    >>> pieces[0, :, 0] = 1
    >>> pieces[1, :, 1] = 1
    >>> pieces[0, :, 2] = 1
    >>> ix_unique = _get_unique_pieces(pieces)
    >>> print(ix_unique)
    [0, 1]

    """
    # find the number of pieces
    num = pieces.shape[2]
    # initialize some lists
    ix_unique = []
    sets = []
    # loop through pieces
    for ix in range(num):
        # alias this piece
        this_piece = pieces[:, :, ix]
        # find the indices in the single vector version of the array and convert to a unique set
        inds = set(np.nonzero(this_piece.ravel())[0])
        # see if this set is in the master set
        if inds not in sets:
            # if not, then this is a new unique piece, so keep it
            ix_unique.append(ix)
            sets.append(inds)
    return ix_unique

#%% Functions - _draw_cube
def _draw_cube(ax, xs=0, ys=0, color='k'):
    r"""
    Draws a plot of the square at a possibly shifted position, with a given color.

    Parameters
    ----------
    ax : matplotlib.pyplot.figure.axis
        Axis to draw the cube on, axis must be a 3D projection
    xs : int, optional
        Amount to shift the cube in the X direction
    ys : int, optional
        Amount to shift the cube in the Y direction
    color : str, optional
        Color code to use to override the default of black

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

    Examples
    --------

    >>> from dstauffman.games.fiver import _draw_cube
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _draw_cube(ax)

    Close the plot
    >>> plt.close(fig)

    """
    box_size = 1
    ax.add_patch(Rectangle((box_size*xs,box_size*ys),box_size, box_size, \
        facecolor=color, edgecolor='k'))

#%% Functions - make_all_pieces
def make_all_pieces():
    r"""
    TBD
    """
    # Hard-coded values
    p1 = np.array([[1, 1, 1, 1, 1]])
    p2 = np.array([\
        [1, 1, 1, 1], \
        [0, 0, 0, 1]])
    p3 = np.array([\
        [1, 1, 1, 1], \
        [0, 0, 1, 0]])
    p4 = np.array([\
        [1, 1, 1, 0], \
        [0, 0, 1, 1]])
    p5 = np.array([\
        [1, 1, 1], \
        [1, 0, 0], \
        [1, 0, 0]])
    p6 = np.array([\
        [1, 1, 1], \
        [0, 1, 0], \
        [0, 1, 0]])
    p7 = np.array([\
        [0, 1, 0], \
        [1, 1, 1], \
        [0, 1, 0]])
    p8 = np.array([\
        [1, 1, 1], \
        [1, 1, 0]])
    p9 = np.array([\
        [1, 1, 0], \
        [0, 1, 0], \
        [0, 1, 1]])
    p10 = np.array([\
        [1, 1], \
        [1, 0], \
        [1, 1]])
    p11 = np.array([\
        [1, 1, 0], \
        [0, 1, 1], \
        [0, 1, 0]])
    p12 = np.array([\
        [0, 1, 1], \
        [1, 1, 0], \
        [1, 0, 0]])
    # loop through all these and make to 5x5
    pieces = -1 * np.ones((SIZE_PIECES, SIZE_PIECES, NUM_PIECES), dtype=int);
    for (ix, this_piece) in enumerate([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]):
        new_piece = _pad_piece(this_piece, SIZE_PIECES)
        pieces[:, :, ix] = (ix + 1) * new_piece
    return pieces

#%% Functions - make_all_permutations
def make_all_permutations(pieces):
    r"""
    TBD
    """
    # initialize the output
    all_pieces = []
    # loop through all the pieces
    for ix in range(NUM_PIECES):
        # preallocate the array
        all_this_piece = -1 * np.ones((SIZE_PIECES, SIZE_PIECES, NUM_ORIENTS), dtype=int)
        # alias this piece
        this_piece = pieces[:, :, ix]
        # find the number of rotations (4)
        rots = NUM_ORIENTS//2
        # do the rotations and keep each piece
        for counter in range(rots):
            this_piece = _rotate_piece(this_piece)
            all_this_piece[:, :, counter] = this_piece
        # flip the piece
        this_piece = _flip_piece(this_piece)
        # do another set of rotations and keep each piece
        for counter in range(rots):
            this_piece = _rotate_piece(this_piece)
            all_this_piece[:, :, counter+rots] = this_piece
        # find the indices to the unique pieces
        ix_unique = _get_unique_pieces(all_this_piece)
        # gather the unique combinations
        all_pieces.append(all_this_piece[:, :, ix_unique])
    return all_pieces

#%% Functions - plot_board
def plot_board(board, title=None, opts=None):
    r"""
    Plots the board or the individual pieces.
    """
    # check for opts
    if opts is None:
        opts = Opts()
    # turn interactive plotting off
    plt.ioff()
    # create the figure
    fig = plt.figure()
    # create the axis
    ax = fig.add_subplot(111)
    # set the title
    if title is not None:
        fig.canvas.set_window_title(title)
        plt.title(title)
    # draw each square
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            _draw_cube(ax, xs=i, ys=j, color=COLORS[board[i,j]])
    # make square
    plt.axis('equal')
    # set limits
    plt.xlim(0, board.shape[0])
    plt.ylim(0, board.shape[1])
    # flip the vertical axis
    ax.invert_yaxis()

    # configure the plot
    setup_plots(fig, opts)

    # return the resulting figure handle
    return fig

#%% Functions - test_docstrings
def test_docstrings():
    r"""
    Tests the docstrings within this file.
    """
    file = os.path.join(get_root_dir(), 'games', 'fiver.py')
    doctest.testfile(file, report=True, verbose=False, module_relative=True)

#%% Main script
if __name__ == '__main__':
    # flags for running code
    run_tests    = False
    make_plots   = True
    make_soln    = True

    if run_tests:
        # Run docstring test
        test_docstrings()

    if make_soln:
        # make all the pieces
        pieces = make_all_pieces()
        # create all the possible permutations of all the pieces
        all_pieces = make_all_permutations(pieces)

        # Create and set Opts
        date = datetime.now()
        opts = Opts()
        opts.case_name = 'Board'
        opts.save_path = os.path.join(get_root_dir(), 'results', date.strftime('%Y-%m-%d'))
        opts.save_plot = True
        opts.show_plot = True
        # Save plots of the possible piece orientations
        if make_plots:
            setup_dir(opts.save_path, rec=True)
            for (ix, these_pieces) in enumerate(all_pieces):
                for ix2 in range(these_pieces.shape[2]):
                    this_title = 'Piece {}, Permutation {}'.format(ix+1, ix2+1)
                    opts.case_name = this_title
                    plot_board(these_pieces[:, :, ix2], title=this_title, opts=opts)
            # print empty boards
            opts.case_name = 'Empty Board 1'
            plot_board(BOARD1[3:-3,3:-3], title='Empty Board 1', opts=opts)
            opts.case_name = 'Empty Board 2'
            plot_board(BOARD2[3:-3,3:-3], title='Empty Board 2', opts=opts)

        # solve the puzzle
        pass # TODO: solve puzzle
