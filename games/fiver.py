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
    r"""
    Pads a piece to a given size.

    Parameters
    ----------
    piece : 2D ndarray
        Piece or board to pad
    max_size : int, or 2 element list of int
        Maximum size of each axis
    pad_value : int, optional, default is 0
        Value to use when padding

    Returns
    -------
    new_piece : 2D ndarray of int
        Padded piece or board

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

    Examples
    --------

    >>> from dstauffman.games.fiver import _pad_piece
    >>> import numpy as np
    >>> piece = np.array([[1, 1, 1, 1], [0, 0, 0, 1]], dtype=int)
    >>> max_size = 5
    >>> new_piece = _pad_piece(piece, max_size)
    >>> print(new_piece)
    [[1 1 1 1 0]
     [0 0 0 1 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]]

    """
    # determine if max_size is a scalar or specified per axis
    if np.isscalar(max_size):
        max_size = [max_size, max_size]
    # get the current size
    (i, j) = piece.shape
    # initialize the output
    new_piece = piece.copy()
    # pad the horizontal direction
    if j < max_size[0]:
        new_piece = np.hstack((new_piece, pad_value*np.ones((i, max_size[1]-j), dtype=int)))
    # pad the vertical direction
    if i < max_size[1]:
        new_piece = np.vstack((new_piece, pad_value*np.ones((max_size[0]-i, max_size[1]), dtype=int)))
    # return the resulting piece
    return new_piece

#%% Functions - _shift_piece
def _shift_piece(piece):
    r"""
    Shifts a piece to the most upper left location within an array.

    Parameters
    ----------
    piece : 2D ndarray of int
        Piece

    Returns
    -------
    new_piece : 2D ndarray of int
        Shifted piece

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

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
    Rotates a piece 90 degrees to the right.

    Parameters
    ----------
    piece : 2D ndarray of int
        Piece

    Returns
    -------
    new_piece : 2D ndarray of int
        Shifted piece

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

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
    Flips a piece about the horizontal axis.

    Parameters
    ----------
    piece : 2D ndarray of int
        Piece

    Returns
    -------
    new_piece : 2D ndarray of int
        Shifted piece

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

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
    Returns the indices to the third dimension for the unique pieces.

    Parameters
    ----------
    pieces : 3D ndarray of int
        3D array of pieces

    Returns
    -------
    ix_unique : ndarray of int
        Unique indices

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

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

#%% Functions - _display_progress
def _display_progress(ix, nums):
    r"""
    Displays the total progress to the command window.
    """
    complete = np.flipud(np.cumprod(np.flipud(nums.astype(np.float))))
    done = 0
    for i in range(len(ix)):
        done = done + ix[i]*complete[i]/nums[i]
    ratio = done / complete[0]
    print(ix)
    print('Progess: {:.6f}%'.format(ratio*100))

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

#%% Functions - is_valid
def is_valid(board, piece, use_blobbing=True):
    r"""
    Determines if the piece is valid for the given board.
    """
    if piece.ndim == 2:
        out = np.logical_not(np.any(board * piece))
        if out and use_blobbing:
            temp = board + piece
            (m, n) = board.shape
            for i in range(m):
                for j in range(n):
                    if temp[i, j] == 0:
                        if temp[i-1,j] and temp[i, j-1] and temp[i+1, j] and temp[i, j+1]:
                            out = False
                            return out
    elif piece.ndim == 3:
        temp = np.expand_dims(board, axis=2) * piece
        out = np.squeeze(np.logical_not(np.any(np.any(temp, axis=0, keepdims=True), axis=1, keepdims=True)))
    else:
        raise ValueError('Unexpected number of dimensions for piece = "{}"'.format(piece.ndim))
    return out

#%% Functions - find_all_valid_locations
def find_all_valid_locations(board, all_pieces):
    r"""
    Finds all the valid locations for each piece on the board.
    """
    (m, n) = board.shape
    max_pieces = (m - SIZE_PIECES - 1)*(n - SIZE_PIECES - 1) * NUM_ORIENTS
    locations = []
    for these_pieces in all_pieces:
        # over-allocate a possible array
        these_locs = np.zeros((m, n, max_pieces), dtype=int)
        counter = 0
        for ix in range(these_pieces.shape[2]):
            start_piece = _pad_piece(these_pieces[:,:,ix], board.shape)
            for i in range(m - SIZE_PIECES + 1):
                this_piece = np.roll(start_piece, i, axis=0)
                if is_valid(board, this_piece):
                    these_locs[:, :, counter] = this_piece
                    counter += 1
                for j in range(1, n - SIZE_PIECES + 1):
                    this_piece2 = np.roll(this_piece, j, axis=1)
                    if is_valid(board, this_piece2):
                        these_locs[:, :, counter] = this_piece2
                        counter += 1
        locations.append(these_locs[:, :, :counter])
    # resort pieces based on numbers, for lowest to highest
    sort_ix = np.array([x.shape[2] for x in locations]).argsort()
    locations = [locations[ix] for ix in sort_ix]
    return locations

#%% Functions - solve_puzzle
def solve_puzzle(board, locations):
    r"""
    Solves the puzzle for the given board and all possible piece locations.
    """
    # initialize the solutions
    solutions = []
    # create a working board
    this_board = board.copy()
    # get the number of permutations for each piece
    nums = np.array([x.shape[2] for x in locations])
    # start solving
    ix0 = np.arange(locations[0].shape[2])
    for i0 in ix0:
        np.add(this_board, locations[0][:, :, i0], this_board)
        ix1 = np.nonzero(is_valid(this_board, locations[1]))[0]
        for i1 in ix1:
            np.add(this_board, locations[1][:, :, i1], this_board)
            ix2 = np.nonzero(is_valid(this_board, locations[2]))[0]
            for i2 in ix2:
                np.add(this_board, locations[2][:, :, i2], this_board)
                ix3 = np.nonzero(is_valid(this_board, locations[3]))[0]
                for i3 in ix3:
                    np.add(this_board, locations[3][:, :, i3], this_board)
                    ix4 = np.nonzero(is_valid(this_board, locations[4]))[0]
                    for i4 in ix4:
                        _display_progress(np.array([i0, i1, i2, i3, i4]), nums)
                        np.add(this_board, locations[4][:, :, i4], this_board)
                        ix5 = np.nonzero(is_valid(this_board, locations[5]))[0]
                        for i5 in ix5:
                            np.add(this_board, locations[5][:, :, i5], this_board)
                            ix6 = np.nonzero(is_valid(this_board, locations[6]))[0]
                            for i6 in ix6:
                                np.add(this_board, locations[6][:, :, i6], this_board)
                                ix7 = np.nonzero(is_valid(this_board, locations[7]))[0]
                                for i7 in ix7:
                                    np.add(this_board, locations[7][:, :, i7], this_board)
                                    ix8 = np.nonzero(is_valid(this_board, locations[8]))[0]
                                    for i8 in ix8:
                                        np.add(this_board, locations[8][:, :, i8], this_board)
                                        ix9 = np.nonzero(is_valid(this_board, locations[9]))[0]
                                        for i9 in ix9:
                                            np.add(this_board, locations[9][:, :, i9], this_board)
                                            ix10 = np.nonzero(is_valid(this_board, locations[10]))[0]
                                            for i10 in ix10:
                                                np.add(this_board, locations[10][:, :, i10], this_board)
                                                ix11 = np.nonzero(is_valid(this_board, locations[11]))[0]
                                                for i11 in ix11:
                                                    np.add(this_board, locations[11][:, :, i11], this_board)
                                                    solutions.append(this_board)
                                                    return solutions
                                                    np.subtract(this_board, locations[11][:, :, i11], this_board)
                                                np.subtract(this_board, locations[10][:, :, i10], this_board)
                                            np.subtract(this_board, locations[9][:, :, i9], this_board)
                                        np.subtract(this_board, locations[8][:, :, i8], this_board)
                                    np.subtract(this_board, locations[7][:, :, i7], this_board)
                                np.subtract(this_board, locations[6][:, :, i6], this_board)
                            np.subtract(this_board, locations[5][:, :, i5], this_board)
                        np.subtract(this_board, locations[4][:, :, i4], this_board)
                    np.subtract(this_board, locations[3][:, :, i3], this_board)
                np.subtract(this_board, locations[2][:, :, i2], this_board)
            np.subtract(this_board, locations[1][:, :, i1], this_board)
        np.subtract(this_board, locations[0][:, :, i0], this_board)
    return solutions

#%% Functions - plot_board
def plot_board(board, title=None, opts=None):
    r"""
    Plots the board or the individual pieces.
    """
    # hard-coded square size
    box_size = 1
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
            # add the rectangle patch to the existing axis
            ax.add_patch(Rectangle((box_size*i,box_size*j),box_size, box_size, \
                facecolor=COLORS[board[i,j]], edgecolor='k'))
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
    run_tests    = True
    make_plots   = False
    make_soln    = True

    if run_tests:
        # Run docstring test
        test_docstrings()

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
    opts.show_plot = False
    # Save plots of the possible piece orientations
    if make_plots:
        setup_dir(opts.save_path, rec=True)
        for (ix, these_pieces) in enumerate(all_pieces):
            for ix2 in range(these_pieces.shape[2]):
                this_title = 'Piece {}, Permutation {}'.format(ix+1, ix2+1)
                opts.case_name = this_title
                fig = plot_board(these_pieces[:, :, ix2], title=this_title, opts=opts)
                plt.close(fig)
        # print empty boards
        opts.case_name = 'Empty Board 1'
        fig = plot_board(BOARD1[3:-3,3:-3], title='Empty Board 1', opts=opts)
        plt.close(fig)
        opts.case_name = 'Empty Board 2'
        fig = plot_board(BOARD2[3:-3,3:-3], title='Empty Board 2', opts=opts)
        plt.close(fig)

    # solve the puzzle
    locations1 = find_all_valid_locations(BOARD1, all_pieces)
    nums = np.array([x.shape[2] for x in locations1])
    print(nums)
    if make_soln:
        solutions1 = solve_puzzle(BOARD1, locations1)

    # plot all the piece locations
    if False:
        for p in [6]: #range(NUM_PIECES):
            for i in range(locations1[p].shape[2]):
                opts.case_name = 'Piece {}, Position {}'.format(p+1, i+1)
                fig = plot_board(BOARD1 + locations1[p][:,:,i], title=opts.case_name, opts=opts)
                plt.close(fig)

    if make_soln and solutions1:
        opts.case_name = 'Solution 1'
        plot_board(solutions1[0], title=opts.case_name, opts=opts)
