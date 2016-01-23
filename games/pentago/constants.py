# -*- coding: utf-8 -*-
r"""
Constants module file for the "pentago" game.  It defines constants.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import doctest
import numpy as np
import unittest

#%% Constants
# color definitions
COLOR             = {}
COLOR['board']    = (1., 1., 0.)
COLOR['win']      = (1., 0., 0.)
COLOR['white']    = (1., 1., 1.)
COLOR['black']    = (0., 0., 0.)
COLOR['maj_edge'] = (0., 0., 0.)
COLOR['min_edge'] = (0., 0., 1.)
COLOR['next_wht'] = (0.6, 0.6, 1.0)
COLOR['next_blk'] = (0.0, 0.0, 0.4)
COLOR['win_wht']  = (1.0, 0.9, 0.9)
COLOR['win_blk']  = (0.2, 0.0, 0.0)

# player enumerations
PLAYER          = {}
PLAYER['white'] = 1
PLAYER['black'] = -1
PLAYER['none']  = 0
PLAYER['draw']  = 2

# sizes of the different pieces and squares
SIZES           = {}
SIZES['piece']  = 0.45
SIZES['win']    = 0.25
SIZES['square'] = 1.0
SIZES['board']  = 6
SIZES['button'] = 71 # number of pixels on rotation buttons

# Gameplay options
OPTIONS                       = {}
OPTIONS['load_previous_game'] = 'Ask' # from ['Yes','No','Ask']
OPTIONS['plot_winning_moves'] = True

# Token value for invalid board positions and such
INT_TOKEN = -101

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
LOGGING = False

#%% _rotate_board
def _rotate_board(board, quadrant, direction, inplace=True):
    r"""
    Rotates the specified board position.

    Parameters
    ----------
    board : 2D ndarray of int
        Board position
    quadrant : int
        Quadrant to rotate
    direction : int
        Direction to rotate the quadrant
    inplace : bool, optional
        Whether to update the `board` variable inplace

    Returns
    -------
    new_board : 2D ndarray of int
        New resulting board, may modify `board` inplace depending on flag

    Notes
    -----
    #.  Modifies board in-place.

    Example
    -------

    >>> from dstauffman.games.pentago import _rotate_board, PLAYER
    >>> import numpy as np
    >>> board = PLAYER['none'] * np.ones((6, 6), dtype=int)
    >>> board[1, 0:3] = PLAYER['white']
    >>> print(board[0:3, 0:3])
    [[0 0 0]
     [1 1 1]
     [0 0 0]]

    >>> quadrant = 1
    >>> direction = -1
    >>> _rotate_board(board, quadrant, direction)
    >>> print(board[0:3, 0:3])
    [[0 1 0]
     [0 1 0]
     [0 1 0]]

    """
    # check the board dimenions
    assert np.mod(SIZES['board'], 2) == 0, 'Board must be square and have an even number of rows and columns.'

    # determine if 6x6 board or 36xN
    (r, c) = board.shape  # current board size
    f = SIZES['board']    # full board size
    h = SIZES['board']//2 # half board size

    # determine if square versus linearized
    if r == f and c == f:
        assert inplace, '{0}x{0} boards must be modified inplace.'.format(f)
        # pull out the quadrant from the whole board
        if quadrant == 1:
            old_sub = board[0:h, 0:h]
        elif quadrant == 2:
            old_sub = board[0:h, h:f]
        elif quadrant == 3:
            old_sub = board[h:f, 0:h]
        elif quadrant == 4:
            old_sub = board[h:f, h:f]
        else:
            raise ValueError('Unexpected value for quadrant.')

        # rotate quadrant
        if direction == -1:
            new_sub = np.rot90(old_sub)
        elif direction == 1:
            new_sub = np.rot90(old_sub, 3)
        else:
            raise ValueError('Unexpected value for dir')

        # update rotated quadrant
        if quadrant == 1:
            board[0:h, 0:h] = new_sub
        elif quadrant == 2:
            board[0:h, h:f] = new_sub
        elif quadrant == h:
            board[h:f, 0:h] = new_sub
        elif quadrant == 4:
            board[h:f, h:f] = new_sub

    elif r == f*f:
        ix_old = np.tile(np.arange(h), h) + f * np.repeat(np.arange(h), h, axis=0)
        # pull out the quadrant from the whole board
        if quadrant == 1:
            ix_old += 0
        elif quadrant == 2:
            ix_old += h
        elif quadrant == 3:
            ix_old += (h*f)
        elif quadrant == 4:
            ix_old += (h*f +h)
        else:
            raise ValueError('Unexpected value for quad')
        # rotate quadrant
        if direction == -1:
            ix_new = ix_old[h * np.tile(np.arange(h), h) + np.repeat(np.arange(h-1, -1, -1), h, axis=0)]
        elif direction == 1:
            ix_new = ix_old[h * np.tile(np.arange(h-1, -1, -1), h) + np.repeat(np.arange(h), h, axis=0)]
        else:
            raise ValueError('Unexpected value for dir')
        # update rotated quadrant
        if inplace:
            board[ix_old, :]     = board[ix_new, :]
        else:
            new_board            = board.copy()
            new_board[ix_old, :] = board[ix_new, :]
            return new_board
    else:
        raise ValueError('Unexpected size of board.')

#%% Calculated constants
# get all possible rotation to win states
ONE_OFF = np.hstack(( \
    _rotate_board(WIN, 1, -1, inplace=False), \
    _rotate_board(WIN, 2, -1, inplace=False), \
    _rotate_board(WIN, 3, -1, inplace=False), \
    _rotate_board(WIN, 4, -1, inplace=False), \
    _rotate_board(WIN, 1,  1, inplace=False), \
    _rotate_board(WIN, 2,  1, inplace=False), \
    _rotate_board(WIN, 3,  1, inplace=False), \
    _rotate_board(WIN, 4,  1, inplace=False)))

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.games.pentago.tests.test_constants', exit=False)
    doctest.testmod(verbose=False)
