# -*- coding: utf-8 -*-
"""
The "knight" file solves the Knight Board puzzle given to Matt Beck during a job interview.

Notes
-----
#.  Written by David C. Stauffer and Matt Beck in September 2015.

[Knight Board]
The knight board can be represented in x,y coordinates.  The upper left position
is (0,0) and the bottom right is (7,7).  Assume there is a single knight chess
piece on the board that can move according to chess rules.  Sample S[tart] and
E[nd] points are shown below:
    . . . . . . . .
    . . . . . . . .
    . S . . . . . .
    . . . . . . . .
    . . . . . E . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
Level 1: Write a function that accepts a sequence of moves and reports
    whether the sequence contains only valid knight moves.  It should also
    optionally print the state of the knight board to the terminal as shown
    above after each move.  The current position should be marked with a 'K'.
Level 2: Compute a valid sequence of moves from a given start point to a given
    end point.
Level 3: Compute a valid sequence of moves from a given start point to a
    given end point in the fewest number of moves.
Level 4: Now repeat the Level 3 task for this 32x32 board.  Also, modify
    your validator from Level 1 to check your solutions.  This board has the
    following additional rules:
        1) W[ater] squares count as two moves when a piece lands there
        2) R[ock] squares cannot be used
        3) B[arrier] squares cannot be used AND cannot lie in the path
        4) T[eleport] squares instantly move you from one T to the other in
            the same move
        5) L[ava] squares count as five moves when a piece lands there
    . . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
    . . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
    . . . . . . . . B . . . L L L . . . L L L . . . . . . . . . . .
    . . . . . . . . B . . . L L L . . L L L . . . R R . . . . . . .
    . . . . . . . . B . . . L L L L L L L L . . . R R . . . . . . .
    . . . . . . . . B . . . L L L L L L . . . . . . . . . . . . . .
    . . . . . . . . B . . . . . . . . . . . . R R . . . . . . . . .
    . . . . . . . . B B . . . . . . . . . . . R R . . . . . . . . .
    . . . . . . . . W B B . . . . . . . . . . . . . . . . . . . . .
    . . . R R . . . W W B B B B B B B B B B . . . . . . . . . . . .
    . . . R R . . . W W . . . . . . . . . B . . . . . . . . . . . .
    . . . . . . . . W W . . . . . . . . . B . . . . . . T . . . . .
    . . . W W W W W W W . . . . . . . . . B . . . . . . . . . . . .
    . . . W W W W W W W . . . . . . . . . B . . R R . . . . . . . .
    . . . W W . . . . . . . . . . B B B B B . . R R . W W W W W W W
    . . . W W . . . . . . . . . . B . . . . . . . . . W . . . . . .
    W W W W . . . . . . . . . . . B . . . W W W W W W W . . . . . .
    . . . W W W W W W W . . . . . B . . . . . . . . . . . . B B B B
    . . . W W W W W W W . . . . . B B B . . . . . . . . . . B . . .
    . . . W W W W W W W . . . . . . . B W W W W W W B B B B B . . .
    . . . W W W W W W W . . . . . . . B W W W W W W B . . . . . . .
    . . . . . . . . . . . B B B . . . . . . . . . . B B . . . . . .
    . . . . . R R . . . . B . . . . . . . . . . . . . B . . . . . .
    . . . . . R R . . . . B . . . . . . . . . . . . . B . T . . . .
    . . . . . . . . . . . B . . . . . R R . . . . . . B . . . . . .
    . . . . . . . . . . . B . . . . . R R . . . . . . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .  # The last four rows originally missing
    . . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
    . . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
    . . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .
Level 5 [HARD]: Compute the longest sequence of moves to complete Level 3 without
    visiting the same square twice.  Use the 32x32 board.
"""
# pylint: disable=C0326, C0103, C0301, E1101

#%% Imports
# backwards compatibility with v2.7
from __future__ import print_function
from __future__ import division
# regular imports
import doctest
from enum import unique, IntEnum
import numpy as np
import time
import unittest

#%% Constants
# hard-coded values
LARGE_INT = 1000000
LOGGING = False # for debugging the recursive solver
# dictionaries
CHAR_DICT = {'.':0, 'S':1, 'E':2, 'K':3, 'W':4, 'R':5, 'B':6, 'T':7, 'L':8, 'x': 9}
NUM_DICT  = {value:key for (key, value) in CHAR_DICT.items()}
COST_DICT = {'normal': 1, 'transport': 1, 'water': 2, 'lava': 5, 'invalid': LARGE_INT, 'start': 0}
# boards
BOARD1 = r"""
. . . . . . . .
. . . . . . . .
. S . . . . . .
. . . . . . . .
. . . . . E . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
"""
BOARD2 = r"""
. . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
. . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .
. . . . . . . . B . . . L L L . . . L L L . . . . . . . . . . .
. . . . . . . . B . . . L L L . . L L L . . . R R . . . . . . .
. . . . . . . . B . . . L L L L L L L L . . . R R . . . . . . .
. . . . . . . . B . . . L L L L L L . . . . . . . . . . . . . .
. . . . . . . . B . . . . . . . . . . . . R R . . . . . . . . .
. . . . . . . . B B . . . . . . . . . . . R R . . . . . . . . .
. . . . . . . . W B B . . . . . . . . . . . . . . . . . . . . .
. . . R R . . . W W B B B B B B B B B B . . . . . . . . . . . .
. . . R R . . . W W . . . . . . . . . B . . . . . . . . . . . .
. . . . . . . . W W . . . . . . . . . B . . . . . . T . . . . .
. . . W W W W W W W . . . . . . . . . B . . . . . . . . . . . .
. . . W W W W W W W . . . . . . . . . B . . R R . . . . . . . .
. . . W W . . . . . . . . . . B B B B B . . R R . W W W W W W W
. . . W W . . . . . . . . . . B . . . . . . . . . W . . . . . .
W W W W . . . . . . . . . . . B . . . W W W W W W W . . . . . .
. . . W W W W W W W . . . . . B . . . . . . . . . . . . B B B B
. . . W W W W W W W . . . . . B B B . . . . . . . . . . B . . .
. . . W W W W W W W . . . . . . . B W W W W W W B B B B B . . .
. . . W W W W W W W . . . . . . . B W W W W W W B . . . . . . .
. . . . . . . . . . . B B B . . . . . . . . . . B B . . . . . .
. . . . . R R . . . . B . . . . . . . . . . . . . B . . . . . .
. . . . . R R . . . . B . . . . . . . . . . . . . B . T . . . .
. . . . . . . . . . . B . . . . . R R . . . . . . B . . . . . .
. . . . . . . . . . . B . . . . . R R . . . . . . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
. . . . . . . . . . . B . . . R R . . . . . . . . . . . . . . .
. . . . . . . . . . . B . . . . . . . . . . . . . . . . . . . .
"""
# moves
#MOVES = frozenset({-4, -3, -2, -1, 1, 2, 3, 4})
MOVES = [-4, -3, -2, -1, 1, 2, 3, 4]
#MOVES = frozenset({-4, -3, -2, -1, 1, 2, 3, 4, -5, 5, -6, 6, -7, 7, -8, 8})
# Move design is based on a first direction step of two moves, followed by a single step to the
# left or right of the first one.
# Note for chess nerds: The move is two steps and then one.  With no obstructions, it can be thought
# of as one and then two, but with obstructions, this counts as a different move and is not allowed.
# The number represents the cardinal direction of the move: 1=North, 2=East, 3=South, 4=West
# The sign of the number represents whether the second step is left or right (positive is clockwise)
# Therefore:
#  Move -1       Move +1        Move -2      Move +2       Move -3       Move +3       Move -4       Move +4
# . E x . .  |  . . x E .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .
# . . x . .  |  . . x . .  |  . . . . E  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  E . . . .
# . . S . .  |  . . S . .  |  . . S x x  |  . . S x x  |  . . S . .  |  . . S . .  |  x x S . .  |  x x S . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . . . E  |  . . x . .  |  . . x . .  |  E . . . .  |  . . . . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . x E .  |  . E x . .  |  . . . . .  |  . . . . .

# Optional extended set?
#  Move -5       Move +5        Move -6      Move +6       Move -7       Move +7       Move -8       Move +8
# . E . . .  |  . . . E .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .
# . x . . .  |  . . . x .  |  . . x x E  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  E x x . .
# . x S . .  |  . . S x .  |  . . S . .  |  . . S . .  |  . . S x .  |  . x S . .  |  . . S . .  |  . . S . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . x x E  |  . . . x .  |  . x . . .  |  E x x . .  |  . . . . .
# . . . . .  |  . . . . .  |  . . . . .  |  . . . . .  |  . . . E .  |  . E . . .  |  . . . . .  |  . . . . .

#%% Classes - Piece
@unique
class Piece(IntEnum):
    r"""
    Enumerator for all the possible types of squares within the board, including start and end
    positions
    """
    null      = 0 # Empty square that has never been used
    start     = 1 # Original starting position
    final     = 2 # Final ending position
    current   = 3 # Current knight position
    water     = 4 # water, costs two moves to land on
    rock      = 5 # rock, can be jumped across, but not landed on
    barrier   = 6 # barrier, cannot be landed on or moved across
    transport = 7 # transport to the other transport, can only be exactly 0 or 2 on the board
    lava      = 8 # lava, costs 5 moves to land on
    visited   = 9 # previously visited square that cannot be used again

#%% Classes - Move
@unique
class Move(IntEnum):
    r"""
    Enumerator for all the cost outcomes for moving a piece.
    """
    off_board = -2
    blocked   = -1
    visited   = 0
    normal    = 1
    transport = 2
    water     = 3
    lava      = 4
    winning   = 5

#%% alternates for speed
Piece2 = {x.name: x.value for x in Piece}
Move2  = {x.name: x.value for x in Move}

#%% _board_to_costs
def _board_to_costs(board):
    r"""
    Translates a board to the associated costs for landing on any square within the board.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    costs : 2D ndarray of int
        Costs for each square on the board layout

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _board_to_costs, Piece
    >>> import numpy as np
    >>> board = Piece.null * np.ones((3, 3), dtype=int)
    >>> board[0, 0] = Piece.water
    >>> board[1, 1] = Piece.start
    >>> board[2, 2] = Piece.barrier
    >>> costs = _board_to_costs(board)
    >>> print(costs) # doctest: +NORMALIZE_WHITESPACE
    [[      2       1       1]
     [      1       0       1]
     [      1       1 1000000]]

    """
    costs = np.empty(board.shape, dtype=int)
    costs.fill(COST_DICT['invalid'])
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            this_piece = board[i, j]
            if this_piece in {Piece.rock, Piece.barrier}:
                continue
            elif this_piece in {Piece.null, Piece.final}:
                costs[i, j] = COST_DICT['normal']
            elif this_piece == Piece.start:
                costs[i, j] = COST_DICT['start']
            elif this_piece == Piece.transport:
                costs[i, j] = COST_DICT['transport']
            elif this_piece == Piece.water:
                costs[i, j] = COST_DICT['water']
            elif this_piece == Piece.lava:
                costs[i, j] = COST_DICT['lava']
            else:
                raise ValueError('Cannot convert piece "{}" to a cost.'.format(this_piece))
    return costs

#%% _get_transports
def _get_transports(board):
    r"""
    Gets the locations of all the transports.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    transports : 2 element list of 1x2 tuples
        Location of the transports

    Notes
    -----
    #.  Written by David C. Stauffer in November 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _get_transports, Piece
    >>> import numpy as np
    >>> board = Piece.null * np.ones((3, 3), dtype=int)
    >>> board[0, 1] = Piece.transport
    >>> board[2, 2] = Piece.transport
    >>> transport = _get_transports(board)
    >>> print(transport)
    [(0, 1), (2, 2)]


    """
    if np.any(board == Piece.transport):
        ix = np.nonzero(board == Piece.transport)
        assert len(ix) == 2, 'Must be a 2D board.'
        assert len(ix[0]) == 0 or len(ix[0]) == 2, 'There must be 0 or exactly 2 transports.'
        transports = [(ix[0][i], ix[1][i]) for i in range(len(ix[0]))]
    else:
        transports = None
    return transports

#%% _get_current_position
def _get_current_position(board):
    r"""
    Gets the current position of the knight.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    x : int
        Current X position
    y : int
        Current Y position

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _get_current_position, Piece
    >>> import numpy as np
    >>> board = np.zeros((5,5), dtype=int)
    >>> board[1,4] = Piece.current
    >>> (x, y) = _get_current_position(board)
    >>> print(x, y)
    1 4

    """
    # find position
    ix = np.nonzero(board == Piece2['current'])
    # alias X and Y locations
    x = ix[0]
    y = ix[1]
    # check that only exactly one current position was found, and if not, print the messed up board
    count = np.count_nonzero(board == Piece2['current'])
    is_valid = count == 1
    if not is_valid:
        print_board(board)
    assert is_valid, 'Only exactly one current position may be found, not {}.'.format(count)
    # otherwise if all is good, return the location
    return (x[0], y[0])

#%% _get_new_position
def _get_new_position(x, y, move, transports):
    r"""
    Gets the new position of the knight after making the desired move.

    Parameters
    ----------
    x : int
        Current X position
    y : int
        Current Y position
    move : int
        Move to be performed

    Returns
    -------
    pos1 : (x,y) tuple
        X and Y positions for step 1 in the move
    pos2 : (x,y) tuple
        X and Y positions for step 2 in the move
    pos3 : (x,y) tuple
        X and Y positions for step 3 in the move

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _get_new_position
    >>> x = 2
    >>> y = 3
    >>> move = 2 # (2 right and 1 down)
    >>> transports = None
    >>> (pos1, pos2, pos3) = _get_new_position(x, y, move, transports)
    >>> print(pos1, pos2, pos3)
    (2, 4) (2, 5) (3, 5)

    """
    # move the piece
    if move == -1:
        pos1 = (x-1, y)
        pos2 = (x-2, y)
        pos3 = (x-2, y-1)
    elif move == 1:
        pos1 = (x-1, y)
        pos2 = (x-2, y)
        pos3 = (x-2, y+1)
    elif move == -2:
        pos1 = (x,   y+1)
        pos2 = (x,   y+2)
        pos3 = (x-1, y+2)
    elif move == 2:
        pos1 = (x,   y+1)
        pos2 = (x,   y+2)
        pos3 = (x+1, y+2)
    elif move == -3:
        pos1 = (x+1, y)
        pos2 = (x+2, y)
        pos3 = (x+2, y+1)
    elif move == 3:
        pos1 = (x+1, y)
        pos2 = (x+2, y)
        pos3 = (x+2, y-1)
    elif move == -4:
        pos1 = (x,   y-1)
        pos2 = (x,   y-2)
        pos3 = (x+1, y-2)
    elif move == 4:
        pos1 = (x,   y-1)
        pos2 = (x,   y-2)
        pos3 = (x-1, y-2)
    elif move in {-5, 5, -6, 6, -7, 7, -8, 8}:
        raise ValueError('Extended moves are not yet implemented.')
    else:
        raise ValueError('Invalid move of "{}"'.format(move))
    # handle landing on a transport
    if transports is not None:
        assert len(transports) == 2, 'There must be exactly 0 or 2 transports.'
        if pos3 in transports:
            if pos3 == transports[0]:
                pos3 = transports[1]
            elif pos3 == transports[1]: #pragma: no branch
                pos3 = transports[0]
    # return the whole set of stuff
    return (pos1, pos2, pos3)

#%% _check_board_boundaries
def _check_board_boundaries(x, y, xmax, ymax):
    r"""
    Checks that a given position is on the board.

    Parameters
    ----------
    x : int
        Current X position
    y : int
        Current Y position
    xmax : int
        Current board X maximum
    ymax : int
        Current board Y maximum

    Returns
    -------
    is_valid : bool
        Whether position is valid (defined as on the board)

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  This current assumes a rectangular board, but is a separate function in case that ever gets
        more complicated.

    Examples
    --------

    >>> from dstauffman.games.knight import _check_board_boundaries
    >>> x = 2
    >>> y = 5
    >>> xmax = 7
    >>> ymax = 7
    >>> is_valid = _check_board_boundaries(x, y, xmax, ymax)
    >>> print(is_valid)
    True

    """
    # determine if X position is valid
    x_valid = 0 <= x <= xmax
    # determine if Y position is valid
    y_valid = 0 <= y <= ymax
    # combine results and return
    is_valid = x_valid and y_valid
    return is_valid

#%% _classify_move
def _classify_move(board, move, transports, start_x, start_y):
    r"""
    Determines if the desired move is valid or not, and what type of move/cost it would have.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    move : int
        Move to be performed
    transports : 2 element list of 1x2 tuples
        Location of the transports
    start_x : int
        Starting X location of the knight
    start_y : int
        Starting Y location of the knight

    Returns
    -------
    move_type : class Move
        Type of move to be performed

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _classify_move, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> move = 2 # (2 right and 1 down)
    >>> transports = None
    >>> start_x = 0
    >>> start_y = 2
    >>> board[start_x, start_y] = Piece.current
    >>> move_type = _classify_move(board, move, transports, start_x, start_y)
    >>> print(move_type)
    1

    """
    # find the size of the board
    xmax = board.shape[0] - 1
    ymax = board.shape[1] - 1
    # find the traversal for the desired move
    (pos1, pos2, pos3) = _get_new_position(start_x, start_y, move, transports)
    # check that the final and intermediate positions were all on the board
    valid_moves = np.array([_check_board_boundaries(pos[0], pos[1], xmax, ymax) for pos in (pos1, pos2, pos3)])
    if np.any(~valid_moves):
        return Move2['off_board']
    # get the values for each position
    p1 = board[pos1[0], pos1[1]]
    p2 = board[pos2[0], pos2[1]]
    p3 = board[pos3[0], pos3[1]]
    # check for error conditions
    if p3 in {Piece2['start'], Piece2['current']}:
        raise ValueError("The piece should never be able to move to it's current or starting position.") # pragma: no cover
    # check for blocked conditions
    if p3 in {Piece2['rock'], Piece2['barrier']} or p1 == Piece2['barrier'] or p2 == Piece2['barrier']:
        return Move2['blocked']
    # remaining moves are valid, determine type
    if p3 == Piece2['visited']:
        move_type = Move2['visited']
    elif p3 == Piece2['null']:
        move_type = Move2['normal']
    elif p3 == Piece2['final']:
        move_type = Move2['winning']
    elif p3 == Piece2['transport']:
        move_type = Move2['transport']
    elif p3 == Piece2['water']:
        move_type = Move2['water']
    elif p3 == Piece2['lava']:
        move_type = Move2['lava']
    else:
        raise ValueError('Unexpected piece type "{}"'.format(p3)) # pragma: no cover
    return move_type

#%% _update_board
def _update_board(board, move, costs, transports, start_x, start_y):
    r"""
    Updates the new board based on the desired move.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    move : int
        Move to be performed
    costs : 2D ndarray of int
        Costs for each square on the board layout
    transports : 2 element list of 1x2 tuples
        Location of the transports
    start_x : int
        Starting X location of the knight
    start_y : int
        Starting Y location of the knight

    Returns
    -------
    cost : int
        Cost of the specified move type
    is_repeat : bool
        Whether the last move was a repeated visit or not
    new_x : int
        New X location of the knight
    new_y : int
        New Y location of the knight

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015
    #.  Modified by David C. Stauffer in November 2015 to include new position information.
    #.  Modifies `board` in-place.

    Examples
    --------

    >>> from dstauffman.games.knight import _update_board, print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((2, 5), dtype=int)
    >>> move = 2 # (2 right and 1 down)
    >>> costs = np.ones(board.shape, dtype=int)
    >>> transports = None
    >>> start_x = 0
    >>> start_y = 2
    >>> board[start_x, start_y] = Piece.current
    >>> (cost, is_repeat, new_x, new_y) = _update_board(board, move, costs, transports, start_x, start_y)
    >>> print(cost)
    1

    >>> print(is_repeat)
    False

    >>> print_board(board)
    . . x . .
    . . . . K

    """
    # initialize outputs
    cost      = LARGE_INT
    is_repeat = False
    # determine the move type
    move_type = _classify_move(board, move, transports, start_x, start_y)
    # if valid, apply the move
    if move_type >= 0:
        # set the current position to visited
        board[start_x, start_y] = Piece.visited
        # get the new position
        (_, _, (new_x, new_y)) = _get_new_position(start_x, start_y, move, transports)
        # set the new position to current
        board[new_x, new_y] = Piece.current
        # determine what the cost was
        cost = costs[new_x, new_y]
        if move_type == Move2['winning']:
            cost = -cost
        elif move_type == Move2['visited']:
            is_repeat = True
    else:
        new_x = start_x
        new_y = start_y
    return (cost, is_repeat, new_x, new_y)

#%% _undo_move
def _undo_move(board, last_move, original_board, transports, start_x, start_y):
    r"""
    Undoes the last move on the board.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    last_move : int
        Last move that was previously performed
    original_board : 2D ndarray of int
        Original board layout before starting solver
    transports : 2 element list of 1x2 tuples
        Location of the transports
    start_x : int
        Starting X location of the knight
    start_y : int
        Starting Y location of the knight


    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Modifies `board` in-place.

    Examples
    --------
    >>> from dstauffman.games.knight import _undo_move, print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((2, 5), dtype=int)
    >>> board[0, 2] = Piece.visited
    >>> last_move = 2 # (2 right and 1 down)
    >>> original_board = np.zeros((2, 5), dtype=int)
    >>> original_board[0, 2] = Piece.start
    >>> transports = None
    >>> start_x = 1
    >>> start_y = 4
    >>> board[start_x, start_y] = Piece.current
    >>> print_board(board)
    . . x . .
    . . . . K

    >>> _undo_move(board, last_move, original_board, transports, start_x, start_y)
    >>> print_board(board)
    . . K . .
    . . . . .

    """
    # set the current position back to it's original piece
    if original_board[start_x, start_y] == Piece2['start']:
        board[start_x, start_y] = Piece2['null']
    else:
        board[start_x, start_y] = original_board[start_x, start_y]
    # find the inverse move
    new_move = _get_move_inverse(last_move)
    # if on a transport, then undo travel it first
    if transports is not None:
        assert len(transports) == 2, 'There must be exactly 0 or 2 transports.'
        if (start_x, start_y) in transports:
            if (start_x, start_y) == transports[0]:
                (start_x, start_y) = transports[1]
            elif (start_x, start_y) == transports[1]: # pragma: no branch
                (start_x, start_y) = transports[0]
    # get the new position (without traversing transports)
    (_, _, (new_x, new_y)) = _get_new_position(start_x, start_y, new_move, transports=None)
    # set the new position to current
    board[new_x, new_y] = Piece2['current']

#%% _get_move_inverse
def _get_move_inverse(move):
    r"""
    Gets the inverse move to go back where you were:
        -/+1 <-> -/+3
        -/+2 <-> -/+4

    Parameters
    ----------
    move : int
        Move to be performed

    Returns
    -------
    inv_move : int
        Move that will undo the last one

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------
    >>> from dstauffman.games.knight import _get_move_inverse
    >>> print(_get_move_inverse(-1))
    -3
    >>> print(_get_move_inverse(-3))
    -1
    >>> print(_get_move_inverse(2))
    4
    >>> print(_get_move_inverse(4))
    2

    """
    assert move in MOVES, 'Invalid move.'
    inv_move = np.sign(move) * (np.mod(np.abs(move) + 1, 4) + 1)
    return inv_move

#%% _predict_cost
def _predict_cost(board):
    r"""
    Predicts the cost from all locations on the board to the final square.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    costs : 2D ndarray of float
        Predicted cost to finish

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _predict_cost, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[0, 4] = Piece.final
    >>> costs = _predict_cost(board)
    >>> print(costs)
    [[ 2.   1.5  1.   0.5  0. ]
     [ 2.   1.5  1.   1.   0.5]]

    """
    # find the final position
    temp = np.nonzero(board == Piece.final)
    x_fin = temp[0][0]
    y_fin = temp[1][0]
    # build a grid of points to evaluate
    (X, Y) = np.meshgrid(np.arange(board.shape[0]), np.arange(board.shape[1]), indexing='ij')
    x_dist = np.abs(X - x_fin)
    y_dist = np.abs(Y - y_fin)
    costs = np.where(x_dist > y_dist, np.maximum(x_dist / 2, y_dist), np.maximum(x_dist, y_dist / 2))
    return costs

#%% _sort_best_moves
def _sort_best_moves(board, moves, costs, transports, start_x, start_y):
    r"""
    Sorts the given moves into the most likely best order based on a predicted cost

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    moves : list of int
        Possible moves to check
    costs : 2D ndarray of float
        Predicted cost to finish
    transports : 2 element list of 1x2 tuples
        Location of the transports
    start_x : int
        Starting X location of the knight
    start_y : int
        Starting Y location of the knight

    Returns
    -------
    sorted_moves : list of int
        Moves sorted by most likely

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _sort_best_moves, Piece, MOVES, _predict_cost
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.current
    >>> board[0, 4] = Piece.final
    >>> moves = MOVES
    >>> costs = _predict_cost(board)
    >>> transports = None
    >>> start_x = 0
    >>> start_y = 0
    >>> sorted_moves = _sort_best_moves(board, moves, costs, transports, start_x, start_y)
    >>> print(sorted_moves)
    [-1, -4, -2, 2, 4, 1]

    """
    # initialize the costs
    pred_costs = np.empty(len(moves))
    pred_costs.fill(np.nan)
    for (ix, move) in enumerate(moves):
        (_, _, (new_x, new_y)) = _get_new_position(start_x, start_y, move, transports)
        try:
            this_cost = costs[new_x, new_y]
            pred_costs[ix] = this_cost
        except IndexError:
            pass
    sorted_ix = pred_costs.argsort()
    sorted_moves = [moves[i] for i in sorted_ix if not np.isnan(pred_costs[i])]
    return sorted_moves

#%% print_baord
def print_board(board):
    r"""
    Prints the current board position to the console window.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((5,5), dtype=int)
    >>> board[2,2] = Piece.current
    >>> print_board(board)
    . . . . .
    . . . . .
    . . K . .
    . . . . .
    . . . . .

    """
    # get the board shape
    (rows, cols) = board.shape
    # loop through pieces
    for i in range(rows):
        for j in range(cols):
            # print each piece in the row without a line continuation
            pad = ' ' if j < cols-1 else ''
            print(NUM_DICT[board[i, j]] + pad, end='')
        # add the line continuation at the end of each row
        print('')

#%% char_board_to_nums
def char_board_to_nums(char_board):
    r"""
    Converts the original board from a character array into a numpy ndarray of int.

    Parameters
    ----------
    char_board : str
        Board as an original character array, where each line is a row of the board

    Returns
    -------
    board : 2D ndarray of int
        Board layout

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import char_board_to_nums
    >>> char_board = '. . S . .\n. . . . E'
    >>> board = char_board_to_nums(char_board)
    >>> print(board)
    [[0 0 1 0 0]
     [0 0 0 0 2]]

    """
    # convert to rows of lines
    lines = char_board.split('\n')
    # remove any empty rows
    lines = [this_line.split(' ') for this_line in lines if this_line]
    # get the size of the board
    rows = len(lines)
    cols = len(lines[0])
    # preallocate null board
    board = np.zeros((rows, cols), dtype=int)
    # loop through and store all pieces and integer equivalents
    for i in range(rows):
        for j in range(cols):
            board[i, j] = CHAR_DICT[lines[i][j]]
    return board

#%% check_valid_sequence
def check_valid_sequence(board, moves, print_status=False, allow_repeats=False):
    r"""
    Checks that the list of moves is a valid sequence to go from start to final position.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    moves : ndarray of int
        Moves to be performed
    print_status : bool, optional
        Whether to print the board after each move is made, default is False
    allow_repeats : bool, optional
        Whether to allow repeat visits to the same square

    Returns
    -------
    is_valid : bool
        Whether the sequence is valid or not.

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import check_valid_sequence, Piece, _board_to_costs
    >>> import numpy as np
    >>> board = np.zeros((3, 5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[2, 4] = Piece.final
    >>> moves = [2, 2]
    >>> is_valid = check_valid_sequence(board, moves)
    >>> print(is_valid)
    True

    """
    # initialize output
    is_valid = True
    is_done  = False
    # determine the costs
    costs = _board_to_costs(board)
    # find transports
    transports = _get_transports(board)
    # create internal board for calculations
    temp_board = board.copy()
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    (x, y) = _get_current_position(temp_board)
    # check that the board has a final goal
    if not np.any(temp_board == Piece.final):
        raise ValueError('The board does not have a finishing location.')
    for (i, this_move) in enumerate(moves):
        # update the board and determine the cost
        (cost, is_repeat, x, y) = _update_board(temp_board, this_move, costs, transports, x, y)
        # if cost was zero, then the move was invalid
        if cost == COST_DICT['invalid']:
            is_valid = False
            break
        # check for repeated conditions
        if not allow_repeats and is_repeat:
            is_valid = False
            if print_status:
                print('No repeats allowed.')
            break
        # check for winning conditions (based on negative cost value)
        if cost < 0:
            is_done = True
            if i < len(moves)-1:
                raise ValueError('Sequence finished, but then kept going.')
    if print_status:
        if is_valid:
            if is_done:
                print('Sequence is valid and finished the puzzle.')
            else:
                print('Sequence is valid, but did not finish the puzzle.')
        else:
            print('Sequence is not valid.')
    return is_valid

#%% print_sequence
def print_sequence(board, moves):
    r"""
    Prints the every board position for the given move sequence.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    moves : ndarray of int
        Moves to be performed

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import print_sequence, Piece
    >>> import numpy as np
    >>> board = np.zeros((3, 5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[2, 4] = Piece.final
    >>> moves = [2, 2]
    >>> print_sequence(board, moves)
    Starting position:
    S . . . .
    . . . . .
    . . . . E
    <BLANKLINE>
    After move 1, cost: 1
    x . . . .
    . . K . .
    . . . . E
    <BLANKLINE>
    After move 2, cost: 2
    x . . . .
    . . x . .
    . . . . K

    """
    # determine costs
    costs = _board_to_costs(board)
    # find transports
    transports = _get_transports(board)
    # create internal board for calculations
    temp_board = board.copy()
    print('Starting position:')
    print_board(temp_board)
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    (x, y) = _get_current_position(temp_board)
    # initialize total costs
    total_cost = 0
    # loop through move sequence
    for (i, this_move) in enumerate(moves):
        # update board
        (cost, _, x, y) = _update_board(temp_board, this_move, costs, transports, x, y)
        if cost != COST_DICT['invalid']:
            # update total costs
            total_cost += abs(cost)
            # print header
            print('\nAfter move {}, cost: {}'.format(i+1, total_cost))
            # print new board
            print_board(temp_board)
        else:
            raise ValueError('Bad sequence.')

#%% _initialize_data
def _initialize_data(board):
    r"""
    Initializers the internal data structure for use in the solver.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    data : dict
        Data dictionary for use in the solver.  Contains the following keys:
            all_boards
            all_moves
            best_costs
            best_moves
            costs
            current_cost
            final_loc
            is_solved
            moves
            original_board
            pred_costs

    Notes
    -----
    #.  Written by David C. Stauffer in November 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _initialize_data, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[0, 4] = Piece.final
    >>> data = _initialize_data(board)
    >>> print(sorted(data.keys())[:6])
    ['all_boards', 'all_moves', 'best_costs', 'best_moves', 'costs', 'current_cost']

    >>> print(sorted(data.keys())[6:])
    ['final_loc', 'is_solved', 'moves', 'original_board', 'pred_costs', 'transports']

    """
    # initialize dictionary
    data = {}
    # save original board for use in undoing moves
    data['original_board'] = board.copy()
    # find transports
    data['transports'] = _get_transports(board)
    # alias the final location for use at the end
    temp = np.nonzero(board == Piece.final)
    data['final_loc'] = (temp[0][0], temp[1][0])
    # calculate the costs for landing on each square
    data['costs'] = _board_to_costs(board)
    # crudely predict all the costs
    data['pred_costs'] = _predict_cost(board)
    # initialize best costs on first run
    data['best_costs'] = LARGE_INT * np.ones(board.shape, dtype=int)
    # initialize best solution
    data['best_moves'] = None
    # initialize moves array and solved status
    data['moves'] = []
    data['is_solved'] = False
    data['all_moves'] = [[[] for y in range(board.shape[1])] for x in range(board.shape[0])]
    data['all_boards'] = np.empty((board.shape[0], board.shape[1], board.shape[0], board.shape[1]), dtype=int)
    # initialize current cost and update in best_costs
    data['current_cost'] = 0
    temp = np.nonzero(board == Piece.start)
    data['best_costs'][temp] = data['current_cost']
    # create temp board and set the current position to the start
    temp_board = board.copy()
    temp_board[board == Piece.start] = Piece.current
    # store the first board
    data['all_boards'][:, :, temp[0][0], temp[1][0]] = temp_board.copy()
    return data

#%% _solve_next_move
def _solve_next_move(board, data, start_x, start_y):
    r"""
    Solves the puzzle using a breadth first approach.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    data : dict
        Mutable internal data dictionary for storing information throughout solver calls, see _initialize_data
    cur_pos : 2 element tuple of int
        Current x and y position

    Notes
    -----
    #.  Written by David C. Stauffer in November 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import _solve_next_move, Piece, _initialize_data
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[0, 4] = Piece.final
    >>> data = _initialize_data(board)
    >>> start_x = 0
    >>> start_y = 0
    >>> board[start_x, start_y] = Piece.current
    >>> _solve_next_move(board, data, start_x, start_y)
    >>> print(data['best_costs'])
    [[      0 1000000 1000000 1000000 1000000]
     [1000000 1000000       1 1000000 1000000]]

    """
    # check for a start piece, in which case something is messed up
    assert not np.any(board == Piece2['start']), 'Empty dicts should not have a start piece and vice versa.'
    # guess the order for the best moves based on predicited costs
    sorted_moves = _sort_best_moves(board, MOVES, data['pred_costs'], data['transports'], start_x, start_y)
    # try all the next possible moves
    for this_move in sorted_moves:
        # make the move
        (cost, is_repeat, new_x, new_y) = _update_board(board, this_move, data['costs'], \
            data['transports'], start_x, start_y)
        # optional logging for debugging
        if LOGGING: # pragma: no cover
            print('this_move = {}, this_cost = {}, total moves = {}'.format(this_move, cost, data['moves']), end='')
        # if the move was invalid then go to the next one
        if cost == COST_DICT['invalid']:
            if LOGGING: # pragma: no cover
                print(' - invalid')
            continue # pragma: no cover - Actually covered, error in coverage tool
        # valid move
        else:
            # determine if move was to a previously visited square of worse cost than another sequence
            if is_repeat or data['current_cost'] + abs(cost) >= data['best_costs'][new_x, new_y]:
                    if LOGGING: # pragma: no cover
                        print(' - worse repeat')
                    # reject move and re-establish the visited state
                    _undo_move(board, this_move, data['original_board'], data['transports'], new_x, new_y)
                    if cost > 0 and is_repeat:
                        board[new_x, new_y] = Piece2['visited']
                    continue # pragma: no cover - Actually covered, error in coverage tool
            # optional logging for debugging
            if LOGGING: # pragma: no cover
                if cost < 0:
                    print(' - solution')
                    print('Potential solution found, moves = {} + {}'.format(data['moves'], this_move))
                elif is_repeat:
                    print(' - better repeat')
                else:
                    print(' - new step')
            # move is new or better, update current and best costs and append move
            assert data['best_costs'][new_x, new_y] > 50000
            data['best_costs'][new_x, new_y] = data['current_cost'] + cost
            data['all_moves'][new_x][new_y] = data['moves'][:] + [this_move]
            data['all_boards'][:, :, new_x, new_y] = board.copy()
            if cost < 0:
                print('Solution found for cost of: {}.'.format(data['current_cost'] + abs(cost)))
                data['is_solved'] = True
                _undo_move(board, this_move, data['original_board'], data['transports'], new_x, new_y)
            else:
                # undo board as prep for next move
                _undo_move(board, this_move, data['original_board'], data['transports'], new_x, new_y)

#%% Functions - solve_min_puzzle
def solve_min_puzzle(board):
    r"""
    Puzzle solver.  Uses a breadth first approach to solve for the minimum length solution.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    moves : list of int
        Moves to solve the puzzle, empty if no solution was found

    Notes
    -----
    #.  Written by David C. Stauffer in November 2015.
    #.  The 'max' length solver has not been written yet.

    Examples
    --------

    >>> from dstauffman.games.knight import solve_min_puzzle, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[0, 4] = Piece.final
    >>> moves = solve_min_puzzle(board) # doctest: +ELLIPSIS
    Initializing solver.
    Solution found for cost of: 2.
    Elapsed time : ...

    >>> print(moves)
    [2, -2]

    """
    # hard-coded values
    MAX_ITERS = 25

    # start timer
    start_solver = time.time()

    # initialize the data structure for the solver
    print('Initializing solver.')

    # check that the board has a final goal
    if not np.any(board == Piece.final):
        raise ValueError('The board does not have a finishing location.')

    # initialize the data structure
    data = _initialize_data(board)

    # solve the puzzle
    for this_iter in range(MAX_ITERS):
        next_moves = np.nonzero(data['best_costs'] == data['current_cost'])
        for ix in range(len(next_moves[0])):
            start_x = next_moves[0][ix]
            start_y = next_moves[1][ix]
            # update the current board and moves for the given position
            temp_board = data['all_boards'][:, :, start_x, start_y]
            data['moves'] = data['all_moves'][start_x][start_y]
            # call solver for this move
            _solve_next_move(temp_board, data, start_x, start_y)
        # increment the minimum cost and continue
        data['current_cost'] += 1
    # if the puzzle was solved, then save the relevant move list
    if data['is_solved']:
        data['moves'] = data['all_moves'][data['final_loc'][0]][data['final_loc'][1]]
    else:
        print('No solution found.')
        data['moves'] = []
    # display the elapsed time
    print('Elapsed time : ' + time.strftime('%H:%M:%S', time.gmtime(time.time()-start_solver)))
    return data['moves'] # or just return data for debugging

#%% Functions - solve_max_puzzle
def solve_max_puzzle(board):
    r"""
    Puzzle solver.  Uses a depth first recursive approach to solve for the maximum length solution.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout

    Returns
    -------
    moves : list of int
        Moves to solve the puzzle, empty if no solution was found

    Notes
    -----
    #.  Not yet written.

    Examples
    --------

    >>> from dstauffman.games.knight import solve_max_puzzle, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[0, 4] = Piece.final
    >>> moves = solve_max_puzzle(board) # doctest: +SKIP

    >>> print(moves) # doctest: +SKIP

    """
    # start timer
    start_solver = time.time()

    # initialize the data structure for the solver
    print('Initializing solver.')

    # TODO: write this function
    moves = []
    if len(moves) == 0:
        print('No solution found.')

    # display the elapsed time
    print('Elapsed time : ' + time.strftime('%H:%M:%S', time.gmtime(time.time()-start_solver)))
    return moves # or just return data for debugging

#%% Unit test
def _main():
    r"""Unit tests."""
    unittest.main(module='dstauffman.games.test_knight', exit=False)
    doctest.testmod(verbose=False)

#%% Script
if __name__ == '__main__':
    # flags for what steps to do
    do_steps = {-1, 0, 1, 2, 3, 4, 5}
    #do_steps = {10}
    print_seq = True

    # Step -1 (Run unit tests)
    if -1 in do_steps:
        _main()

    # convert board to numeric representation for efficiency
    board1 = char_board_to_nums(BOARD1)
    board2 = char_board_to_nums(BOARD2)

    # Step 0
    if 0 in do_steps:
        print('Step 0: print the boards.')
        print_board(board1)
        print('')
        print_board(board2)
        print('')

    # Step 1
    if 1 in do_steps:
        print('\nStep 1: see if the sequence is valid.')
        moves1 = [2, 2]
        is_valid1 = check_valid_sequence(board1, moves1, print_status=True)
        if is_valid1 and print_seq:
            print_sequence(board1, moves1)

    # Step 2
    if 2 in do_steps:
        print('\nStep 2: solve the first board for any length solution.')
        moves2 = solve_min_puzzle(board1)
        print(moves2)
        is_valid2 = check_valid_sequence(board1, moves2, print_status=True)
        if is_valid2 and print_seq:
            print_sequence(board1, moves2)

    # Step 3
    if 3 in do_steps:
        print('\nStep 3: solve the first board for the minimum length solution.')
        moves3 = solve_min_puzzle(board1)
        print(moves3)
        is_valid3 = check_valid_sequence(board1, moves3, print_status=True)
        if is_valid3 and print_seq:
            print_sequence(board1, moves3)

    # Step 4
    if 4 in do_steps:
        print('\nStep 4: solve the second board for the minimum length solution.')
        board2[0, 0] = Piece.start
        #board2[-1, -1] = Piece.final # doesn't use transport
        board2[11, -1] = Piece.final # uses transport
        moves4 = solve_min_puzzle(board2)
        print(moves4)
        is_valid4 = check_valid_sequence(board2, moves4, print_status=True)
        if is_valid4 and print_seq:
            print_sequence(board2, moves4)

    # Step 5
    if 5 in do_steps:
        pass

    # Step 10, alternate solver for testing
    if 10 in do_steps:
        print('\nStep 10: solve a test board for the minimum length solution.')
        board3 = np.zeros((6, 5), dtype=int)
        board3[:, 2] = Piece.barrier
        board3[0, 0] = Piece.start
        board3[5, 3] = Piece.final
        board3[4, 0] = Piece.transport
        board3[1, 3] = Piece.transport
        print_board(board3)
        print('')
        moves10 = solve_min_puzzle(board3)
        is_valid10 = check_valid_sequence(board3, moves10, print_status=True)
        print(moves10)
        if is_valid10:
            print_sequence(board3, moves10)
