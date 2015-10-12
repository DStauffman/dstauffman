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
from enum import unique
import numpy as np
import unittest
# personal dstauffman libary imports
from dstauffman import IntEnumPlus

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
class Piece(IntEnumPlus):
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
    transport = 7 # transports to the other transport, can only be exactly 0 or 2 on the board
    lava      = 8 # lava, costs 5 moves to land on
    visited   = 9 # previously visited square that cannot be used again

#%% Classes - Move
@unique
class Move(IntEnumPlus):
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

#%% board_to_costs
def board_to_costs(board):
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

    >>> from dstauffman.games.knight import board_to_costs, Piece
    >>> import numpy as np
    >>> board = Piece.null * np.ones((3, 3), dtype=int)
    >>> board[0, 0] = Piece.water
    >>> board[1, 1] = Piece.start
    >>> board[2, 2] = Piece.barrier
    >>> costs = board_to_costs(board)
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

#%% get_current_position
def get_current_position(board):
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

    >>> from dstauffman.games.knight import get_current_position, Piece
    >>> import numpy as np
    >>> board = np.zeros((5,5), dtype=int)
    >>> board[1,4] = Piece.current
    >>> (x, y) = get_current_position(board)
    >>> print(x, y)
    1 4

    """
    # find position
    ix = np.nonzero(board == Piece.current)
    # alias X and Y locations
    x = ix[0]
    y = ix[1]
    # check that only exactly one current position was found, and if not, print the messed up board
    is_valid = len(x) == 1 and len(y) == 1
    if not is_valid:
        print_board(board)
    assert is_valid, 'Only exactly one current position may be found.'
    # otherwise if all is good, return the location
    return (x[0], y[0])

#%% get_new_position
def get_new_position(x, y, move):
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

    >>> from dstauffman.games.knight import get_new_position
    >>> x = 2
    >>> y = 3
    >>> move = 2 # (2 right and 1 down)
    >>> (pos1, pos2, pos3) = get_new_position(x, y, move)
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
    # return the whole set of stuff
    return (pos1, pos2, pos3)

#%% check_board_boundaries
def check_board_boundaries(x, y, xmax, ymax):
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

    >>> from dstauffman.games.knight import check_board_boundaries
    >>> x = 2
    >>> y = 5
    >>> xmax = 7
    >>> ymax = 7
    >>> is_valid = check_board_boundaries(x, y, xmax, ymax)
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

#%% classify_move
def classify_move(board, move):
    r"""
    Determines if the desired move is valid or not, and what type of move/cost it would have.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    move : int
        Move to be performed

    Returns
    -------
    move_type : class Move
        Type of move to be performed

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import classify_move, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 2] = Piece.current
    >>> move = 2 # (2 right and 1 down)
    >>> move_type = classify_move(board, move)
    >>> print(move_type)
    Move.normal: 1

    """
    # find the size of the board
    xmax = board.shape[0] - 1
    ymax = board.shape[1] - 1
    # find current position
    (x, y) = get_current_position(board)
    # find the traversal for the desired move
    (pos1, pos2, pos3) = get_new_position(x, y, move)
    # check that the final and intermediate positions were all on the board
    valid_moves = np.array([check_board_boundaries(pos[0], pos[1], xmax, ymax) for pos in (pos1, pos2, pos3)])
    if np.any(~valid_moves):
        return Move.off_board
    # get the values for each position
    p1 = board[pos1[0], pos1[1]]
    p2 = board[pos2[0], pos2[1]]
    p3 = board[pos3[0], pos3[1]]
    # check for error conditions
    if p3 in {Piece.start, Piece.current}:
        raise ValueError("The piece should never be able to move to it's current or starting position.")
    # check for blocked conditions
    if p3 in {Piece.rock, Piece.barrier} or p1 == Piece.barrier or p2 == Piece.barrier:
        return Move.blocked
    # remaining moves are valid, determine type
    if p3 == Piece.visited:
        move_type = Move.visited
    elif p3 == Piece.null:
        move_type = Move.normal
    elif p3 == Piece.final:
        move_type = Move.winning
    elif p3 == Piece.transport:
        move_type = Move.transport
    elif p3 == Piece.water:
        move_type = Move.water
    elif p3 == Piece.lava:
        move_type = Move.lava
    else:
        raise ValueError('Unexpected piece type "{}"'.format(p3))
    return move_type

#%% update_board
def update_board(board, move, costs=None):
    r"""
    Updates the new board based on the desired move.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    move : int
        Move to be performed
    costs : 2D ndarray of int, optional
        Costs for each square on the board layout

    Returns
    -------
    cost : int
        Cost of the specified move type
    is_repeat : bool
        Whether the last move was a repeated visit or not

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015
    #.  Modifies `board` in-place.

    Examples
    --------

    >>> from dstauffman.games.knight import update_board, print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((2, 5), dtype=int)
    >>> board[0, 2] = Piece.current
    >>> move = 2 # (2 right and 1 down)
    >>> (cost, is_repeat) = update_board(board, move)
    >>> print(cost)
    1

    >>> print(is_repeat)
    False

    >>> print_board(board)
    . . x . .
    . . . . K

    """
    # check for optional inputs
    if costs is None:
        costs = np.ones(board.shape, dtype=int)
    # initialize outputs
    cost      = LARGE_INT
    is_repeat = False
    # find current position
    (x, y) = get_current_position(board)
    # determine the move type
    move_type = classify_move(board, move)
    # if valid, apply the move
    if move_type >= 0:
        # set the current position to visited
        board[x, y] = Piece.visited
        # get the new position
        (_, _, (new_x, new_y)) = get_new_position(x, y, move)
        # set the new position to current
        board[new_x, new_y] = Piece.current
        # determine what the cost was
        cost = costs[new_x, new_y]
        if move_type == Move.winning:
            cost = -cost
        if move_type == 0:
            is_repeat = True
    return (cost, is_repeat)

#%% undo_move
def undo_move(board, last_move, original_board):
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

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Modifies `board` in-place.

    Examples
    --------
    >>> from dstauffman.games.knight import undo_move, print_board, Piece
    >>> import numpy as np
    >>> board = np.zeros((2, 5), dtype=int)
    >>> board[0, 2] = Piece.visited
    >>> board[1, 4] = Piece.current
    >>> last_move = 2 # (2 right and 1 down)
    >>> original_board = np.zeros((2, 5), dtype=int)
    >>> original_board[0, 2] = Piece.start
    >>> print_board(board)
    . . x . .
    . . . . K

    >>> undo_move(board, last_move, original_board)
    >>> print_board(board)
    . . K . .
    . . . . .

    """
    # find the current position
    (x, y) = get_current_position(board)
    # set the current position back to a null piece
    board[x, y] = original_board[x, y]
    # find the inverse move
    new_move = get_move_inverse(last_move)
    # get the new position
    (_, _, (new_x, new_y)) = get_new_position(x, y, new_move)
    # set the new position to current
    board[new_x, new_y] = Piece.current

#%% get_move_inverse
def get_move_inverse(move):
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
    >>> from dstauffman.games.knight import get_move_inverse
    >>> print(get_move_inverse(-1))
    -3
    >>> print(get_move_inverse(-3))
    -1
    >>> print(get_move_inverse(2))
    4
    >>> print(get_move_inverse(4))
    2

    """
    assert move in MOVES, 'Invalid move.'
    inv_move = np.sign(move) * (np.mod(np.abs(move) + 1, 4) + 1)
    return inv_move

#%% check_valid_sequence
def check_valid_sequence(board, moves, costs, print_status=False, allow_repeats=False):
    r"""
    Checks that the list of moves is a valid sequence to go from start to final position.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    moves : ndarray of int
        Moves to be performed
    costs : 2D ndarray of int
        Costs for each square on the board layout
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

    >>> from dstauffman.games.knight import check_valid_sequence, Piece, board_to_costs
    >>> import numpy as np
    >>> board = np.zeros((3, 5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[2, 4] = Piece.final
    >>> moves = [2, 2]
    >>> costs = board_to_costs(board)
    >>> is_valid = check_valid_sequence(board, moves, costs)
    >>> print(is_valid)
    True

    """
    # initialize output
    is_valid = True
    is_done  = False
    # create internal board for calculations
    temp_board = board.copy()
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    # check that the board has a final goal
    if not np.any(temp_board == Piece.final):
        raise ValueError('The board does not have a finishing location.')
    for (i, this_move) in enumerate(moves):
        # update the board and determine the cost
        (cost, is_repeat) = update_board(temp_board, this_move, costs)
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
def print_sequence(board, moves, costs=None):
    r"""
    Prints the every board position for the given move sequence.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    moves : ndarray of int
        Moves to be performed
    costs : 2D ndarray of int, optional
        Costs for each square on the board layout

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
    After move 1
    x . . . .
    . . K . .
    . . . . E
    <BLANKLINE>
    After move 2
    x . . . .
    . . x . .
    . . . . K

    """
    # check for optional inputs
    if costs is None:
        costs = np.ones(board.shape, dtype=int)
    # create internal board for calculations
    temp_board = board.copy()
    print('Starting position:')
    print_board(temp_board)
    # set the current position to the start
    temp_board[temp_board == Piece.start] = Piece.current
    # loop through move sequence
    for (i, this_move) in enumerate(moves):
        # print header
        print('\nAfter move {}'.format(i+1))
        # update board
        (cost, _) = update_board(temp_board, this_move, costs)
        # print new board
        if cost != COST_DICT['invalid']:
            print_board(temp_board)
        else:
            raise ValueError('Bad sequence.')

#%% predict_cost
def predict_cost(board):
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

    >>> from dstauffman.games.knight import predict_cost, Piece
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[0, 4] = Piece.final
    >>> costs = predict_cost(board)
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

#%% sort_best_moves
def sort_best_moves(board, moves, costs):
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

    Returns
    -------
    sorted_moves : list of int
        Moves sorted by most likely

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

    Examples
    --------

    >>> from dstauffman.games.knight import sort_best_moves, Piece, MOVES, predict_cost
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.current
    >>> board[0, 4] = Piece.final
    >>> moves = MOVES
    >>> costs = predict_cost(board)
    >>> sorted_moves = sort_best_moves(board, moves, costs)
    >>> print(sorted_moves)
    [-1, -4, -2, 2, 4, 1]

    """
    # get the current position
    (x, y) = get_current_position(board)
    # initialize the costs
    pred_costs = np.empty(len(moves))
    pred_costs.fill(np.nan)
    for (ix, move) in enumerate(moves):
        (_, _, new_pos) = get_new_position(x, y, move)
        try:
            this_cost = costs[new_pos[0], new_pos[1]]
            pred_costs[ix] = this_cost
        except IndexError:
            pass
    sorted_ix = pred_costs.argsort()
    sorted_moves = [moves[i] for i in sorted_ix if not np.isnan(pred_costs[i])]
    return sorted_moves

#%% solve_puzzle
def solve_puzzle(board, costs, solve_type='min', data={}):
    r"""
    Solves the puzzle with the desired solution type, from 'min', 'max', 'first'.

    Parameters
    ----------
    board : 2D ndarray of int
        Board layout
    costs : 2D ndarray of int, optional
        Costs for each square on the board layout
    solve_type : str
        Solver type, from {'min', 'max', 'first'}
    data : dict
        Mutable internal data dictionary for storing information through recursive calls

    moves : ndarray of int
        Moves to be performed
    original_board : 2D ndarray of int, optional, not intended by user, only recursively
        Original board layout before doing any moves
    current_cost : int, optional, not intended by user, only recursively
        Cumulative current cost for the solution

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Modifies `data` inplace and should always be called by the user with an empty dictionary.
        Updates keys: 'moves', TBD...

    Examples
    --------

    >>> from dstauffman.games.knight import solve_puzzle, Piece, board_to_costs
    >>> import numpy as np
    >>> board = np.zeros((2,5), dtype=int)
    >>> board[0, 0] = Piece.start
    >>> board[0, 4] = Piece.final
    >>> costs = board_to_costs(board)
    >>> solve_type = 'first'
    >>> data = {}
    >>> solve_puzzle(board, costs, solve_type, data)
    Initializing solver.
    Solution found!

    >>> print(data['moves'])
    [2, -2]

    """
    if len(data) == 0:
        # print message
        print('Initializing solver.')
        # check that the board has a final goal
        if not np.any(board == Piece.final):
            raise ValueError('The board does not have a finishing location.')
        # save original board for use in undoing moves
        data['original_board'] = board.copy()
        # alias the final location for use later
        temp = np.nonzero(board == Piece.final)
        data['final_loc_x'] = temp[0][0]
        data['final_loc_y'] = temp[1][0]
        # crudely predict all the costs
        data['pred_costs'] = predict_cost(board)
        # initialize best costs on first run
        data['best_costs'] = LARGE_INT * np.ones(board.shape, dtype=int)
        # initialize best solution
        data['best_moves'] = None
        # initialize moves array and solved status
        data['moves'] = []
        data['is_solved'] = False
        # initialize current cost and update in best_costs
        data['current_cost'] = 0
        data['best_costs'][board == Piece.start] = data['current_cost']
        # set the current position to the start
        board[board == Piece.start] = Piece.current
    else:
        # check for a start piece, in which case something is messed up
        assert not np.any(board == Piece.start), 'Empty dicts should not have a start piece and vice versa.'
    # guess the order for the best moves based on predicited costs
    sorted_moves = sort_best_moves(board, MOVES, data['pred_costs'])
    # try all the next possible moves
    for this_move in sorted_moves:
        # optimization for longer moves that we know won't be better
        if solve_type == 'min' and data['current_cost'] >= data['best_costs'][data['final_loc_x'],data['final_loc_y']]:
            break
        # make the move
        (cost, is_repeat) = update_board(board, this_move, costs)
        if LOGGING:
            print('this_move = {}, this_cost = {}, total moves = {}'.format(this_move, cost, data['moves']), end='')
        # if the move was invalid then go to the next one, if legal check if done, otherwise call recursively
        if cost == COST_DICT['invalid']:
            if LOGGING:
                print(' - invalid')
            continue
        # winning move
        elif cost < 0:
            if LOGGING:
                print(' - solution')
            # get the new current location
            temp_loc = board == Piece.current
            if LOGGING:
                print('Potential solution found, moves = {} + {}'.format(data['moves'], this_move))
            # solution is found
            if solve_type == 'first' or data['current_cost'] + abs(cost) < data['best_costs'][temp_loc]:
                if solve_type == 'first':
                    print('Solution found!')
                else:
                    print('Solution found for cost of: {}.'.format(data['current_cost'] + abs(cost)))
                data['current_cost'] = data['current_cost'] + abs(cost)
                data['best_costs'][temp_loc] = data['current_cost']
                data['moves'].append(this_move)
                data['is_solved'] = True
                data['best_moves'] = data['moves'].copy()
                return
            else:
                # reject move and re-establish the visited state
                undo_move(board, this_move, data['original_board'])
                board[temp_loc] = Piece.final
                continue
        # valid, but not winning move
        else:
            # get the new current location
            temp_loc = board == Piece.current
            # determine if move was to a previously visited square
            if is_repeat:
                # determine if a better sequence
                if solve_type == 'first' or data['current_cost'] + cost >= data['best_costs'][temp_loc]:
                    if LOGGING:
                        print(' - worse repeat')
                    # reject move and re-establish the visited state
                    undo_move(board, this_move, data['original_board'])
                    board[temp_loc] = Piece.visited
                    continue
            if LOGGING:
                if is_repeat:
                    print(' - better repeat')
                else:
                    print(' - new step')
            # move is new or better, update current and best costs and append move
            data['current_cost'] = data['current_cost'] + cost
            data['best_costs'][temp_loc] = data['current_cost']
            data['moves'].append(this_move)
            # make next move recursively
            solve_puzzle(board, costs, solve_type, data)
            # if this didn't solve the puzzle, then the branch is dead, so continue to next move
            if not data['is_solved']:
                continue
            else:
                # if solver is the 'first' type, then exit now once you have a solution
                if solve_type == 'first':
                    return
                # if looking for more solutions, then back out (2 moves) and let algorithm continue
                else:
                    for i in range(2):
                        last_move = data['moves'].pop()
                        data['current_cost'] = data['current_cost'] - costs[board == Piece.current][0]
                        undo_move(board, last_move, data['original_board'])
                    data['is_solved'] = False
                    continue # (not return, need to go through other moves at this level)
    # all possible moves didn't work, so this leg is dead, back out last move and exit
    if len(data['moves']) > 0:
        if LOGGING:
            print('moves = {} is dead, backing out last_move = {}'.format(data['moves'], data['moves'][-1]))
        last_move = data['moves'].pop()
        data['current_cost'] = data['current_cost'] - costs[board == Piece.current][0]
        undo_move(board, last_move, data['original_board'])
    else:
        # check if any branches solved the puzzle
        if data['best_moves'] is not None:
            data['is_solved'] = True
            data['moves'] = data['best_moves'].copy()
        else:
            print('No solution found.')
    return

#%% Unit test
def main():
    r"""Unit tests."""
    unittest.main(module='dstauffman.games.test_knight', exit=False)
    doctest.testmod(verbose=False)

#%% Script
if __name__ == '__main__':
    # flags for what steps to do
    do_steps = {-1, 0, 1, 2, 3, 4, 5}
    do_steps = {4}
    do_steps = {-1, 0, 1, 2, 3}
    print_seq = False

    # Step -1 (Run unit tests)
    if -1 in do_steps:
        main()

    # convert board to numeric representation for efficiency
    board1 = char_board_to_nums(BOARD1)
    board2 = char_board_to_nums(BOARD2)
    #board3 = np.zeros((5, 5), dtype=int)
    #board3[2, 2] = Piece.start
    #board3[3, 4] = Piece.final
    board3 = np.zeros((3, 5), dtype=int)
    board3[0, 0] = Piece.start
    board3[0, 4] = Piece.final
    #board3 = np.zeros((3, 3), dtype=int)
    #board3[0, 2] = Piece.start
    #board3[0, 0] = Piece.final
    costs1 = board_to_costs(board1)
    costs2 = board_to_costs(board2)
    costs3 = board_to_costs(board3)

    # Step 0
    if 0 in do_steps:
        print('Step 0: print the boards.')
        print_board(board1)
        print('')
        print_board(board2)
        print('')
        #print_board(board3)
        #print('')

    # Step 1
    if 1 in do_steps:
        print('\nStep 1: see if the sequence is valid.')
        moves1 = [2, 2]
        is_valid1 = check_valid_sequence(board1, moves1, costs1, print_status=True)
        if is_valid1 and print_seq:
            print_sequence(board1, moves1)

    # Step 2
    if 2 in do_steps:
        print('\nStep 2: solve the first board for any length solution.')
        soln_board = board1.copy()
        data2 = {}
        solve_puzzle(soln_board, costs1, solve_type='first', data=data2)
        print(data2['moves'])
        is_valid2 = check_valid_sequence(board1, data2['moves'], costs1, print_status=True)
        if is_valid2 and print_seq:
            print_sequence(board1, data2['moves'])

    # Step 3
    if 3 in do_steps:
        print('\nStep 3: solve the first board for the minimum length solution.')
        soln_board = board1.copy()
        data3 = {}
        solve_puzzle(soln_board, costs1, solve_type='min', data=data3)
        print(data3['moves'])
        is_valid3 = check_valid_sequence(board1, data3['moves'], costs1, print_status=True)
        if is_valid3:
            print_sequence(board1, data3['moves'])

    # Step 4
    if 4 in do_steps:
        print('\nStep 4: solve the second board for the minimum length solution.')
        board2[0, 0] = Piece.start
        board2[-1, -1] = Piece.final
        costs2 = board_to_costs(board2)
        soln_board = board2.copy()
        data4 = {}
        solve_puzzle(soln_board, costs2, solve_type='min', data=data4)
        print(data4['moves'])
        is_valid4 = check_valid_sequence(board2, data4['moves'], costs2, print_status=True)
        if is_valid4:
            print_sequence(board2, data4['moves'])

    # Step 5
    if 5 in do_steps:
        pass
