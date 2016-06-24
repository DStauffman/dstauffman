# -*- coding: utf-8 -*-
"""
The "knight.pyx" file contains Cython versions of their Python counterparts, enabling automatic
compilation to C, and hopefully faster runtimes.

Notes
-----
#.  Written by David C. Stauffer in June 2016.
"""

#%% Imports
cimport cython
from libcpp cimport bool

#%% check_board_boundaries
def check_board_boundaries(int x, int y, int xmax, int ymax):
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

    >>> from dstauffman.games.knight2 import check_board_boundaries
    >>> x = 2
    >>> y = 5
    >>> xmax = 7
    >>> ymax = 7
    >>> is_valid = check_board_boundaries(x, y, xmax, ymax)
    >>> print(is_valid)
    True

    """
    cdef bool is_valid
    is_valid = (0 <= x) & (x <= xmax) & (0 <= y) & (y <= ymax)
    return is_valid
