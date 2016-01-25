# -*- coding: utf-8 -*-
r"""
Pentago board game as a Python GUI.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Logging
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

#%% Imports
from .classes   import GameStats, Move, State # TODO: update these
from .constants import COLOR, INT_TOKEN, PLAYER, ONE_OFF, OPTIONS, SIZES, WIN
from .gui       import PentagoGui, RotationButton
from .plotting  import plot_board, plot_cur_move, plot_piece, plot_possible_win, plot_win
from .utils     import calc_cur_move, check_for_win, create_board_from_moves, find_moves, get_root_dir, rotate_board

#%% Unit Test
if __name__ == '__main__':
    pass
