# -*- coding: utf-8 -*-
r"""
Tic Tac Toe board game as a Python GUI.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
from .classes   import GameStats, Move, State
from .constants import COLOR, LOGGING, PLAYER, OPTIONS, SIZES, WIN
from .gui       import TicTacToeGui
from .plotting  import plot_board, plot_cur_move, plot_piece, plot_possible_win, plot_powers, \
                       plot_win
from .utils     import get_root_dir, calc_cur_move, check_for_win, create_board_from_moves, \
                       find_moves, make_move, play_ai_game

#%% Unit Test
if __name__ == '__main__':
    pass
