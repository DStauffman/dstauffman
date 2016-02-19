# -*- coding: utf-8 -*-
r"""
Constants module file for the "tictactoe" game.  It defines constants.

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
COLOR['o']        = (1., 1., 1.)
COLOR['x']        = (0., 0., 0.)
COLOR['edge']     = (0., 0., 0.)
COLOR['win_o']    = (1.0, 0.9, 0.9)
COLOR['win_x']    = (0.2, 0.0, 0.0)
COLOR['win_ox']   = (0.7, 0.7, 0.7)

# player enumerations
PLAYER          = {}
PLAYER['o']     = 1
PLAYER['x']     = -1
PLAYER['none']  = 0
PLAYER['draw']  = 2

# piece radius
SIZES           = {}
SIZES['piece']  = 0.9
SIZES['square'] = 1.0
SIZES['board']  = 3

# Gameplay options
OPTIONS                       = {}
OPTIONS['load_previous_game'] = 'No' # from ['Yes','No','Ask']
OPTIONS['plot_best_moves']    = False
OPTIONS['plot_move_power']    = False
OPTIONS['o_is_computer']      = False
OPTIONS['x_is_computer']      = True

# all possible winning combinations
WIN = np.array([\
[1,0,0,1,0,0,1,0],\
[1,0,0,0,1,0,0,0],\
[1,0,0,0,0,1,0,1],\
\
[0,1,0,1,0,0,0,0],\
[0,1,0,0,1,0,1,1],\
[0,1,0,0,0,1,0,0],\
\
[0,0,1,1,0,0,0,1],\
[0,0,1,0,1,0,0,0],\
[0,0,1,0,0,1,1,0],\
], dtype=bool)

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.games.tictactoe.tests.test_constants', exit=False)
    doctest.testmod(verbose=False)
