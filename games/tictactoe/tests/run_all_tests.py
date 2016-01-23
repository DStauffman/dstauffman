# -*- coding: utf-8 -*-
r"""
Test file to execute all the tests from the unittest library within the tictactoe code using nose.

Notes
-----
#.  Written by David C. Stauffer in January 2016.
"""

#%% Imports
import nose
import dstauffman.games.tictactoe as ttt

if __name__ == '__main__':
    nose.run(ttt)