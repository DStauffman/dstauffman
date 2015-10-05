# -*- coding: utf-8 -*-
r"""
Simulations module file for the "dstauffman.archery" library.  It defines constants.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""
# pylint: disable=E1101, C0326

#%% Imports
# normal imports
from __future__ import print_function
from __future__ import division
import numpy as np
# model imports
from dstauffman.archery.tournaments.constants import \
    COL_SCORE1, COL_X_COUNT1, COL_SCORE2, COL_X_COUNT2, COL_SCORE_TOT, COL_X_COUNT_TOT

#%% Local constants
MAX_INDIV_SCORE = 360
MAX_INDIV_X     = 36
MAX_SET         = 6
MAX_CUM         = 150

#%% Functions - simulate_individual_scores
def simulate_individual_scores(data):
    r"""
    Simulates the individual scoring rounds.
    """
    # get the number of archers
    num_archers           = len(data)
    # generate random scores and X counts
    random_scores1        = np.random.randint(0, MAX_INDIV_SCORE, size=num_archers)
    random_x1             = np.random.randint(0, MAX_INDIV_X,     size=num_archers)
    random_scores2        = np.random.randint(0, MAX_INDIV_SCORE, size=num_archers)
    random_x2             = np.random.randint(0, MAX_INDIV_X,     size=num_archers)
    # update the scores
    data[COL_SCORE1]      = random_scores1
    data[COL_X_COUNT1]    = random_x1
    data[COL_SCORE2]      = random_scores2
    data[COL_X_COUNT2]    = random_x2
    # update the combined score columns
    data[COL_SCORE_TOT]   = random_scores1 + random_scores2
    data[COL_X_COUNT_TOT] = random_x1 + random_x2
    # return the updated answer, TODO: I don't know why this set isn't necessary
    return data

#%% Functions - simulate_bracket_scores
def simulate_bracket_scores(data, round_=''):
    r"""
    Simulates the bracket scores, round by round.
    """
    if len(round_) == 0:
        return

#%% Unit test function
def _main():
    r"""Unit test case."""
    pass

#%% Unit test
if __name__ == '__main__':
    _main()

