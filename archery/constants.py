# -*- coding: utf-8 -*-
r"""
Constants module file for the "dstauffman.archery" library.  It defines constants.

Notes
-----
#.  Written by David C. Stauffer in February 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division

#%% Constants
# genders, divisions and possible bale positions
GENDERS   = ['Female', 'Male']
DIVISIONS = ['Barebow', 'Recurve', 'Bowhunter', 'Compound', 'Guest']
BALE_POS  = ['A', 'B', 'C', 'D']

# Team sizes
TEAMS_SIZE = 3
MIXED_SIZE = 2

# Excel sheetnames
SHEET_NAME_INDIV = 'Archers'
SHEET_NAME_TEAMS = 'Teams'
SHEET_NAME_MIXED = 'Mixed'

# Excel column header names
COL_LASTNAME     = 'Lastname'
COL_FIRSTNAME    = 'Firstname'
COL_GENDER       = 'Gender'
COL_SCHOOL       = 'School'
COL_DIVISION     = 'Division'
COL_BALE         = 'Bale'
COL_SCORE1       = 'Score 1'
COL_X_COUNT1     = "X's 1"
COL_SCORE2       = 'Score 2'
COL_X_COUNT2     = "X's 2"
COL_SCORE_TOT    = 'Total Score'
COL_X_COUNT_TOT  = "Total X's"
COL_SEED         = 'Seed'

# round headings
COL_32_BALE      = r'1/32 Bale'
COL_32_SCORE     = r'1/32 Score'
COL_32_WIN       = r'1/32 Win'
COL_16_BALE      = r'1/16 Bale'
COL_16_SCORE     = r'1/16 Score'
COL_16_WIN       = r'1/16 Win'
COL_08_BALE      = r'1/8 Bale'
COL_08_SCORE     = r'1/8 Score'
COL_08_WIN       = r'1/8 Win'
COL_04_BALE      = r'1/4 Bale'
COL_04_SCORE     = r'1/4 Score'
COL_04_WIN       = r'1/4 Win'
COL_02_BALE      = r'1/2 Bale'
COL_02_SCORE     = r'1/2 Score'
COL_02_WIN       = r'1/2 Win'
COL_01_BALE      = 'Final Bale'
COL_01_SCORE     = 'Final Score'
COL_01_WIN       = 'Final Win'
COL_01_PLACE     = 'Final Place'

# additional "Teams" sheet headings
COL_TEAM_NUM     = 'Team'
COL_LASTNAME1    = 'Lastname1'
COL_LASTNAME2    = 'Lastname2'
COL_LASTNAME3    = 'Lastname3'
COL_FIRSTNAME1   = 'Firstname1'
COL_FIRSTNAME2   = 'Firstname2'
COL_FIRSTNAME3   = 'Firstname3'
COL_SCORE_PER1   = 'Person1 Score'
COL_SCORE_PER2   = 'Person2 Score'
COL_SCORE_PER3   = 'Person3 Score'
COL_X_COUNT_PER1 = "Person1 X's"
COL_X_COUNT_PER2 = "Person2 X's"
COL_X_COUNT_PER3 = "Person3 X's"

# additional "Mixed" sheet headings
COL_FEMALE_LASTNAME  = 'Female Lastname'
COL_FEMALE_FIRSTNAME = 'Female Firstname'
COL_MALE_LASTNAME    = 'Male Lastname'
COL_MALE_FIRSTNAME   = 'Male Firstname'
COL_FEMALE_SCORE     = 'Female Score'
COL_FEMALE_X_COUNT   = "Female X's"
COL_MALE_SCORE       = 'Male Score'
COL_MALE_X_COUNT     = "Male X's"

# limits
MAX_WAVES = 5 # (1/16 round with 32 people as maximum starting round)

#%% Unit test
if __name__ == '__main__':
    pass
