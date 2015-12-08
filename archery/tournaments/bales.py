# -*- coding: utf-8 -*-
r"""
Bales module file for the "dstauffman.archery.tournaments" library.  It defines functions to assign
bales based on the list of registered archers.

Notes
-----
#.  Written by David C. Stauffer in Feb 2015.
"""

#%% Imports
import random
from dstauffman.archery.tournaments.constants import \
    DIVISIONS, GENDERS, COL_GENDER, COL_DIVISION, COL_BALE, BALE_POS

#%% Functions - assign_bales
def assign_bales(data):
    r"""
    Randomly assigns the archers to bales based on division and gender.
    """
    # find sub data frame for each division/gender combination
    counter = 0
    n_pos   = len(BALE_POS)
    for div in DIVISIONS:
        for sex in GENDERS:
            # get index for this subgroup
            ix1 = data[(data[COL_GENDER].str.startswith(sex[0])) & (data[COL_DIVISION] == div)].index
            # for now, randomly assign to bale with no other checks by building an integer list
            n   = len(ix1)
            ix2 = random.sample(range(n), n)
            # assign bale numbers based on list
            for i in ix2:
                this_index = ix1[i]
                # get text for this bale
                this_bale = '{}'.format(counter // n_pos + 1) + BALE_POS[counter % n_pos]
                # assign bale to this person
                data.ix[this_index, COL_BALE] = this_bale
                # increment counter for next person
                counter = counter + 1
            # force next group to start on a new bale
            # (This line is cryptic, the -(-a//b) does a ceiling command)
            counter = (-(-counter // n_pos))*n_pos
    return data

#%% Functions - validate_bales
def validate_bales(data):
    r"""
    Error checks the bale assignments and displays any problems.
    """
    # get bale, spot on bale and corresponding index
    bales = [int(x[0:-1]) for x in data[COL_BALE].values]
    # spots = [x[-1] for x in data['Bale'].values]
    # ix    = data['Bale'].index
    num_bales = max(bales)
    print('Validating Bale assignments ...')
    for i in range(0, num_bales):
        sub_ix = [a for a, b in enumerate(bales) if b == i+1]
        count = len(sub_ix)
        if count == 0:
            print(' Bale {} has no people assigned.'.format(i+1))
        elif count < 3:
            print(' Bale {} has less than 3 people assigned.'.format(i+1))
        elif count > 4:
            print(' Bale {} has more than 4 people assigned.'.format(i+1))
        else:
            pass
    print('Done.')

#%% Unit test
if __name__ == '__main__':
    pass

