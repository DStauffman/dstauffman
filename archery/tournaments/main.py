# -*- coding: utf-8 -*-
r"""
Main module file for the "dstauffman.archery" library.  It defines the main update and display
functions.

Notes
-----
#.  Written by David C. Stauffer in December 2014.
"""
# pylint: disable=C0326, E0611

#%% Imports
# normal imports
from __future__ import print_function
from __future__ import division
import os
import pandas as pd
# model imports
from dstauffman import write_text_file
from dstauffman.archery.tournaments.constants import \
    TEAMS_SIZE, MIXED_SIZE, COL_LASTNAME, COL_FIRSTNAME, COL_GENDER, COL_SCHOOL, COL_DIVISION, \
    COL_BALE, COL_SCORE1, COL_X_COUNT1, COL_SCORE2, COL_X_COUNT2, COL_SCORE_TOT, COL_X_COUNT_TOT, \
    COL_SEED, COL_TEAM_NUM, COL_LASTNAME1, COL_LASTNAME2, COL_LASTNAME3, COL_FIRSTNAME1, \
    COL_FIRSTNAME2, COL_FIRSTNAME3, COL_SCORE_PER1, COL_SCORE_PER2, COL_SCORE_PER3, \
    COL_X_COUNT_PER1, COL_X_COUNT_PER2, COL_X_COUNT_PER3, COL_FEMALE_LASTNAME, \
    COL_FEMALE_FIRSTNAME, COL_MALE_LASTNAME, COL_MALE_FIRSTNAME, COL_FEMALE_SCORE, \
    COL_FEMALE_X_COUNT, COL_MALE_SCORE, COL_MALE_X_COUNT
from dstauffman.archery.tournaments.utils import \
    generic_html_start, generic_html_end

#%% Functions (Private) - _set_seeds
def _set_seeds(data, max_seed=None):
    r"""
    Sorts the dataFrame by total score and total X count and gives a 1 to N seed number.

    Notes
    -----
    #. Assumes the dataFrame is already subselected to only be one division and gender combination.
    """
    # Sort the data by total score and then total X count
    data.sort([COL_SCORE_TOT, COL_X_COUNT_TOT], ascending=[False, False], inplace=True)
    # TODO: what if the scores and X counts are the same?
    if max_seed is None:
        num = len(data)+1
    else:
        num = min((len(data)+1, max_seed+1))
    data[COL_SEED][0:num-1] = range(1,num)
    # return updated dataframe with the seed numbers included
    return data

#%% Functions - write_registered_archers
def write_registered_archers(data, filename='', show_bales=True):
    r"""
    Writes the list of registered archers out to an html file.
    """
    # check that a non-empty filename was specified
    if len(filename) == 0:
        raise ValueError('A filename must be specified.')
    # get the title from the filename
    (_, title) = os.path.split(filename)
    # get the html prequel text
    html = generic_html_start()
    # replace the generic "Title" with this filename
    html = html.replace(r'<title>Title</title>', r'<title>' + title.split('.')[0] + r'</title>')
    # determine which columns to write out
    cols = [COL_LASTNAME, COL_FIRSTNAME, COL_GENDER, COL_SCHOOL, COL_DIVISION]
    if show_bales:
        cols.append(COL_BALE)
    # write the dataframe to an html text equivalent
    html_table = data.to_html(columns=cols, index_names=False, bold_rows=False, na_rep='')
    # combine all the html together
    html = html + html_table + generic_html_end()
    # write out to the specified file
    write_text_file(filename, html)

#%% Functions - write_indiv_results
def write_indiv_results(data, filename=''):
    r"""
    Writes the individual archer results out to an html file.
    """
    # check that a non-empty filename was specified
    if len(filename) == 0:
        raise ValueError('A filename must be specified.')
    # get the title from the filename
    (_, title) = os.path.split(filename)
    # get the html prequel text
    html = generic_html_start()
    # replace the generic "Title" with this filename
    html = html.replace(r'<title>Title</title>', r'<title>' + title.split('.')[0] + r'</title>')
    # determine which columns to write out
    cols = [COL_LASTNAME, COL_FIRSTNAME, COL_GENDER, COL_SCHOOL, COL_DIVISION, COL_BALE, \
            COL_SCORE1, COL_X_COUNT1, COL_SCORE2, COL_X_COUNT2, COL_SCORE_TOT, COL_X_COUNT_TOT, COL_SEED]
    # write the dataframe to an html text equivalent
    html_table = data.to_html(columns=cols, index_names=False, bold_rows=False, na_rep='')
    # combine all the html together
    html = html + html_table + generic_html_end()
    # write out to the specified file
    write_text_file(filename, html)

#%% Functions - update_indiv
def update_indiv(data, use_gender=True):
    r"""
    Updates the individual results with a seed number by division and gender.
    """
    if use_gender:
        # apply the seed when grouped by division and gender
        data_out = data.groupby([COL_DIVISION, COL_GENDER], group_keys=False).apply(_set_seeds)
    else:
        # apply the seed only by division (for mixed teams)
        data_out = data.groupby(COL_DIVISION, group_keys=False).apply(_set_seeds)
    # resort based on index
    data_out.sort(None, inplace=True)
    # return update dataframe
    return data_out

#%% Functions - update_teams
def update_teams(data_indiv, data_teams):
    r"""
    Updates the team results based on the highest three individuals from a particular school.

    Notes
    -----
    #. Substitutions are handled outside of this function.
    """
    # pull out the relevant subgroups
    grouped = data_indiv.groupby([COL_DIVISION, COL_GENDER, COL_SCHOOL], group_keys=False)
    # initialize a counter
    counter = 0
    # loop through groups
    for (this_key, this_group) in grouped:
        # find groups with at least the required number of people
        if len(this_group) >= TEAMS_SIZE:
            # sort this group by score
            this_group.sort([COL_SCORE_TOT, COL_X_COUNT_TOT], ascending=[False, False], inplace=True)
            # build the information:
            # Team
            data_teams.ix[counter, COL_TEAM_NUM] = counter + 1 # TODO: can't be an empty frame
            # TODO: try this? data_teams = data_teams.append(pd.DataFrame({COL_TEAM_NUM: counter + 1}, range(1)), ignore_index=True)
            # School
            data_teams.ix[counter, COL_SCHOOL]   = this_group.iloc[0][COL_SCHOOL]
            # Division
            data_teams.ix[counter, COL_DIVISION] = this_group.iloc[0][COL_DIVISION]
            # Gender
            data_teams.ix[counter, COL_GENDER]   = this_group.iloc[0][COL_GENDER]
            # Names
            data_teams.ix[counter, COL_LASTNAME1]  = this_group.iloc[0][COL_LASTNAME]
            data_teams.ix[counter, COL_LASTNAME2]  = this_group.iloc[1][COL_LASTNAME]
            data_teams.ix[counter, COL_LASTNAME3]  = this_group.iloc[2][COL_LASTNAME]
            data_teams.ix[counter, COL_FIRSTNAME1] = this_group.iloc[0][COL_FIRSTNAME]
            data_teams.ix[counter, COL_FIRSTNAME2] = this_group.iloc[1][COL_FIRSTNAME]
            data_teams.ix[counter, COL_FIRSTNAME3] = this_group.iloc[2][COL_FIRSTNAME]
            # Scores
            data_teams.ix[counter, COL_SCORE_PER1]   = this_group.iloc[0][COL_SCORE_TOT]
            data_teams.ix[counter, COL_SCORE_PER2]   = this_group.iloc[1][COL_SCORE_TOT]
            data_teams.ix[counter, COL_SCORE_PER3]   = this_group.iloc[2][COL_SCORE_TOT]
            data_teams.ix[counter, COL_X_COUNT_PER1] = this_group.iloc[0][COL_X_COUNT_TOT]
            data_teams.ix[counter, COL_X_COUNT_PER2] = this_group.iloc[1][COL_X_COUNT_TOT]
            data_teams.ix[counter, COL_X_COUNT_PER3] = this_group.iloc[2][COL_X_COUNT_TOT]
            # increment counter
            counter += 1
    # Combined Scores
    data_teams[COL_SCORE_TOT]   = data_teams[COL_SCORE_PER1] + data_teams[COL_SCORE_PER2] + data_teams[COL_SCORE_PER3]
    data_teams[COL_X_COUNT_TOT] = data_teams[COL_X_COUNT_PER1] + data_teams[COL_X_COUNT_PER2] + data_teams[COL_X_COUNT_PER3]
    # Set seeds
    data_teams_out = update_indiv(data_teams)
    return data_teams_out

#%% Functions - update_mixed
def update_mixed(data_indiv, data_mixed):
    r"""
    Updates the mixed team results based on the highest female and male individuals from a particular school.

    Notes
    -----
    #. Substitutions are handled outside of this function.
    """
    # pull out the relevant subgroups
    grouped = data_indiv.groupby([COL_DIVISION, COL_SCHOOL], group_keys=False)
    # initialize a counter
    counter = 0
    # loop through groups
    for (this_key, this_group) in grouped:
        # ensure one male and one female
        sub_grouped = this_group.groupby([COL_GENDER], group_keys=False)
        if len(sub_grouped) >= MIXED_SIZE:
            # initialize a new frame
            mixed_frame = pd.DataFrame()
            # loop through genders
            for (_, this_gender_group) in sub_grouped:
                # sort by score
                this_gender_group.sort([COL_SCORE_TOT, COL_X_COUNT_TOT], ascending=[False, False], inplace=True)
                # append person
                mixed_frame = mixed_frame.append(this_gender_group.iloc[0], ignore_index=True)
            # sort by gender
            mixed_frame.sort(COL_GENDER, ascending=True, inplace=True)
            # build the information:
            # Team
            data_mixed.ix[counter, COL_TEAM_NUM] = counter + 1 # TODO: can't be an empty frame
            # School
            data_mixed.ix[counter, COL_SCHOOL]   = mixed_frame.iloc[0][COL_SCHOOL]
            # Division
            data_mixed.ix[counter, COL_DIVISION] = mixed_frame.iloc[0][COL_DIVISION]
            # Names
            data_mixed.ix[counter, COL_FEMALE_LASTNAME]  = mixed_frame.iloc[0][COL_LASTNAME]
            data_mixed.ix[counter, COL_FEMALE_FIRSTNAME] = mixed_frame.iloc[0][COL_FIRSTNAME]
            data_mixed.ix[counter, COL_MALE_LASTNAME]    = mixed_frame.iloc[1][COL_LASTNAME]
            data_mixed.ix[counter, COL_MALE_FIRSTNAME]   = mixed_frame.iloc[1][COL_FIRSTNAME]
            # Scores
            data_mixed.ix[counter, COL_FEMALE_SCORE]   = mixed_frame.iloc[0][COL_SCORE_TOT]
            data_mixed.ix[counter, COL_MALE_SCORE]     = mixed_frame.iloc[1][COL_SCORE_TOT]
            data_mixed.ix[counter, COL_FEMALE_X_COUNT] = mixed_frame.iloc[0][COL_X_COUNT_TOT]
            data_mixed.ix[counter, COL_MALE_X_COUNT]   = mixed_frame.iloc[1][COL_X_COUNT_TOT]
            # increment counter
            counter += 1
    # Combined Scores
    data_mixed[COL_SCORE_TOT]   = data_mixed[COL_FEMALE_SCORE] + data_mixed[COL_MALE_SCORE]
    data_mixed[COL_X_COUNT_TOT] = data_mixed[COL_FEMALE_X_COUNT] + data_mixed[COL_MALE_X_COUNT]
    # Set seeds
    data_mixed_out = update_indiv(data_mixed, use_gender=False)
    return data_mixed_out

#%% Unit test
if __name__ == '__main__':
    pass

