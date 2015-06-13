# -*- coding: utf-8 -*-
r"""
The "archery" module is a collection of code to support tournament scoring, bracket generation and
field layout.

The module is broken up into multiple files.
.__init__.py  includes the relative imports to be executed when someone does "import archery"
.bales.py     includes the logic to assign bales based on the list of registered archers.
.brackets.py  includes the logic to generate brackets based on individual seed order.
.constants.py includes global constants that don't change during execution.
.main.py      includes nothing right now.  It might later include more of a wrapper function.
.pretty.py    includes the functions that create files for user output in a "pretty" format.
.utils.py     includes general utilities that are used by the different pieces of the code.


The master data is defined in an excel spreadsheet.  Details TBD...



Notes
-----
#.  Written by David C. Stauffer in January 2015.
#.  Originally developed by David C. Stauffer for use by Stanford University.  It was
    then continued as a fun coding project and a way to learn more about Python programming.  This
    code is intended to be open source and  freely distributed.  Hopefully it will grow to be useful
    for other people and other tournaments.
#.  Ported to the "dstauffman" library by David C. Stauffer in June 2015.
"""
# pylint: disable=C0301

#%% Relative imports
from .bales       import assign_bales, validate_bales
from .brackets    import write_brackets, build_seed_order, get_name_by_index, validate_brackets, \
                         build_bracket_html_template, replace_css, replace_bracket_tokens
from .constants   import GENDERS, DIVISIONS, BALE_POS, TEAMS_SIZE, MIXED_SIZE, \
                         SHEET_NAME_INDIV, SHEET_NAME_TEAMS, SHEET_NAME_MIXED, \
                         COL_LASTNAME, COL_FIRSTNAME, COL_GENDER, COL_SCHOOL, COL_DIVISION, COL_BALE, \
                         COL_SCORE1, COL_SCORE2, COL_SCORE_TOT, COL_X_COUNT1, COL_X_COUNT2, COL_X_COUNT_TOT, \
                         COL_SEED, COL_32_BALE, COL_32_SCORE, COL_32_WIN, COL_16_BALE, COL_16_SCORE, COL_16_WIN, \
                         COL_08_BALE, COL_08_SCORE, COL_08_WIN, COL_04_BALE, COL_04_SCORE, COL_04_WIN, \
                         COL_02_BALE, COL_02_SCORE, COL_02_WIN, COL_01_BALE, COL_01_SCORE, COL_01_WIN, \
                         COL_TEAM_NUM, COL_LASTNAME1, COL_LASTNAME2, COL_LASTNAME3, COL_FIRSTNAME1, \
                         COL_FIRSTNAME2, COL_FIRSTNAME3, COL_SCORE_PER1, COL_SCORE_PER2, COL_SCORE_PER3, \
                         COL_X_COUNT_PER1, COL_X_COUNT_PER2, COL_X_COUNT_PER3, \
                         COL_FEMALE_LASTNAME, COL_FEMALE_FIRSTNAME, COL_MALE_LASTNAME, COL_MALE_FIRSTNAME, \
                         COL_FEMALE_SCORE, COL_FEMALE_X_COUNT, COL_MALE_SCORE, COL_MALE_X_COUNT, \
                         MAX_WAVES
from .main        import write_registered_archers, write_indiv_results, update_indiv, update_teams, \
                         update_mixed
from .simulations import simulate_individual_scores, simulate_bracket_scores
from .utils       import display_info, import_from_excel, export_to_excel, generic_html_start, \
                         generic_html_end, get_root_dir

#%% Unit test
if __name__ == '__main__':
    pass
