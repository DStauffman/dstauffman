# -*- coding: utf-8 -*-
r"""
Archery scoring __main__ function that runs on execution.

Notes
-----
#.  Adapted to a __main__ function by David C. Stauffer in February 2016.
#.  This model has three methods of running based on command arguments, 'null', 'test' or 'run'.
    The 'run' is the default.  'test' executes the unit tests, and 'null' does nothing.
"""

#%% Imports
import doctest
import getpass
import os
import sys
import unittest
import dstauffman.archery.scoring as score
from dstauffman import Opts

#%% Argument parsing
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    mode = 'run'

#%% Execution
if mode == 'run':
    # folder and file locations
    username        = getpass.getuser()
    folder          = os.path.join(r'C:\Users', username, r'Google Drive\Python\2015-16_Indoor_Scores')
    xlsx_datafile   = os.path.join(folder, '2015-16 Indoor Scorecards.xlsx')
    html_scoresheet = os.path.join(folder, 'scoresheet.htm')

    # opts settings for plots
    opts = Opts()
    opts.case_name = 'David'
    opts.save_path = folder
    opts.save_plot = True
    opts.plot_type = 'png'

    # process data and create HTML report
    (scores, names, dates) = score.read_from_excel_datafile(xlsx_datafile)
    fig = score.plot_mean_and_std(scores, opts)
    score.create_scoresheet(html_scoresheet, scores, names, 'David - Score Distribution.png')
    #plt.close(fig)

    # For Katie:
    #xlsx_datafile2   = os.path.join(folder, '2014-15 Indoor Scorecards-Katie Novotny.xlsx')
    #html_scoresheet2 = os.path.join(folder, 'scoresheet_katie.htm')
    #opts.case_name = 'Katie'
    #(scores2, names2, dates2) = read_from_excel_datafile(xlsx_datafile2)
    #plot_mean_and_std(scores2, opts)
    #create_scoresheet(html_scoresheet2, scores2, names2, 'Katie - Score Distribution.png')

elif mode == 'test':
    # find the test cases
    test_suite = unittest.TestLoader().discover('dstauffman.archery.scoring')
    # run the tests
    unittest.TextTestRunner(verbosity=1).run(test_suite)
    # run the docstrings
    verbose = False
    folder = score.get_root_dir()
    files = ['scoring']
    for file in files:
        if verbose:
            print('')
            print('******************************')
            print('******************************')
            print('Testing ' + file + '.py:')
        doctest.testfile(os.path.join(folder, file+'.py'), report=True, verbose=verbose, module_relative=True)
elif mode == 'null':
    pass
else:
    raise ValueError('Unexpected mode of "{}".'.format(mode))
