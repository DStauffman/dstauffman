# -*- coding: utf-8 -*-
r"""
"scoring"  is a collection of Archery scoring code developed by David C. Stauffer.

The code is mostly for experimentation by David and may eventually lead to a useful application.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Adapted into dstauffman library by David C. Stauffer in October 2015.
"""
# pylint: disable=C0326, C0103, E1101

#%% Imports
from datetime import datetime, timedelta
import doctest
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unittest
# library imports
from dstauffman import setup_dir, Opts, setup_plots

#%% Constants
PLOT_LIMITS  = [250,300]
PLOT_ACTUALS = True

#%% Functions - get_root_dir
def get_root_dir():
    r"""
    Returns the folder that contains this source file and thus the root folder for the whole code.

    Returns
    -------
    folder : str
        Location of the folder that contains all the source files for the code.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman.archery.scoring import get_root_dir
    >>> folder = get_root_dir()

    """
    # this folder is the root directory based on the location of this file (utils.py)
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return folder

#%% Functions - score_text_to_number
def score_text_to_number(text, flag='nfaa'):
    r"""
    Converts text scores to numeric values, with options for X=10 and M=0, for both NFAA and USAA scoring.

    Parameters
    ----------
    text : str
        The text to be converted to a numeric score
    flag : str, optional
        The scoring system to use, from {'usaa', 'nfaa'}

    Returns
    -------
    value : int
        Equivalent numeric for the supplied text

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman.archery.scoring import score_text_to_number
    >>> text = ['X', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1', 'M']
    >>> nums = [score_text_to_number(x) for x in text]
    >>> print(nums)
    [10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    """
    # force flag to be lower case to match any case style
    flag = flag.lower()
    # handle str instances
    if isinstance(text, str):
        # force lowercase
        text = text.lower()
        # check for specific values, or convert to int
        if text == "x":
            return 10
        elif text == '10' and flag == 'usaa':
            return 9
        elif text == "m":
            return 0
        else:
            return int(text)
    else:
        # assume anything else is numeric and force it to be an int
        value = int(text)
        # check for USAA conversion case, otherwise return numeric value
        if value == 10 and flag == 'usaa':
            return 9
        else:
            return value

#%% Functions - convert_data_to_scores
def convert_data_to_scores(scores):
    r"""
    Makes the USAA and NFAA scores from the individual arrow score histories.

    Parameters
    ----------
    scores : list of list of int/str
        List of scores for each round based on sublist of scores for each arrow in the round

    Returns
    -------
    nfaa_score : ndarray of int
        Total round score for each given round in the original list

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman.archery.scoring import convert_data_to_scores
    >>> scores = [10*['X', 10, 9], 10*[9, 9, 9]]
    >>> (nfaa_score, usaa_score) = convert_data_to_scores(scores)
    >>> print(nfaa_score)
    [290, 270]

    >>> print(usaa_score)
    [280, 270]

    """
    # initialize lists
    nfaa_score = []
    usaa_score = []
    # convert string scores into numbers based on NFAA and USAA scoring rules
    for this_round in scores:
        nfaa_score.append(sum([score_text_to_number(x, 'nfaa') for x in this_round]))
        usaa_score.append(sum([score_text_to_number(x, 'usaa') for x in this_round]))
    return (nfaa_score, usaa_score)

#%% Functions - plot_mean_and_std
def plot_mean_and_std(scores, opts=None, perfect_score=300):
    r"""
    Plots the bell curve based on the given scores.

    Parameters
    ----------
    scores : list of list of int/str
        List of scores for each round based on sublist of scores for each arrow in the round
    opts : class Opts
        Optional plotting controls
    perfect_score : int, optional, defaults to 300
        Value of a perfect score used to determine range for plotting

    Returns
    -------
    fig : class object
        Figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.
    #.  Updated by David C. Stauffer in October 2015 to use setup_plots and Opts functionality.

    Examples
    --------

    >>> from dstauffman.archery.scoring import plot_mean_and_std
    >>> import matplotlib.pyplot as plt
    >>> scores = [10*['X', 10, 9], 10*[9, 9, 9]]
    >>> fig = plot_mean_and_std(scores)

    Close the figure
    >>> plt.close(fig)

    """
    # hard-coded values
    num2per = 100
    # check for optional arguments
    if opts is None:
        opts = Opts()
        opts.case_name = ''
    # split into NFAA and USAA scores
    (nfaa_score, usaa_score) = convert_data_to_scores(scores)
    # calculate mean and standard deviations, use pandas Series instead of numpy for N-1 definition of std.
    nfaa_mean   = np.mean(nfaa_score)
    usaa_mean   = np.mean(usaa_score)
    if len(nfaa_score) > 1:
        nfaa_std    = pd.Series(nfaa_score).std()
        usaa_std    = pd.Series(usaa_score).std()
    else:
        nfaa_std = 0
        usaa_std = 0
    # create score range to evaluate for plotting
    dt          = 0.1
    score_range = np.arange(0, perfect_score+dt, dt)
    # create actuals for scores
    act_range   = np.arange(PLOT_LIMITS[0], PLOT_LIMITS[1]+1)
    nfaa_acts   = np.empty(act_range.shape)
    usaa_acts   = np.empty(act_range.shape)
    num_scores  = len(nfaa_score)
    for (ix, score) in enumerate(act_range):
        nfaa_acts[ix] = np.sum(nfaa_score == score) / num_scores
        usaa_acts[ix] = np.sum(usaa_score == score) / num_scores
    # create figure
    fig = plt.figure()
    fig.canvas.set_window_title('Score Distribution')
    ax = fig.add_subplot(111)
    # plot data
    ax.plot(score_range, num2per*normal_curve(score_range, nfaa_mean, nfaa_std), 'r', label='NFAA Normal')
    if PLOT_ACTUALS:
        ax.bar(act_range, num2per*nfaa_acts, color='r', label='NFAA Actuals')
    ax.plot(score_range, num2per*normal_curve(score_range, usaa_mean, usaa_std), 'b', label='USAA Normal')
    if PLOT_ACTUALS:
        ax.bar(act_range, num2per*usaa_acts, color='b', label='USAA Actuals')
    # add labels and legends
    plt.xlabel('Score')
    plt.ylabel('Distribution [%]')
    plt.title(fig.canvas.get_window_title())
    plt.xlim(PLOT_LIMITS)
    plt.legend()
    plt.grid(True)
    # optionally save and format plot
    setup_plots(fig, opts, 'dist_no_y_scale')
    return fig

#%% Functions - normal_curve
def normal_curve(x, mu=0, sigma=1):
    r"""
    Calculates a normal bell curve for the given points, based on a mean and standard deviation.

    Parameters
    ----------
    x : array_like
        Values over which to evaluate the normal curve
    mu : float, optional, default is 0
        Mean of normal curve
    sigma : float, optional, default is 1
        Standard deviation of normal curve

    Returns
    -------
    y : array_like
        Normal curve values for each of the given x

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman.archery.scoring import normal_curve
    >>> import numpy as np
    >>> x = np.arange(-3, 3.01, 0.01)
    >>> y = normal_curve(x)

    """
    if sigma < 0:
        raise ValueError('The sigma must be positive, not {}.'.format(sigma))
    elif sigma == 0:
        y = np.where(x == mu, 1, 0)
    else:
    #with np.errstate(invalid='ignore', divide='ignore'): # because mu or sigma can be zero
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return y

#%% Functions - excel_date_to_str
def excel_date_to_str(excel_num):
    r"""
    Convert an excel date to a Python datetime.datetime object.

    Parameters
    ----------
    excel_num : float
        Scalar number that Excel uses to represent a date

    Returns
    -------
    date : str
        Equivalent string (YYYY-MM-DD) representation of the Excel date

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman.archery.scoring import excel_date_to_str
    >>> excel_num = 36526
    >>> date = excel_date_to_str(excel_num)
    >>> print(date)
    2000-01-01

    """
    # Excel's date zero as a Python datetime object
    date_zero = datetime(1899, 12, 30, 0, 0, 0)
    num_zero = 0

    # calculate the number of days since date zero
    dt = timedelta(days=excel_num - num_zero)

    # increment the datetime object
    date = date_zero + dt

    # output the object as a string
    return date.strftime('%Y-%m-%d')

#%% Functions - read_from_excel_datafile
def read_from_excel_datafile(filename):
    r"""
    Reads the score data from the given excel file and also returns the archer names.

    Parameters
    ----------
    filename : str
        Filename to read data from

    Returns
    -------
    scores : list of list of int/str
        Scores for each round
    names : list of str
        List of names for the scoring rounds

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman.archery.scoring import read_from_excel_datafile, get_root_dir
    >>> import os
    >>> filename = os.path.join(get_root_dir(), 'test_score_template.xlsx')
    >>> (scores, names, dates) = read_from_excel_datafile(filename)
    >>> print(scores[0][0:3])
    ['X' 9.0 9.0]

    >>> print(names[0])
    10/1/2015

    >>> print(dates[0])
    2015-10-01 00:00:00

    """
    # read data from excel into DataFrame
    data = pd.io.excel.read_excel(filename, sheetname='Scorecard', parse_cols='B:AT', skiprows=2)

    # get index to rows with valid scores
    ix = (data['Archer'].notnull())

    # reduce data
    subdata = data[ix]

    # drop rows with NaNs for scores in End 1
    subdata = subdata.dropna(subset=['End 1'])

    # column index to scoring ends
    cols = sorted(np.concatenate((range(1,40,4),range(2,40,4),range(3,40,4))))

    # pull out scores
    scores = subdata[cols].values

    # pull out names
    names = subdata['Archer'].values

    # convert names to dates
    dates = [datetime.strptime(this_name.split(' ')[0], '%m/%d/%Y') for this_name in names]

    # return a tuple of the scores and associated names
    return (scores, names, dates)

#%% Functions - create_scoresheet
def create_scoresheet(filename, scores, names, plotname='Score Distribution.png'):
    r"""
    Creates an HTML file scoresheet of the given information.

    Parameters
    ----------
    filename : str
        Filename to read data from
    scores : list of list of int/str
        Scores for each round
    names : list of str
        List of names for the scoring rounds
    plotname : str, optional
        Name of the plot to include in the scoresheet

    Returns
    -------
    html : str
        Resulting HTML mark-up code for htm file

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.

    Examples
    --------

    >>> from dstauffman.archery.scoring import create_scoresheet
    >>> filename = ''
    >>> scores = [10*['X', 10, 9], 10*[9, 9, 9]]
    >>> names = ['First Round', 'Second Round']
    >>> html = create_scoresheet(filename, scores, names)
    >>> print(html[:56])
    <!DOCTYPE html>
    <html>
    <head>
    <title>Score Sheet</title>

    """

    # determine if making a file
    if filename == "":
        make_file = False
    else:
        make_file = True

    # TODO: want to move CSS information into a separate style sheet in the future.
    htm = """<!DOCTYPE html>
<html>
<head>
<title>Score Sheet</title>

<style type="text/css">
table.score_table {
    font-family: verdana,arial,sans-serif;
    font-size:11px;
    border-width: 1px;
    border-color: #000000;
    border-collapse: collapse;
    color: inherit;
    background-color: #ffffff
}
table.score_table th {
    border-width: 1px;
    padding: 8px;
    border-style: solid;;
    background-color: #dedede;
}
table.score_table td {
    border-width: 1px;
    padding: 8px;
    border-style: solid;
    text-align: center;
}
table.score_table td.Y {
    background-color: #ffff00;
}
table.score_table td.R {
    background-color: #ff0000;
}
table.score_table td.B {
    background-color: #0099FF;
}
table.score_table td.K {
    color: #ffffff;
    background-color: #000000;
}
table.score_table td.W {
    background-color: #ffffff;
}
table.score_table td.M {
    color: #ff0000
    background-color: #ffffff;
}
span.red {color: #ff0000;}
span.white {color: #ffffff;}
</style>
</head>

<body>
"""
    table1 = """<table class="score_table">
 <tr>
  <th>Archer</th>
  <th colspan="4">End 1</th>
  <th colspan="4">End 2</th>
  <th colspan="4">End 3</th>
  <th colspan="4">End 4</th>
  <th colspan="4">End 5</th>
  <th colspan="4">End 6</th>
  <th colspan="4">End 7</th>
  <th colspan="4">End 8</th>
  <th colspan="4">End 9</th>
  <th colspan="4">End 10</th>
  <th>Total (NFAA)</th>
  <th>Total (USAA)</th>
  <th>Total (X's)</th>
  <th>Total (Hits)</th>
 </tr>
"""

    table2 = """<table class="score_table">
 <thead>
  <tr>
   <td colspan="13">Arrow Count</td>
  </tr>
  <tr>
   <td></td>
   <td class="Y">X</td>
   <td class="Y">10</td>
   <td class="Y">9</td>
   <td class="R">8</td>
   <td class="R">7</td>
   <td class="B">6</td>
   <td class="B">5</td>
   <td class="K"><span class="white">4</span></td>
   <td class="K"><span class="white">3</span></td>
   <td class="W">2</td>
   <td class="W">1</td>
   <td class="W"><span class="red">M</span></td>
  </tr>
 </thead>
"""

    plot1 = """<p><img src=""" + plotname.replace(' ','%20') + """ alt="Normal Distribution plot" height="597" width="800"> </p>
"""

    for i in range(0,len(scores)):
        table1 = table1 + ' <tr>\n  <td rowspan="2">' + names[i] + '</td>\n'
        this_data   = scores[i]
        this_data   = [x if isinstance(x,str) else str(int(x)) for x in this_data]
        this_nums   = [score_text_to_number(x) for x in this_data]
        this_cumsum = np.cumsum(this_nums)
        for j in range(0,len(this_data)):
            this_text = this_data[j]
            if this_text in {'X','x','10','9'}:
                c = 'Y'
                s = ''
            elif this_text in {'8','7'}:
                c = 'R'
                s = ''
            elif this_text in {'6','5'}:
                c = 'B'
                s = ''
            elif this_text in {'4','3'}:
                c = 'K"><span class="white'
                s = '</span>'
            elif this_text in {'2','1'}:
                c = 'W'
                s = ''
            elif this_text in {'M','m','0'}:
                c = 'M"><span class="red'
                s = '</span>'
            else:
                raise ValueError('Unexpected Value for score "{}".'.format(this_text))
            table1 = table1 + '  <td class="' + c + '">' + this_text + s + '</td>\n'
            if j % 3 == 2:
                table1 = table1 + '  <td>{}</td>\n'.format(np.sum(this_nums[j-2:j+1]))
        num_tens = np.sum([x=='10' for x in this_data])
        table1 = table1 + '  <td rowspan="2">{}</td>\n'.format(this_cumsum[-1])
        table1 = table1 + '  <td rowspan="2">{}</td>\n'.format(this_cumsum[-1]-num_tens)
        table1 = table1 + '  <td rowspan="2">{}</td>\n'.format(np.sum([x.lower()=='x' for x in this_data]))
        table1 = table1 + '  <td rowspan="2">{}</td>\n'.format(num_tens)
        table1 = table1 + ' </tr>\n <tr>\n'
        for j in range(0,len(this_data)):
            if j % 3 == 2:
                table1 = table1 + '  <td colspan="4">' + '{}'.format(this_cumsum[j]) + '</td>\n'
        table1 = table1 + ' </tr>\n'
        table2 = table2 + '<tr>'
        table2 = table2 + '<td>' + names[i] + '</td>\n'
        table2 = table2 + '<td class="Y">{}</td>\n'.format(np.sum([x.lower()=='x' for x in this_data]))
        table2 = table2 + '<td class="Y">{}</td>\n'.format(np.sum([x=='10' for x in this_data]))
        table2 = table2 + '<td class="Y">{}</td>\n'.format(np.sum([x=='9' for x in this_data]))
        table2 = table2 + '<td class="R">{}</td>\n'.format(np.sum([x=='8' for x in this_data]))
        table2 = table2 + '<td class="R">{}</td>\n'.format(np.sum([x=='7' for x in this_data]))
        table2 = table2 + '<td class="B">{}</td>\n'.format(np.sum([x=='6' for x in this_data]))
        table2 = table2 + '<td class="B">{}</td>\n'.format(np.sum([x=='5' for x in this_data]))
        table2 = table2 + '<td class="K"><span class="white">{}</span></td>\n'.format(np.sum([x=='4' for x in this_data]))
        table2 = table2 + '<td class="K"><span class="white">{}</span></td>\n'.format(np.sum([x=='3' for x in this_data]))
        table2 = table2 + '<td class="W">{}</td>\n'.format(np.sum([x=='2' for x in this_data]))
        table2 = table2 + '<td class="W">{}</td>\n'.format(np.sum([x=='1' for x in this_data]))
        table2 = table2 + '<td class="W"><span class="red">{}</span></td>\n'.format(np.sum([x.lower()=='m' or x == '0' for x in this_data]))
        table2 = table2 + '</tr>\n'
    table1 = table1 + '</table>\n'

    table2 = table2 + '</table>\n'

    htm = htm + table1 + '<br /><br />\n' + table2 + '<br /><br />' + plot1 + '</body>\n\n</html>\n'

    # write text to file
    if make_file:
        folder = os.path.split(filename)[0]
        if not os.path.isdir(folder):
            setup_dir(folder)
        with open(filename, 'w') as file:
            file.write(htm)

    return htm

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='test_scoring', exit=False)
    doctest.testmod(verbose=False)
