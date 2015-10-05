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
import csv
from datetime import datetime, timedelta
import doctest
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unittest
from xml.dom import minidom
# library imports
from dstauffman import setup_dir

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

#%% Functions - scores_to_totals
def scores_to_totals(scores,flag='nfaa'):
    r"""
    Convert an ndarray of mixed text and numeric scores into the desired total score history.
    """
    x=scores
    if flag == 'nfaa':
        num_scores = np.reshape(np.array([0 if x[i,j] == 'M' else x[i,j] if x[i,j] != 'X' else 10 \
        for i in range(0,x.shape[0]) for j in range(0,x.shape[1])],dtype=int),x.shape)
    elif flag == 'usaa':
        num_scores = np.reshape(np.array([0 if x[i,j] == 'M' else 9 if x[i,j]==10 else 10 if x[i,j] == 'X' else x[i,j] \
        for i in range(0,x.shape[0]) for j in range(0,x.shape[1])],dtype=int),x.shape)
    else:
        raise ValueError('Unexpected value for flag = "{}"'.format(flag))
    total_scores = np.sum(num_scores,axis=1)
    return total_scores

#%% Functions - convert_data_to_scores
def convert_data_to_scores(all_data):
    r"""
    Makes the USAA and NFAA scores from the individual arrow score histories.
    """
    # initialize lists
    nfaa_score = []
    usaa_score = []
    # convert string scores into numbers based on NFAA and USAA scoring rules
    for i in range(0,len(all_data)):
        nfaa_score.append(sum([score_text_to_number(x,'nfaa') for x in all_data[i]]))
        usaa_score.append(sum([score_text_to_number(x,'usaa') for x in all_data[i]]))
    return (nfaa_score,usaa_score)

#%% Functions - plot_mean_and_std
def plot_mean_and_std(all_data, filename=''):
    r"""
    Plots the bell curve based on the given scores.
    """
    (nfaa_score,usaa_score) = convert_data_to_scores(all_data)
    # calculate mean and standard deviations, use pandas Series instead of numpy for N-1 definition of std.
    nfaa_mean   = np.mean(nfaa_score)
    usaa_mean   = np.mean(usaa_score)
    nfaa_std    = pd.Series(nfaa_score).std()
    usaa_std    = pd.Series(usaa_score).std()
    # create score range to evaluate for plotting
    dt          = 0.1
    score_range = np.arange(0,300+dt,dt)
    # create actuals for scores
    act_range   = np.arange(PLOT_LIMITS[0],PLOT_LIMITS[1]+1)
    nfaa_acts   = np.empty(len(act_range))
    usaa_acts   = np.empty(len(act_range))
    num_scores  = len(nfaa_score)
    for ix, score in enumerate(act_range):
        nfaa_acts[ix] = np.sum(nfaa_score == score) / num_scores
        usaa_acts[ix] = np.sum(usaa_score == score) / num_scores
    # create plot
    plt.figure()
    plt.plot(score_range,100*normal_curve(score_range,nfaa_mean,nfaa_std),'r')
    if PLOT_ACTUALS:
        plt.plot(act_range,100*nfaa_acts,'r.')
    plt.plot(score_range,100*normal_curve(score_range,usaa_mean,usaa_std),'b')
    if PLOT_ACTUALS:
        plt.plot(act_range,100*usaa_acts,'b.')
    plt.xlabel('Score')
    plt.ylabel('Distribution [%]')
    plt.title('Score Distribution')
    plt.xlim(PLOT_LIMITS)
    if PLOT_ACTUALS:
        plt.legend(['NFAA Normal','NFAA Actuals','USAA Normal','USAA Actuals'])
    else:
        plt.legend(['NFAA','USAA'])
    plt.grid(True)
    plt.show()
    plt.savefig(filename)

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
    with np.errstate(invalid='ignore'): # because mu or sigma can be zero
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return y

#%% Functions - excel_date_to_str
def excel_date_to_str(excel_num):
    r"""
    Convert an excel date to a Python datetime.datetime object.
    """
    date_zero = datetime(1899, 12, 30, 0, 0, 0)
    num_zero = 0

    dt = timedelta(days=excel_num - num_zero)

    date = date_zero + dt

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

    >>> from dstauffman.archery.scoring import read_from_excel_datafile
    >>> filename = r"C:\Users\DStauffman\Documents\Archery\2015-16 Indoor\2015-16 Indoor Scorecards.xlsx"
    >>> (scores, names) = read_from_excel_datafile(filename)

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

    # return a tuple of the scores and associated names
    return (scores, names)

#%% Functions - read_from_xml_datafile
def read_from_xml_datafile(filename, display=True):
    r"""
    Reads the score data from the given xml data file and also returns the archer names.
    """
    dom    = minidom.parse(filename)
    rounds = dom.getElementsByTagName('round')
    all_data = []
    names    = []
    for i in range(0,len(rounds)):
        this_round = rounds[i]
        data       = []
        archer     = this_round.getAttribute('archer')
        date       = this_round.getAttribute('date')
        # TODO: make this check for <end> tag in future.
        ends       = this_round.childNodes
        for end in ends:
            scores = end.childNodes
            for score in scores:
                if score.nodeType == score.ELEMENT_NODE:
                    data.append(score.firstChild.nodeValue)
        all_data.append(data)
        names.append(archer + ' (' + date + ')')
        if display:
            print('Archer: ' + archer)
            print('Date:   ' + date)
            print('Scores: ',data,sep='')
    if display:
        print(' ')
    return (all_data,names)

#%% Functions - read_from_csv_datafile
def read_from_csv_datafile(filename):
    r"""
    Reads the score data from the given csv data file.
    """
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            print(', '.join(row))

#%% Functions - create_scoresheet
def create_scoresheet(filename, data, names, plotname='scores.png'):
    r"""
    Creates an HTML file scoresheet of the given information.
    """

    # determine if making a file
    if filename == "":
        make_file = False
    else:
        make_file = True

    # TODO: want to move CSS information into a separate style sheet in the future.
    htm = """<!DOCTYPE html>
<html>
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

<body>
"""
    table1 = """<table class="score_table">
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

    plot1 = """<p><img src=""" + plotname + """ alt="Normal Distribution plot" height="597" width="800"> </p>
"""

    for i in range(0,len(data)):
        table1 = table1 + ' <tr>\n  <td rowspan="2">' + names[i] + '</td>\n'
        this_data   = data[i]
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
        with open(filename,'w') as f:
            f.write(htm)

    return htm

#%% Unit test logic
def main():
    folder          = r'C:\Users\DStauffman\Documents\Archery\2015-16 Indoor'
    xlsx_datafile   = os.path.join(folder, '2015-16 Indoor Scorecards.xlsx')
    #xml_datafile    = os.path.join(folder, 'score_sheet.xml')
    html_scoresheet = os.path.join(folder, 'scoresheet.htm')
    csv_datafile    = os.path.join(folder, '2015-16 Score History.csv')

    # (all_data,names) = read_from_xml_datafile(xml_datafile)
    (all_data,names) = read_from_excel_datafile(xlsx_datafile)
    create_scoresheet(html_scoresheet,all_data,names)
    plot_mean_and_std(all_data,os.path.join(folder,'scores.png'))

    # For Katie:
    #xlsx_datafile2   = os.path.join(folder,'2014-15 Indoor Scorecards-Katie Novotny.xlsx')
    #html_scoresheet2 = os.path.join(folder,'scoresheet_katie.htm')
    #(all_data2,names2) = read_from_excel_datafile(xlsx_datafile2)
    #create_scoresheet(html_scoresheet2,all_data2,names2,'scores_katie.png')
    #plot_mean_and_std(all_data2,os.path.join(folder,'scores_katie.png'))


#%% Unit test
if __name__ == '__main__':
    unittest.main(module='test_scoring', exit=False)
    doctest.testmod(verbose=False)
    main()
