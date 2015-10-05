# -*- coding: utf-8 -*-
r"""
Utils module file for the "dstauffman.archery" library.  It defines common utilities for use in
the rest of the code.

Notes
-----
#.  Written by David C. Stauffer in February 2015.
"""

#%% Imports
# normal imports
from __future__ import print_function
from __future__ import division
import os
import pandas as pd
# model imports
from dstauffman.archery.tournaments.constants import \
    COL_GENDER, COL_DIVISION, DIVISIONS, GENDERS, SHEET_NAME_INDIV

#%% Functions - display_info
def display_info(data):
    r"""
    Displays summary information about the registered archers.
    """
    # find numbers of each gender/division
    for div in DIVISIONS:
        for sex in GENDERS:
            temp = data[(data[COL_GENDER] == sex) & (data[COL_DIVISION] == div)].index
            print('Div: ' + sex + ' ' + div + ' {}'.format(len(temp)))
    # find numbers by school
    print(data.groupby('School').size().head())

#%% Functions - import_from_excel
def import_from_excel(filename, sheet=SHEET_NAME_INDIV):
    r"""
    Reads the spreadsheet data in from the given filename.
    """
    # create Excel file object
    xls = pd.ExcelFile(filename)
    # parse data and return
    data = xls.parse(sheet, index_col=None, na_values=['NA'])
    return data

#%% Functions - export_to_excel
def export_to_excel(data, filename, sheet=SHEET_NAME_INDIV):
    r"""
    Writes the spreadsheet data out to the give filename.
    """
    # force inputs to always be lists
    if not isinstance(data, list):
        data = [data]
    if not isinstance(sheet, list):
        sheet = [sheet]
    # Determine if writing to csv or excel
    file_ext = filename.split('.')[-1]
    if file_ext in {'xlsx', 'xls'}:
        # open the Excel writer
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        # write out data to object
        for i in range(len(data)):
            data[i].to_excel(writer, sheet_name=sheet[i])
        # write to disk
        writer.save()
    elif file_ext == 'csv':
        # check that only one set is used when writing to csv
        if len(data) > 1:
            raise ValueError('Only one data set can be saved to a csv.')
        # write to CSV file
        data[0].to_csv(filename)
    else:
        raise ValueError('Unexpected file extension.')

#%% Functions - generic_html_start
def generic_html_start():
    r"""Creates the start of a generic html file."""
    lines = []
    lines.append(r'<!DOCTYPE html>')
    lines.append(r'<html>')
    lines.append(r'<head>')
    lines.append(r'    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />')
    lines.append(r'    <title>Title</title>')
    lines.append(r'    <link rel="stylesheet" type="text/css" href="bracket.css" />')
    lines.append(r'</head>')
    lines.append(r'')
    lines.append(r'<body>')
    lines.append(r'')
    return '\n'.join(lines)

#%% Functions - generic_html_end
def generic_html_end():
    r"""Creates the end of a generic html file."""
    lines = []
    lines.append(r'')
    lines.append(r'</body>')
    lines.append(r'</html>')
    return '\n'.join(lines)

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

    >>> from dstauffman import get_root_dir
    >>> folder = get_root_dir()

    """
    # this folder is the root directory based on the location of this file (utils.py)
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return folder

#%% Unit test
if __name__ == '__main__':
    pass

