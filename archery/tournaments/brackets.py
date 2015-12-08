# -*- coding: utf-8 -*-
r"""
Brackets module file for the "dstauffman.archery" library.  It defines functions to generate
brackets based on individual seed order.

Notes
-----
#.  Written by David C. Stauffer in January 2015.
"""
# pylint: disable=C0301, C0326

#%% Imports
# normal imports
import math
import os
# model imports
from dstauffman import read_text_file, write_text_file
from dstauffman.archery.tournaments.constants import \
    GENDERS, DIVISIONS, COL_LASTNAME, COL_FIRSTNAME, COL_GENDER, COL_DIVISION, MAX_WAVES, \
    COL_32_WIN, COL_16_WIN, COL_08_WIN, COL_04_WIN, COL_02_WIN, COL_01_WIN
from dstauffman.archery.tournaments.utils import \
    import_from_excel, export_to_excel, display_info
from dstauffman.archery.tournaments.bales import \
    assign_bales, validate_bales

#%% Functions - write_brackets
def write_brackets(data, filename='', team='indiv', display=True):
    r"""
    Writes out the brackets based on scores from earlier competition.
    """
    # initializ the bracket list
    brackets = []
    # loop through all the divisions
    for div in DIVISIONS:
        # loop through genders
        for sex in GENDERS:
            # get the index for this specific gender/division combination
            ix    = data[(data[COL_GENDER].str.startswith(sex[0])) & (data[COL_DIVISION] == div)].index
            # find out how many exist, and skip if this has zero length
            num   = len(ix)
            if num > 0:
                # calculate the number of waves needed for this number of people
                waves = int(math.ceil(math.log(num, 2)))
                # throw an error if this is above 5 (1/16 round)
                if waves > MAX_WAVES:
                    raise ValueError('Code currently assumes a 1/{} round as the highest round'.format(2**(MAX_WAVES-1)))
                # get the initial seed order for wave 1
                seeds = build_seed_order(waves)
                # determine winners for the rest of the matches that have been completed
                all_seeds = determine_seed_winners(data, seeds, waves)
                # preallocate the text info for the bracket output file
                all_texts = [None] * len(all_seeds)
                # Build the pairings for all the waves
                for (this_wave, this_seed) in enumerate(all_seeds[:-1]):
                    # get a reverse index for the text for the html file
                    ix_rev = waves-this_wave
                    # initialize this text
                    all_texts[ix_rev] = []
                    # display some info
                    if display:
                        print('Wave {0}: {1} {2}'.format(this_wave+1, sex, div))
                    # loop through pairs
                    for this_pair in this_seed:
                        # get the names of the pairing
                        name1 = get_name_by_index(data, this_pair[0]-1) if this_pair[0]-1 < num else 'Bye'
                        name2 = get_name_by_index(data, this_pair[1]-1) if this_pair[1]-1 < num else 'Bye'
                        # get the combined text and save for later
                        text1 = ' #{0} {1}'.format(this_pair[0], name1)
                        text2 = ' #{0} {1}'.format(this_pair[1], name2)
                        all_texts[ix_rev].append(text1)
                        all_texts[ix_rev].append(text2)
                        # display info
                        if display:
                            print(text1 + ' vs. ' + text2)
                # handle the final results
                if display:
                    print('Final Results for {} {}:'.format(sex, div))
                text   = []
                places = ['1st','2nd','3rd','4th']
                for [ix, this_seed] in enumerate(all_seeds[-1]):
                    text.append(' {} Place: {}'.format(places[ix], get_name_by_index(data, this_seed-1) if this_seed-1 < num else ''))
                all_texts[0] = text
                if display:
                    print('\n'.join(text) + '\n')
                # write to html file
                title = 'Individual Brackets ' + sex + ' ' + div
                html  = build_bracket_html_template(waves)
                html  = replace_bracket_tokens(html, title, all_texts)
                brackets.append(html)
                if filename[sex + ' ' + div]:
                    # write to file
                    write_text_file(filename[sex + ' ' + div], html)
    return brackets

#%% Functions - build_seed_order
def build_seed_order(waves):
    r"""
    Creates the order of indices for the given number of waves in the bracket.
    """
    # initialize the list
    x = []
    # start with the number one for anything with one or more waves (zero waves returns an empty list)
    if waves >= 1:
        x.append(1)
    # loop through each successive wave
    for i in range(2, waves+1):
        # find the complement, which depends on the total number of people, starting with 1 vs 2 = 3, 1 vs 4 = 5, 1 vs 8 = 9 etc.
        comp = 2**(i-1)+1
        # loop through the list backwards and insert numbers as appropriate
        for j in range(len(x)-1, -1, -1):
            # for every existing entry, insert the complement into the list
            x.insert(j+1, comp-x[j])
    # build a tuple of seeds
    seeds = [(i,2**waves+1 - i) for i in x]
    # return the finished list of seed orders
    return seeds

#%% Functions - determine_seed_winners
def determine_seed_winners(data, seeds, waves):
    r"""
    Expands the initial round of seed numbers into successive rounds based on the True/False value in the "Win" column.
    """
    # subfunction
    def _winner_of_pair(results, pair, flip=False):
        # find the results of pairing 1 and determine who won
        pairing = (results[pair[0]-1], results[pair[1]-1])
        if   pairing[0] and not pairing[1]:
            if not flip:
                return pair[0]
            else:
                return pair[1]
        elif pairing[1] and not pairing[0]:
            if not flip:
                return pair[1]
            else:
                return pair[0]
        else:
            print(pairing)
            raise ValueError('No one won?')

    # preallocate
    all_seeds = [None] * (waves + 1)
    # set the given first round of seeds
    all_seeds[0] = seeds
    # loop through the waves
    for i in range(0, waves):
        # alias the last round
        last_seed = all_seeds[i]
        # initialize this round
        new_seed  = []
        # loop through half the number of matches
        for j in range(0, len(last_seed)//2):
            # find the last two pairings
            last_pair1 = last_seed[2*j]
            last_pair2 = last_seed[2*j+1]
            # get the key to find the data within the dataframe
            key = eval('COL_{:02}_WIN'.format(2**(waves-1-i))) #pylint: disable=W0123
            # pull out the results
            results = data[key]
            # find the results of pairings and determine who won
            win1 = _winner_of_pair(results,last_pair1)
            win2 = _winner_of_pair(results,last_pair2)
            # append the new pairing
            new_seed.append((win1,win2))
        # keep the new pairings in the master list
        all_seeds[i+1] = new_seed
    # Add the 3rd & 4th place match
    if len(all_seeds) >= 3:
        missing_pair = (list(set(all_seeds[-3][0]) - set(all_seeds[-2][0]))[0], list(set(all_seeds[-3][1]) - set(all_seeds[-2][0]))[0])
        all_seeds[-2].append(missing_pair)
    # give the list of winners
    results = data[COL_01_WIN]
    all_seeds[-1].append(_winner_of_pair(results, all_seeds[-2][0]))            # 1st place
    all_seeds[-1].append(_winner_of_pair(results, all_seeds[-2][0], flip=True)) # 2nd place
    all_seeds[-1].append(_winner_of_pair(results, all_seeds[-2][1]))            # 3rd place
    all_seeds[-1].append(_winner_of_pair(results, all_seeds[-2][1], flip=True)) # 4th place
    # return the results
    return all_seeds

#%% Functions - get_name_by_index
def get_name_by_index(data, index):
    r"""
    Returns the name of the desired person, or an empty string if the person does not exist.
    """
    # check that the index exists, or return an empty string
    if index >= len(data):
        return ''
    else:
        # get the "Lastname, Firstname" for the desired individual
        return '{}, {}'.format(data[COL_LASTNAME][index], data[COL_FIRSTNAME][index])

#%% Functions - validate_brackets
def validate_brackets(data, team='indiv'):
    r"""
    Error checks the bracket results and displays any problems.
    """
    # TODO: Include individual and team and mixed team bracket validation and check the bales for all of them.
    pass

#%% Functions - build_bracket_html_template
def build_bracket_html_template(waves):
    r"""
    Builds a generic HTML template to later use and replace on "Text_A1" etc. strings.
    """
    # get the root location of the module
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # get filename for bracket template
    filename = os.path.join(folder, r'brackets{}.htm'.format(2**waves))
    # read the template
    txt = read_text_file(filename)
    # read the matching style sheet
    css = read_text_file(os.path.join(folder, r'bracket.css'))
    # insert the style sheet into the template
    txt = replace_css(txt, css)
    # return the text based on the template and style sheet
    return txt

#%% Functions - replace_css
def replace_css(txt, css):
    r"""
    Replaces the text/css link in the html file with the contents of the css file directly embedded.
    """
    # build the text token for the old link
    token = r'<link rel="stylesheet" type="text/css" href="bracket.css" />'
    # find the relative link to the style sheet
    old_ix = str.find(txt, token)
    if old_ix == -1:
        raise ValueError('Expected token was not found.')
    # replace the token with the inline style sheet
    new = txt[0:old_ix] + '\n<style type="text/css">\n<!-- \n' + css + '-->\n</style>\n' + txt[old_ix+len(token):]
    # return the updated text
    return new

#%% Functions - replace_bracket_tokens
def replace_bracket_tokens(html, title, names=[]):
    r"""
    Replaces the "Text_A1" tokens with the given text lists
    """
    # replace the title
    html = html.replace('Text_Title_X1', title)
    # loop through the 2D list of names
    for i in range(0, len(names)):
        # inner loop on names
        if names[i] is None:
            break
        for j in range(0, len(names[i])):
            # alias this name
            this_name = names[i][j]
            # find the token letter for the string to replace
            this_letter = chr(ord('A') + i)
            # replace the token "Text_A1" etc string with the desired name
            html = html.replace('Text_' + this_letter + '{}'.format(j+1), this_name, 1)
    # return the updated html text
    return html

#%% Unit test functions
def _main():
    r"""Unit test case."""
    # folder and file locations
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file   = r'Test_Case_1.xlsx'
    filename1 = os.path.join(folder, 'tests', file)
    # import data from excel
    data = import_from_excel(filename1)
    # display some information
    display_info(data)
    # assign to bales
    assign_bales(data)
    # validate bales
    validate_bales(data)
    # assign to brackets
    file_list_indiv = {}
    file_list_indiv['Male Recurve'] = os.path.join(folder, 'output', 'Individual Brackets Male Recurve.htm')
    write_brackets(data, file_list_indiv)
    # validate brackets
    validate_brackets(data)
    # write updated information to output
    filename2 = os.path.join(folder, 'output', file.replace('.xlsx', '.csv'))
    export_to_excel(data, filename2)

def _main2():
    r"""Unit test case 2."""
    waves = 4
    seeds = build_seed_order(waves)
    folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    data  = import_from_excel(os.path.join(folder, 'tests', r'Test_Case_1.xlsx'))
    all_seeds = determine_seed_winners(data, seeds ,waves)
    print(seeds)
    print(all_seeds)

#%% Unit test
if __name__ == '__main__':
    _main()
    #_main2()
