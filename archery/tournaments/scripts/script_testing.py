# -*- coding: utf-8 -*-
"""
Main test script for running the code.

Created on Mon Dec 15 13:21:07 2014

@author: DStauffman
"""

#%% Imports
# normal imports
from __future__ import print_function
from __future__ import division
import os
# model imports
import dstauffman as dcs
import dstauffman.archery.tournaments as arch

#%% Variables
# simulate stuff?
SIMULATE = True
# folder and file locations
folder              = arch.get_root_dir()
output_folder       = os.path.realpath(os.path.join(folder, '..', 'output'))
file                = r'Test_Case_4.xlsx'
test_file           = os.path.join(folder, 'tests', file)
if not os.path.isfile(test_file):
    raise ValueError('The specfied test file "{}" was not found.'.format(test_file))
output_file         = os.path.join(output_folder, file)
output_file_indiv   = output_file.replace('.xlsx', '_indiv.csv')
output_file_teams   = output_file.replace('.xlsx', '_teams.csv')
output_file_mixed   = output_file.replace('.xlsx', '_mixed.csv')
# HTML output files
file_registered     = os.path.join(output_folder, 'Registered Archers.htm')
file_reg_bales      = os.path.join(output_folder, 'Individual Bale Assignments.htm')
file_indiv_res      = os.path.join(output_folder, 'Individual Results.htm')
# bracket lists
# ex: Individual Brackets Female Bare Bow.htm
file_list_indiv = {}
file_list_teams = {}
file_list_mixed = {}
for div in arch.DIVISIONS:
    for sex in arch.GENDERS:
        file_list_indiv[sex + ' ' + div] = os.path.join(output_folder, 'Individual Brackets ' + sex + ' ' + div + '.htm')
        file_list_teams[sex + ' ' + div] = os.path.join(output_folder, 'Team Brackets ' + sex + ' ' + div + '.htm')
        if sex == arch.GENDERS[0]:
            file_list_mixed[div]         = os.path.join(output_folder, 'Mixed Team Brackets ' + div + '.htm')

#%% Output folder
# create the output folder if it doesn't already exist
if not os.path.isdir(output_folder):
    dcs.setup_dir(output_folder)

#%% Process data
# import data from excel
data_indiv = arch.import_from_excel(test_file, sheet=arch.SHEET_NAME_INDIV)
data_teams = arch.import_from_excel(test_file, sheet=arch.SHEET_NAME_TEAMS)
data_mixed = arch.import_from_excel(test_file, sheet=arch.SHEET_NAME_MIXED)

# display some information
arch.display_info(data_indiv)

# write out list of registered archers
arch.write_registered_archers(data_indiv, filename=file_registered, show_bales=False)

# assign to bales
data_indiv = arch.assign_bales(data_indiv)

# validate bales
arch.validate_bales(data_indiv)

# write out list of registered archers with bale assignments now included
arch.write_registered_archers(data_indiv, filename=file_reg_bales, show_bales=True)

# enter scores (simulated)
if SIMULATE:
    arch.simulate_individual_scores(data_indiv)
else:
    # read by in from updated excel file
    pass

# determine individual seeds
data_indiv = arch.update_indiv(data_indiv)

# display final individual rankings
arch.write_indiv_results(data_indiv, filename=file_indiv_res)

# update information for teams based on individual results
data_teams = arch.update_teams(data_indiv, data_teams)
data_mixed = arch.update_mixed(data_indiv, data_mixed)

#%% Brackets
# assign bracket bales based on seeds/number of archers in each division and field layout
# TODO: write this

## validate all brackets & bales
#arch.validate_brackets(data_indiv, team='indiv')
#arch.validate_brackets(data_teams, team='teams')
#arch.validate_brackets(data_mixed, team='mixed')
#
## write initial brackets for individual competition
#arch.write_brackets(data_indiv, filename=file_list_indiv, team='indiv')
#
## enter individual bracket scores (simulated for each round), start with 1/16 round
#if SIMULATE:
#    arch.simulate_bracket_scores(data_indiv, round_='1/16')
#arch.write_brackets(data_indiv, filename=file_list_indiv, team='indiv')
## update brackets 1/8 round and rewrite brackets
#if SIMULATE:
#    arch.simulate_bracket_scores(data_indiv, round_='1/8')
#arch.write_brackets(data_indiv, filename=file_list_indiv, team='indiv')
## update brackets 1/4 (quarter-final) round and rewrite brackets
#if SIMULATE:
#    arch.simulate_bracket_scores(data_indiv, round_='1/4')
#arch.write_brackets(data_indiv, filename=file_list_indiv, team='indiv')
## update brackets 1/2 (semi-final) round and rewrite brackets
#if SIMULATE:
#    arch.simulate_bracket_scores(data_indiv, round_='1/2')
#arch.write_brackets(data_indiv, filename=file_list_indiv, team='indiv')
## update brackets 1 (final) round and produce final results brackets
#if SIMULATE:
#    arch.simulate_bracket_scores(data_indiv, round_='1/1')
#arch.write_brackets(data_indiv, filename=file_list_indiv, team='indiv')
#
## enter mixed team bracket scores (simulated for each round)
##TODO: repeat again
## write initial brackets for team competition
##arch.write_brackets(data_teams, filename=file_list_teams, team='teams')
#
## enter team bracket scores (simulated for each round)
##TODO: repeat again
## write initial brackets for mixed team competition
##arch.write_brackets(data_mixed, filename=file_list_mixed, team='mixed')

# write updated information to output (CSV)
arch.export_to_excel(data_indiv, output_file_indiv)
arch.export_to_excel(data_teams, output_file_teams)
arch.export_to_excel(data_mixed, output_file_mixed)

# write updated information to output (Excel) (Must write all sheets at once?)
arch.export_to_excel([data_indiv, data_teams, data_mixed], output_file, \
    sheet=[arch.SHEET_NAME_INDIV, arch.SHEET_NAME_TEAMS, arch.SHEET_NAME_MIXED])
