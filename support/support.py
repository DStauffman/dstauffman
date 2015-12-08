# -*- coding: utf-8 -*-
r"""
Support module file for the dstauffman code.  It defines stuff done outside of the model, but
creates stuff used within the model.  Such as porting stuff from ghap to dstauffman automatically.

Notes
-----
#.  Written by David C. Stauffer in April 2015.
"""

# pylint: disable=C0103, C0326

#%% Imports
import os
import ghap as model
import dstauffman as dcs

#%% Functions

#%% Unittest
if __name__ == '__main__':
    files        = ['classes', 'constants', 'plotting', 'utils']
    replacements = [('ghap', 'dstauffman'), ('model', 'dcs'), ('GHAP', 'dstauffman')]
    ghap_folder  = model.get_root_dir()
    dcs_folder   = dcs.get_root_dir()
    stage_folder = dcs.get_data_dir()
    for this_file in files:
        old_path = os.path.join(ghap_folder, this_file + '.py')
        new_path = os.path.join(stage_folder, this_file + '.py')
        # read original file
        text = dcs.read_text_file(old_path)
        # do some replacements
        for (org, new) in replacements:
            text = text.replace(org, new)
        # write out staged file
        dcs.write_text_file(new_path, text)

    # Repeat for tests folder
    ghap_folder = model.get_tests_dir()
    files2      = ['run_all_docstrings', 'run_all_tests', 'test_classes', 'test_constants', \
        'test_plotting', 'test_utils']
    for this_file in files2:
        old_path = os.path.join(ghap_folder, this_file + '.py')
        new_path = os.path.join(stage_folder, this_file + '.py')
        # read original file
        text = dcs.read_text_file(old_path)
        # do some replacements
        for (org, new) in replacements:
            text = text.replace(org, new)
        # write out staged file
        dcs.write_text_file(new_path, text)