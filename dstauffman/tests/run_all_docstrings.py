# -*- coding: utf-8 -*-
r"""
Test file to execute all the docstrings within the dstauffman code.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import os

import dstauffman as dcs

#%% Execution
if __name__ == '__main__':
    files = dcs.list_python_files(dcs.get_root_dir())
    files.extend(dcs.list_python_files(os.path.join(dcs.get_root_dir(), 'commands')))
    dcs.run_docstrings(files, verbose=False)
