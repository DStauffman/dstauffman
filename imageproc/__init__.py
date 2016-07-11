# -*- coding: utf-8 -*-
r"""
The "imageproc" submodule within the "dstauffman" library is a subset of tools used for processing
and manipulating images.

Notes
-----
#.  Updated by David C. Stauffer in July 2016 to be in a submodule not imported by default.
"""

#%% Imports
from .photos import find_missing_nums, find_unexpected_ext, rename_old_picasa_files, \
                        rename_upper_ext, find_long_filenames, batch_resize, \
                        convert_tif_to_jpg, number_files

#%% Unit test
if __name__ == '__main__':
    pass
