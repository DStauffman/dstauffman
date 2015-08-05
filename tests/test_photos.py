# -*- coding: utf-8 -*-
r"""
Test file for the `photos` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import os
import unittest
import dstauffman as dcs

#%% Classes for testing
# find_missing_nums
class Test_find_missing_nums(unittest.TestCase):
    r"""
    Tests the find_missing_nums function with the following cases:
        Nominal Usage
        Folder exclusions
        Ignore digit errors
        Nothing missing
    """
    @classmethod
    def setUpClass(cls):
        cls.folder = dcs.get_tests_dir()
        cls.folder_exclusions = [os.path.join(cls.folder, 'coverage_html_report')]
        file1 = os.path.join(cls.folder, 'temp image 01.jpg')
        file2 = os.path.join(cls.folder, 'temp image 02.jpg')
        file3 = os.path.join(cls.folder, 'temp image 04.jpg')
        file4 = os.path.join(cls.folder, 'temp image 006.jpg')
        file5 = os.path.join(cls.folder, 'temp something else 1.jpg')
        file6 = os.path.join(cls.folder, 'Picasa.ini')
        file7 = os.path.join(cls.folder, 'temp image 10 10.jpg')
        file8 = os.path.join(cls.folder_exclusions[0], 'temp image 01.jpg')
        cls.files = [file1, file2, file3, file4, file5, file6, file7, file8]
        for this_file in cls.files:
            dcs.write_text_file(this_file, '')

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.find_missing_nums(self.folder)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('Old Picasa file: "'))
        self.assertTrue(lines[1].startswith('Weird numbering: "'))
        self.assertTrue(lines[2].startswith('Missing: "'))
        self.assertTrue(lines[2].endswith('": {3, 5}'))
        self.assertTrue(lines[3].startswith('Inconsistent digits: "'))
        self.assertTrue(lines[4].startswith('No number found: "'))

    def test_folder_exclusions(self):
        with dcs.capture_output() as (out, _):
            dcs.find_missing_nums(self.folder, folder_exclusions=self.folder_exclusions)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('Old Picasa file: "'))
        self.assertTrue(lines[1].startswith('Weird numbering: "'))
        self.assertTrue(lines[2].startswith('Missing: "'))
        self.assertTrue(lines[2].endswith('": {3, 5}'))
        self.assertTrue(lines[3].startswith('Inconsistent digits: "'))
        self.assertTrue(len(lines) < 5)

    def test_ignore_digits(self):
        with dcs.capture_output() as (out, _):
            dcs.find_missing_nums(self.folder, digit_check=False)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('Old Picasa file: "'))
        self.assertTrue(lines[1].startswith('Weird numbering: "'))
        self.assertTrue(lines[2].startswith('Missing: "'))
        self.assertTrue(lines[2].endswith('": {3, 5}'))
        self.assertFalse(lines[3].startswith('Inconsistent digits: "'))

    def test_nothing_missing(self):
        with dcs.capture_output() as (out, _):
            dcs.find_missing_nums(self.folder_exclusions[0])
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('No number found: "'))

    @classmethod
    def tearDownClass(cls):
        for this_file in cls.files:
            if os.path.isfile(this_file):
                os.remove(this_file)

# find_unexpected_ext
class Test_find_unexpected_ext(unittest.TestCase):
    r"""
    Tests the find_unexpected_ext function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.find_unexpected_ext(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('Finding any unexpected file extensions...\n Unexpected: "'))
        self.assertTrue(output.endswith('"\nDone.'))

# rename_old_picasa_files
class Test_rename_old_picasa_files(unittest.TestCase):
    r"""
    Tests the rename_old_picasa_files function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder   = dcs.get_tests_dir()
        self.file_old = os.path.join(self.folder, 'Picasa.ini')
        self.file_new = os.path.join(self.folder, '.picasa.ini')
        dcs.write_text_file(self.file_old, '')

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.rename_old_picasa_files(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('Renaming: "'))

    def tearDown(self):
        if os.path.isfile(self.file_new):
            os.remove(self.file_new)

# rename_upper_ext
class Test_rename_upper_ext(unittest.TestCase):
    r"""
    Tests the rename_upper_ext function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder   = dcs.get_tests_dir()
        self.file_old = os.path.join(self.folder, 'temp image 01.JPG')
        self.file_new = os.path.join(self.folder, 'temp image 01.jpg')
        dcs.write_text_file(self.file_old, '')

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.rename_upper_ext(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('Searching for file extensions to rename...\n Renaming: "'))
        self.assertTrue(output.endswith('\nDone.'))

    def tearDown(self):
        if os.path.isfile(self.file_new):
            os.remove(self.file_new)

# find_long_filenames
class Test_find_long_filenames(unittest.TestCase):
    r"""
    Tests the find_long_filenames function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.find_long_filenames(self.folder)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[-4].startswith(' max name = '))
        self.assertTrue(lines[-3].startswith(' max root = '))
        self.assertTrue(lines[-2].startswith(' max full = '))
        self.assertEqual(lines[-1], 'Done.')

# batch_resize
class Test_batch_resize(unittest.TestCase):
    r"""
    Tests the batch_resize function with the following cases:
        Nominal Usage
        Bad inputs
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()
        self.max_width  = 2048
        self.max_height = 2048

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.batch_resize(self.folder, self.max_width, self.max_height)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))

    def test_bad_inputs(self):
        with dcs.capture_output() as (out, _):
            dcs.batch_resize(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertEqual(output, 'Invalid arguments. You must overwrite all three options')

# convert_tif_to_jpg
class Test_convert_tif_to_jpg(unittest.TestCase):
    r"""
    Tests the convert_tif_to_jpg function with the following cases:
        Nominal Usage
    """
    def setUp(self):
        self.folder = dcs.get_tests_dir()

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dcs.convert_tif_to_jpg(self.folder)
        output = out.getvalue().strip()
        out.close()
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
