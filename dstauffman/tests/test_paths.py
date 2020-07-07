r"""
Test file for the `paths` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in February 2019.
"""

#%% Imports
import inspect
import os
import unittest

import dstauffman as dcs

#%% get_root_dir
class Test_get_root_dir(unittest.TestCase):
    r"""
    Tests the get_root_dir function with these cases:
        call the function
    """
    def test_function(self):
        filepath      = inspect.getfile(dcs.get_root_dir)
        expected_root = os.path.split(filepath)[0]
        folder = dcs.get_root_dir()
        self.assertEqual(folder, expected_root)
        self.assertTrue(os.path.isdir(folder))

#%% get_tests_dir
class Test_get_tests_dir(unittest.TestCase):
    r"""
    Tests the get_tests_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_tests_dir()
        self.assertEqual(folder, os.path.join(dcs.get_root_dir(), 'tests'))

#%% get_data_dir
class Test_get_data_dir(unittest.TestCase):
    r"""
    Tests the get_data_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_data_dir()
        self.assertEqual(folder, os.path.abspath(os.path.join(dcs.get_root_dir(), '..', 'data')))

#%% get_images_dir
class Test_get_images_dir(unittest.TestCase):
    r"""
    Tests the get_images_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_images_dir()
        self.assertEqual(folder, os.path.abspath(os.path.join(dcs.get_root_dir(), '..', 'images')))

#%% get_output_dir
class Test_get_output_dir(unittest.TestCase):
    r"""
    Tests the get_output_dir function with these cases:
        call the function
    """
    def test_function(self):
        folder = dcs.get_output_dir()
        self.assertEqual(folder, os.path.abspath(os.path.join(dcs.get_root_dir(), '..', 'results')))

#%% Unit test execution
if __name__ == '__main__':
    unittest.main(exit=False)
