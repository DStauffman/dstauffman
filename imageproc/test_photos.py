# -*- coding: utf-8 -*-
r"""
Test file for the `photos` module of the "dstauffman" library.  It is intented to contain test
cases to demonstrate functionaliy and correct outcomes for all the functions within the module.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import numpy as np
import os
from PIL import Image
import unittest
import dstauffman as dcs
import dstauffman.imageproc as dip

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
        cls.folder_exclusions = [os.path.join(cls.folder, 'temp_dir'), \
            os.path.join(cls.folder, 'coverage_html_report')]
        file1 = os.path.join(cls.folder, 'temp image 01.jpg')
        file2 = os.path.join(cls.folder, 'temp image 02.jpg')
        file3 = os.path.join(cls.folder, 'temp image 04.jpg')
        file4 = os.path.join(cls.folder, 'temp image 006.jpg')
        file5 = os.path.join(cls.folder, 'temp something else 1.jpg')
        file6 = os.path.join(cls.folder, 'Picasa.ini')
        file7 = os.path.join(cls.folder, 'temp image 10 10.jpg')
        if not os.path.isdir(cls.folder_exclusions[0]):
            os.mkdir(cls.folder_exclusions[0])
        file8 = os.path.join(cls.folder_exclusions[0], 'temp image 01.jpg')
        file9 = os.path.join(cls.folder_exclusions[0], 'temp longimagename.jpg')
        cls.files = [file1, file2, file3, file4, file5, file6, file7, file8, file9]
        for this_file in cls.files:
            dcs.write_text_file(this_file, '')

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dip.find_missing_nums(self.folder)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('Old Picasa file: "'))
        self.assertTrue(lines[1].startswith('Weird numbering: "'))
        self.assertTrue(lines[2].startswith('Missing: "'))
        self.assertTrue(lines[2].endswith('": {3, 5}') or lines[2].endswith('": set([3, 5])'))
        self.assertTrue(lines[3].startswith('Inconsistent digits: "'))
        self.assertTrue(lines[4].startswith('No number found: "'))

    def test_folder_exclusions(self):
        with dcs.capture_output() as (out, _):
            dip.find_missing_nums(self.folder, folder_exclusions=self.folder_exclusions)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('Old Picasa file: "'))
        self.assertTrue(lines[1].startswith('Weird numbering: "'))
        self.assertTrue(lines[2].startswith('Missing: "'))
        self.assertTrue(lines[2].endswith('": {3, 5}') or lines[2].endswith('": set([3, 5])'))
        self.assertTrue(lines[3].startswith('Inconsistent digits: "'))
        self.assertTrue(len(lines) < 5)

    def test_ignore_digits(self):
        with dcs.capture_output() as (out, _):
            dip.find_missing_nums(self.folder, digit_check=False)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('Old Picasa file: "'))
        self.assertTrue(lines[1].startswith('Weird numbering: "'))
        self.assertTrue(lines[2].startswith('Missing: "'))
        self.assertTrue(lines[2].endswith('": {3, 5}') or lines[2].endswith('": set([3, 5])'))
        self.assertFalse(lines[3].startswith('Inconsistent digits: "'))

    def test_nothing_missing(self):
        with dcs.capture_output() as (out, _):
            dip.find_missing_nums(self.folder_exclusions[0])
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(lines[0].startswith('No number found: "'))

    @classmethod
    def tearDownClass(cls):
        for this_file in cls.files:
            if os.path.isfile(this_file):
                os.remove(this_file)
        if os.path.isdir(cls.folder_exclusions[0]):
            os.rmdir(cls.folder_exclusions[0])

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
            dip.find_unexpected_ext(self.folder)
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
            dip.rename_old_picasa_files(self.folder)
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
            dip.rename_upper_ext(self.folder)
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
            dip.find_long_filenames(self.folder)
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
        No images
        No upscale
        With upscale
    """
    @classmethod
    def setUpClass(cls):
        cls.source = os.path.join(dcs.get_images_dir(), 'close_all.png')
        cls.name1  = 'image1.jpg'
        cls.name2  = 'image2.jpg'
        cls.name3  = 'image3.jpg'
        cls.name4  = 'image4.jpg'
        cls.name5  = 'image5.jpg'
        cls.name6  = 'image6.jpeg'
        cls.folder = os.path.join(dcs.get_tests_dir(), 'images')
        cls.extra  = os.path.join(cls.folder, 'extra')
        cls.size1  = 128
        cls.size2  = 96
        cls.size3  = 32
        cls.size4  = 128*2
        cls.size5  = 96*2
        cls.size6  = 4
        cls.output = os.path.join(cls.folder, 'resized')
        with dcs.capture_output():
            dcs.setup_dir(cls.folder)
            dcs.setup_dir(cls.extra)
        with open(cls.source, 'rb') as file:
            img = Image.open(file)
            img.load()
        new_img = img.resize((cls.size1, cls.size1), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name1))
        new_img.save(os.path.join(cls.folder, cls.name6))
        new_img = img.resize((cls.size2, cls.size1), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name2))
        new_img = img.resize((cls.size1, cls.size2), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name3))
        new_img = img.resize((cls.size1, cls.size6), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name4))
        new_img = img.resize((cls.size6, cls.size1), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name5))
        img.close()
        new_img.close()

    def test_resize(self):
        with dcs.capture_output() as (out, _):
            dip.batch_resize(self.folder, self.size3, self.size3)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            if this_line.startswith(' Resizing image  : "'):
                break
        else:
            self.assertTrue(False,'No images were resized.')
        for this_line in lines:
            if this_line.startswith(' Skipping file   : "{}"'.format(self.name6)):
                break
        else:
            self.assertTrue(False,'File "{}" was not skipped.'.format(self.name6))
        with open(os.path.join(self.output, self.name1), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size3, self.size3])
        fact = self.size2 / self.size1
        with open(os.path.join(self.output, self.name2), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [int(self.size3*fact), self.size3])
        with open(os.path.join(self.output, self.name3), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size3, int(self.size3*fact)])
        with open(os.path.join(self.output, self.name4), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size3, self.size6//4])
        with open(os.path.join(self.output, self.name5), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size6//4, self.size3])
        img.close()

    def test_no_images(self):
        with dcs.capture_output() as (out, _):
            dip.batch_resize(dcs.get_data_dir())
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            self.assertFalse(this_line.startswith(' Resizing image'))

    def test_no_upscale(self):
        with dcs.capture_output() as (out, _):
            dip.batch_resize(self.folder, max_width=self.size4, max_height=self.size4, enlarge=False)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            if this_line.startswith(' Not enlarging'):
                break
        else:
            self.assertTrue(False,'No images were resized.')
        for this_line in lines:
            if this_line.startswith(' Skipping file   : "{}"'.format(self.name6)):
                break
        else:
            self.assertTrue(False,'File "{}" was not skipped.'.format(self.name6))
        with open(os.path.join(self.output, self.name1), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size1, self.size1])
        with open(os.path.join(self.output, self.name2), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size2, self.size1])
        with open(os.path.join(self.output, self.name3), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size1, self.size2])
        with open(os.path.join(self.output, self.name4), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size1, self.size6])
        with open(os.path.join(self.output, self.name5), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size6, self.size1])
        img.close()

    def test_upscale(self):
        with dcs.capture_output() as (out, _):
            dip.batch_resize(self.folder, max_width=self.size4, max_height=self.size4, enlarge=True)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            if this_line.startswith(' Resizing image'):
                break
        else:
            self.assertTrue(False,'No images were resized.')
        for this_line in lines:
            if this_line.startswith(' Skipping file   : "{}"'.format(self.name6)):
                break
        else:
            self.assertTrue(False,'File "{}" was not skipped.'.format(self.name6))
        with open(os.path.join(self.output, self.name1), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size4, self.size4])
        fact = self.size2 / self.size1
        with open(os.path.join(self.output, self.name2), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size4*fact, self.size4])
        with open(os.path.join(self.output, self.name3), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size4, self.size4*fact])
        with open(os.path.join(self.output, self.name4), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size4, self.size6*2])
        with open(os.path.join(self.output, self.name5), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size6*2, self.size4])
        img.close()

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.extra):
            os.rmdir(cls.extra)
        if os.path.isdir(cls.output):
            with dcs.capture_output():
                dcs.setup_dir(cls.output)
            os.rmdir(cls.output)
        if os.path.isdir(cls.folder):
            with dcs.capture_output():
                dcs.setup_dir(cls.folder)
            os.rmdir(cls.folder)

# convert_tif_to_jpg
class Test_convert_tif_to_jpg(unittest.TestCase):
    r"""
    Tests the convert_tif_to_jpg function with the following cases:
        Nominal Usage
        No images
        No upscale
        With upscale
    """
    @classmethod
    def setUpClass(cls):
        cls.source = os.path.join(dcs.get_images_dir(), 'close_all.png')
        cls.name1  = 'image1.tif'
        cls.name2  = 'image2.tif'
        cls.name3  = 'image3.tif'
        cls.name4  = 'image4.tif'
        cls.name5  = 'image5.tif'
        cls.name6  = 'image6.jpeg'
        cls.folder = os.path.join(dcs.get_tests_dir(), 'images')
        cls.extra  = os.path.join(cls.folder, 'extra')
        cls.size1  = 128
        cls.size2  = 96
        cls.size3  = 32
        cls.size4  = 128*2
        cls.size5  = 96*2
        cls.size6  = 4
        with dcs.capture_output():
            dcs.setup_dir(cls.folder)
            dcs.setup_dir(cls.extra)
        with open(cls.source, 'rb') as file:
            img = Image.open(file)
            img.load()
        new_img = img.resize((cls.size1, cls.size1), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name1))
        new_img.save(os.path.join(cls.folder, cls.name6))
        new_img = img.resize((cls.size2, cls.size1), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name2))
        new_img = img.resize((cls.size1, cls.size2), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name3))
        new_img = img.resize((cls.size1, cls.size6), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name4))
        new_img = img.resize((cls.size6, cls.size1), Image.ANTIALIAS)
        new_img.save(os.path.join(cls.folder, cls.name5))
        img.close()
        new_img.close()

    def test_resize(self):
        with dcs.capture_output() as (out, _):
            dip.convert_tif_to_jpg(self.folder, self.size3, self.size3, replace=True)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            if this_line.startswith(' Saving image    : "'):
                break
        else:
            self.assertTrue(False,'No images were saved.')
        for this_line in lines:
            if this_line.startswith(' Skipping file   : "{}"'.format(self.name6)):
                break
        else:
            self.assertTrue(False,'File "{}" was not skipped.'.format(self.name6))
        with open(os.path.join(self.folder, self.name1.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size3, self.size3])
        fact = self.size2 / self.size1
        with open(os.path.join(self.folder, self.name2.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [int(self.size3*fact), self.size3])
        with open(os.path.join(self.folder, self.name3.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size3, int(self.size3*fact)])
        with open(os.path.join(self.folder, self.name4.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size3, self.size6//4])
        with open(os.path.join(self.folder, self.name5.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size6//4, self.size3])
        img.close()

    def test_no_images(self):
        with dcs.capture_output() as (out, _):
            dip.convert_tif_to_jpg(dcs.get_data_dir())
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            self.assertFalse(this_line.startswith(' Resizing image'))

    def test_no_upscale(self):
        with dcs.capture_output() as (out, _):
            dip.convert_tif_to_jpg(self.folder, max_width=self.size4, max_height=self.size4, enlarge=False, replace=True)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            if this_line.startswith(' Saving (not enlarging'):
                break
        else:
            self.assertTrue(False,'No images were resized.')
        for this_line in lines:
            if this_line.startswith(' Skipping file   : "{}"'.format(self.name6)):
                break
        else:
            self.assertTrue(False,'File "{}" was not skipped.'.format(self.name6))
        with open(os.path.join(self.folder, self.name1.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size1, self.size1])
        with open(os.path.join(self.folder, self.name2.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size2, self.size1])
        with open(os.path.join(self.folder, self.name3.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size1, self.size2])
        with open(os.path.join(self.folder, self.name4.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size1, self.size6])
        with open(os.path.join(self.folder, self.name5.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size6, self.size1])
        img.close()

    def test_upscale(self):
        with dcs.capture_output() as (out, _):
            dip.convert_tif_to_jpg(self.folder, max_width=self.size4, max_height=self.size4, enlarge=True, replace=True)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            if this_line.startswith(' Saving image    : "'):
                break
        else:
            self.assertTrue(False,'No images were saved.')
        for this_line in lines:
            if this_line.startswith(' Skipping file   : "{}"'.format(self.name6)):
                break
        else:
            self.assertTrue(False,'File "{}" was not skipped.'.format(self.name6))
        with open(os.path.join(self.folder, self.name1.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size4, self.size4])
        fact = self.size2 / self.size1
        with open(os.path.join(self.folder, self.name2.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [int(self.size4*fact), self.size4])
        with open(os.path.join(self.folder, self.name3.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size4, int(self.size4*fact)])
        with open(os.path.join(self.folder, self.name4.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size4, self.size6*2])
        with open(os.path.join(self.folder, self.name5.replace('.tif','.jpg')), 'rb') as file:
            img = Image.open(file)
            img.load()
        np.testing.assert_array_equal(img.size, [self.size6*2, self.size4])
        img.close()

    def test_noreplace(self):
        with dcs.capture_output() as (out, _):
            dip.convert_tif_to_jpg(self.folder, self.size3, self.size3, replace=False)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        self.assertTrue(output.endswith('Batch processing complete.'))
        for this_line in lines:
            if this_line.startswith(' Skipping due to pre-existing jpg file: "'):
                break
        else:
            self.assertTrue(False,'No images were skipped due to pre-existing ones.')
        for this_line in lines:
            if this_line.startswith(' Skipping file   : "{}"'.format(self.name6)):
                break
        else:
            self.assertTrue(False,'File "{}" was not skipped.'.format(self.name6))

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.extra):
            os.rmdir(cls.extra)
        if os.path.isdir(cls.folder):
            with dcs.capture_output():
                dcs.setup_dir(cls.folder)
            os.rmdir(cls.folder)


#%% number_files
class Test_number_files(unittest.TestCase):
    r""" Tests the number_files function with the following cases:
        Nominal
    """
    def setUp(self):
        self.folder    = dcs.get_tests_dir()
        self.file_old1 = os.path.join(self.folder, 'temp image A.jpg')
        self.file_old2 = os.path.join(self.folder, 'temp xtra image B.jpg')
        self.file_new1 = os.path.join(self.folder, 'Photo 01.jpg')
        self.file_new2 = os.path.join(self.folder, 'Photo 02.jpg')
        self.prefix    = 'Photo '
        self.start     = 1
        self.digits    = 2
        dcs.write_text_file(self.file_old1, '')
        dcs.write_text_file(self.file_old2, '')

    def test_nominal(self):
        with dcs.capture_output() as (out, _):
            dip.number_files(self.folder, self.prefix, self.start, self.digits)
        output = out.getvalue().strip()
        out.close()
        lines = output.split('\n')
        self.assertTrue(output.startswith('Processing folder: "'))
        for this_line in lines:
            if this_line == ' Renaming : "temp image A.jpg" to "Photo 01.jpg"':
                break
        else:
            self.assertTrue(False, 'File "{}" was not renamed.'.format(self.file_old2))
        for this_line in lines:
            if this_line == ' Renaming : "temp xtra image B.jpg" to "Photo 02.jpg"':
                break
        else:
            self.assertTrue(False, 'File "{}" was not renamed.'.format(self.file_old2))
        self.assertTrue(output.endswith('Batch processing complete.'))

    def tearDown(self):
        files = [self.file_new1, self.file_new2]
        for this_file in files:
            if os.path.isfile(this_file):
                os.remove(this_file)

#%% Unit test execution
if __name__ == '__main__':
    # run the tests
    unittest.main(exit=False)
