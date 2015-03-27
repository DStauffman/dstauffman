# -*- coding: utf-8 -*-
r"""
Photos module file for the "dstauffman" library.  It contains a collection of commands that are
useful for maintaining photo galleries.

Notes
-----
#.  Written by David C. Stauffer in December 2013.
"""

# pylint: disable=C0326

#%% Imports
from __future__ import print_function
from __future__ import division
import os
from PIL import Image
import unittest

#%% Local Constants
ALLOWABLE_EXTENSIONS = frozenset(['.jpg', '.ini', '.png', '.gif'])
PROCESS_EXTENSIONS   = frozenset(['.jpg', '.png', '.gif'])
INT_TOKEN            = -1

#%% Functions - find_missing_nums
def find_missing_nums(folder, old_picasa=True, digit_check=True, \
        process_extensions=PROCESS_EXTENSIONS, folder_exclusions=None):
    r"""
    Finds the missing numbers in a file sequence:
    Photos 001.jpg
    Photos 002.jpg
    Photos 004.jpg

    Finds missing Photos 003.jpg

    Parameters
    ----------
    folder : str
        Name of folder to process
    old_picasa : bool, optional
        Determines if printing a warning about old .picasa.ini files, default is True
    digit_check : bool, optional
        Determines if checking for a consistent number of digits in the numbering
        01, 02, 03 versus 001, 02, 003, etc. Default is True

    """
    for (root, _, files) in os.walk(folder):
        name_dict = dict()
        nums_list = list()
        digs_list = list()
        counter   = 0
        for name in files:
            (file_name, file_ext) = os.path.splitext(name)
            if old_picasa and file_ext == '.ini' and file_name != '.picasa':
                print('Old Picasa file: "{}"'.format(os.path.join(root, name)))
            if file_ext not in process_extensions:
                continue
            excluded = False
            for excl in folder_exclusions:
                if excl == root[0:len(excl)]:
                    excluded = True
            if excluded:
                continue
            parts = file_name.split()
            text = r' '.join([s for s in parts if not s.isdigit()])
            strs = [s for s in parts if s.isdigit()]
            nums = [int(s) for s in strs]
            if len(nums) > 1:
                print('Weird numbering: "{}"'.format(os.path.join(root, name)))
                break
            elif len(nums) == 0:
                print('No number found: "{}"'.format(os.path.join(root, name)))
                continue
            nums = nums[0]
            digs = len(strs[0])
            if text not in name_dict:
                name_dict[text] = counter
                counter += 1
            pos = name_dict[text]
            if nums_list:
                try:
                    nums_list[pos].append(nums)
                    digs_list[pos].append(digs)
                except: # pylint: disable=W0702
                    nums_list.append([nums])
                    digs_list.append([digs])
            else:
                nums_list.append([nums])
                digs_list.append([digs])
        for nams in name_dict:
            nums = nums_list[name_dict[nams]]
            digs = digs_list[name_dict[nams]]
            missing = set(nums) ^ set(range(1, max(nums)+1))
            digits  = [nums[i] for i in range(0, len(digs)) if digs[i] != max(digs)]
            if missing:
                print('Missing: "{}": '.format(os.path.join(root, nams)), end='')
                print(missing)
            if digit_check and digits:
                print('Inconsistent digits: "{}": '.format(os.path.join(root, nams)), end='')
                print(set(digits))

#%% Functions - find_unexpected_ext
def find_unexpected_ext(folder, allowable_extensions=ALLOWABLE_EXTENSIONS):
    r"""
    Lists any files in the folder that don't have the expected file extensions.
    """
    # print status
    print('Finding any unexpected file extensions...')
    # walk through folder
    for (root, _, files) in os.walk(folder):
        # go through files
        for name in files:
            # check for allowable extensions
            (_, file_ext) = os.path.splitext(name)
            if not file_ext in allowable_extensions:
                # print files not in allowable extension list
                print(' Unexpected: "{}"'.format(os.path.join(root, name)))
    print('Done.')

#%% Functions - rename_old_picasa_files
def rename_old_picasa_files(folder):
    r"""
    Renames the old ".picasa.ini" to the newer "Picasa.ini" standard.
    """
    # definitions
    old_name = r'Picasa.ini'
    new_name = r'.picasa.ini'
    # walk through folder
    for (root, _, files) in os.walk(folder):
        # go through files
        for name in files:
            # find any that match the old name
            if name == old_name:
                # get fullpath names for the old and new standards
                old_path = os.path.join(root, old_name)
                new_path = os.path.join(root, new_name)
                # print status of the rename
                print('Renaming: "{}" to "{}"'.format(old_path, new_path))
                try:
                    # do rename
                    os.rename(old_path, new_path)
                except: # pylint: disable=W0702
                    # print any problems and then continue
                    print('Unable to rename: "{}"'.format(old_path))
                    continue

#%% Functions - rename_upper_ext
def rename_upper_ext(folder, allowable_extensions=ALLOWABLE_EXTENSIONS):
    r"""
    Renames any expected file types to have all lowercase file extensions.

    Common use is to rename the *.JPG extensions from my camera to *.jpg
    """
    # update status
    print('Searching for file extensions to rename...')
    # walk through folder
    for (root, _, files) in os.walk(folder):
        # go through files
        for name in files:
            # split the filename and extension
            (file_name, file_ext) = os.path.splitext(name)
            # check that the lowercase version is in the allowable set, but the given one isn't
            # if true, then this means the extension has non lowercase letters and needs to be fixed
            if not file_ext in allowable_extensions and file_ext.lower() in allowable_extensions:
                # get the old name
                old_path = os.path.join(root, name)
                # get the new fixed lowercase name
                new_path = os.path.join(root, file_name + file_ext.lower())
                # print the status for the rename command
                print(' Renaming: "{}" to "{}"'.format(old_path, new_path))
                try:
                    # do rename
                    os.rename(old_path, new_path)
                except: # pylint: disable=W0702
                    # print any exceptions, but then continue
                    print(' Unable to rename: "{}"'.format(old_path))
                    continue
    print('Done.')

#%% Functions - find_long_filenames
def find_long_filenames(folder):
    r"""
    Finds any files with really long filenames.
    """
    print('Finding long filenames...')
    max_name = 0
    max_root = 0
    max_full = 0
    len_root = len('\\'.join(folder.split('\\')[:-1]))
    for (root, _, files) in os.walk(folder):
        for name in files:
            (file_name, file_ext) = os.path.splitext(name)
            if ''.join(file_name.split()) == '' or (file_ext == '' and file_name[0] == '.'):
                print(os.path.join(root, name))
            temp = len(name)
            if temp > max_name:
                max_name = temp
            #temp_name = ''.join(name.split())
            #if len(temp_name) < 5:
            #    print(os.path.join(root,name))
            if temp > 106:
                print(os.path.join(root, name))
            temp = len(root) - len_root
            if temp > max_root:
                max_root = temp
            if temp > 106:
                pass# print(root)
            temp = len(name) + len(root) - len_root
            if temp > max_full:
                max_full = temp
            if temp > 212:
                print(os.path.join(root, name))
            if ';' in name:
                print(os.path.join(root, name))
    print(' max name = {}'.format(max_name))
    print(' max root = {}'.format(max_root))
    print(' max full = {}'.format(max_full))
    print('Done.')

#%% Functions - batch_resize
def batch_resize(folder, max_width=INT_TOKEN, max_height=INT_TOKEN, \
        process_extensions=PROCESS_EXTENSIONS):
    r"""
    Resize the specified folder of images to the given max width and height.
    """
    # We have to make sure that all of the arguments were passed.
    if max_width == INT_TOKEN or max_height == INT_TOKEN or not folder:
        print('Invalid arguments. You must overwrite all three options')
        # If an argument is missing then return without doing anything
        return

    # Iterate through every image given in the folder argument and resize it.
    for image in os.listdir(folder):
        # check if valid image file
        if os.path.isdir(image):
            continue
        elif image[-4:] not in process_extensions:
            print('Skipping file: "{}"'.format(image))
            continue

        # Update status
        print('Resizing image: "{}"'.format(image))

        # Open the image file.
        img = Image.open(os.path.join(folder, image))

        # Get current properties
        cur_width    = img.size[0]
        cur_height   = img.size[1]
        aspect_ratio = cur_width / cur_height

        # Calucalte desired size
        cal_width  = int(max_height * aspect_ratio)
        cal_height = int(max_width / aspect_ratio)

        # set new size
        if cal_height < max_height:
            new_width  = max_width
            new_height = cal_height
        elif cal_width < max_width:
            new_width  = cal_width
            new_height = max_height
        else:
            # need to shrink further
            scale1 = cal_width/max_width
            scale2 = cal_height/max_height
            if scale1 >= scale2:
                new_width  = cal_width  / scale1
                new_height = max_height / scale1
            else:
                new_width  = max_width  / scale2
                new_height = cal_height / scale2


        # Resize it.
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # Save it back to disk.
        img.save(os.path.join(folder, 'resized', image))

    print('Batch processing complete.')

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_photos', exit=False)
