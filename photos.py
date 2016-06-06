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
import doctest
import os
from PIL import Image
import shutil
import unittest
from dstauffman.utils     import setup_dir

#%% Local Constants
ALLOWABLE_EXTENSIONS = frozenset(['.jpg', '.ini', '.png', '.gif'])
PROCESS_EXTENSIONS   = frozenset(['.jpg', '.png', '.gif'])

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

    Notes
    -----
    #.  Written by David C. Stauffer in December 2013.

    Examples
    --------

    >>> from dstauffman import get_data_dir, find_missing_nums
    >>> folder = get_data_dir()
    >>> find_missing_nums(folder)

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
            if folder_exclusions is not None:
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
                except: # pragma: no cover # pylint: disable=W0702
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

    Parameters
    ----------
    folder : str
        Name of folder to process
    allowable_extensions : set of str, optional
        List of extensions to consider allowable in the folder

    Notes
    -----
    #.  Written by David C. Stauffer in December 2013.

    Examples
    --------

    >>> from dstauffman import get_data_dir, find_unexpected_ext
    >>> folder = get_data_dir()
    >>> find_unexpected_ext(folder) # doctest: +ELLIPSIS
    Finding any unexpected file extensions...
     Unexpected: "..."
    Done.

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
    Renames the old "Picasa.ini" to the newer ".picasa.ini" standard.

    Parameters
    ----------
    folder : str
        Name of folder to process

    Notes
    -----
    #.  Written by David C. Stauffer in December 2013.

    Examples
    --------

    >>> from dstauffman import get_data_dir, rename_old_picasa_files
    >>> folder = get_data_dir()
    >>> rename_old_picasa_files(folder)

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
                except: # pragma: no cover # pylint: disable=W0702
                    # print any problems and then continue
                    print('Unable to rename: "{}"'.format(old_path))
                    continue

#%% Functions - rename_upper_ext
def rename_upper_ext(folder, allowable_extensions=ALLOWABLE_EXTENSIONS):
    r"""
    Renames any expected file types to have all lowercase file extensions.

    Common use is to rename the *.JPG extensions from my camera to *.jpg

    Parameters
    ----------
    folder : str
        Name of folder to process
    allowable_extensions : set of str, optional
        List of extensions to consider allowable in the folder

    Notes
    -----
    #.  Written by David C. Stauffer in December 2013.

    Examples
    --------

    >>> from dstauffman import get_data_dir, rename_upper_ext
    >>> folder = get_data_dir()
    >>> rename_upper_ext(folder)
    Searching for file extensions to rename...
    Done.

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
                except: # pragma: no cover # pylint: disable=W0702
                    # print any exceptions, but then continue
                    print(' Unable to rename: "{}"'.format(old_path))
                    continue
    print('Done.')

#%% Functions - find_long_filenames
def find_long_filenames(folder):
    r"""
    Finds any files with really long filenames.

    Parameters
    ----------
    folder : str
        Name of folder to process

    Notes
    -----
    #.  Written by David C. Stauffer in December 2013.

    Examples
    --------

    >>> from dstauffman import get_data_dir, find_long_filenames
    >>> folder = get_data_dir()
    >>> find_long_filenames(folder) # doctest: +ELLIPSIS
    Finding long filenames...
     max name = ...
     max root = ...
     max full = ...
    Done.

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
            if temp > 106: # pragma: no cover
                print(os.path.join(root, name))
            temp = len(root) - len_root
            if temp > max_root:
                max_root = temp
            if temp > 106:  # pragma: no cover
                print(root)
            temp = len(name) + len(root) - len_root
            if temp > max_full:
                max_full = temp
            if temp > 212: # pragma: no cover
                print(os.path.join(root, name))
            if ';' in name: # pragma: no cover
                print(os.path.join(root, name))
    print(' max name = {}'.format(max_name))
    print(' max root = {}'.format(max_root))
    print(' max full = {}'.format(max_full))
    print('Done.')

#%% Functions - batch_resize
def batch_resize(folder, max_width=8192, max_height=8192, \
        process_extensions=PROCESS_EXTENSIONS, enlarge=False):
    r"""
    Resize the specified folder of images to the given max width and height.

    Parameters
    ----------
    folder : str
        Name of folder to process
    max_width : int
        Maximum width for the resized photo
    max_height : int
        Maximum height for the resized photo
    process_extensions : set of str, optional
        List of extensions to be processed within the folder
    enlarge : bool
        Enlarge smaller images to the max size (True), or only shrink large ones (False)

    Notes
    -----
    #.  Written by David C. Stauffer in December 2013.
    #.  Updated to optionally not enlarge small images by David C. Stauffer in August 2015.
    #.  Updated by David C. Stauffer in December 2015 to use QImage instead of PIL (which is no
        longer maintained).  QImage is more powerful, but harder to use.  This function could now be
        rewritten to use the better resize options of QImage if desired.
    #.  Updated by David C. Stauffer in June 2016 to go back to PIL now that it (finally) supports
        Python 3.X.

    Examples
    --------

    >>> from dstauffman import get_data_dir, batch_resize
    >>> folder = get_data_dir()
    >>> batch_resize(folder, max_width=2048, max_height=2048) # doctest: +ELLIPSIS
    Processing folder: "..."
     Skipping file   : "..."
    Batch processing complete.

    """
    # update status
    print('Processing folder: "{}"'.format(folder))

    # Iterate through every image given in the folder argument and resize it.
    for image in os.listdir(folder):
        image_fullpath = os.path.join(folder, image)
        # check if valid image file
        if os.path.isdir(image_fullpath):
            continue
        elif image[-4:] not in process_extensions:
            print(' Skipping file   : "{}"'.format(image))
            continue

        # Open and load the image file
        with open(image_fullpath, 'rb') as file:
            img = Image.open(file)
            img.load()

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
        elif cal_height == max_height and cal_width == max_width:
            new_width  = cal_width
            new_height = cal_height
        else:
            raise ValueError("You shouldn't be able to get here, check out the logic and fix!") # pragma: no cover

        # Assert that everything is as expected
        assert new_width <= max_width, 'New width: "{}" is not <= max_width: "{}"'.format(new_width, max_width)
        assert new_height <= max_height, 'New height: "{}" is not <= max_height: "{}"'.format(new_height, max_height)
        temp = int(round(aspect_ratio * new_height))
        assert temp-1 <= new_width <= temp+1, 'New width: "{}" gives wrong aspect ratio height: "{}"'.format(new_width, temp)
        temp = int(round(new_width / aspect_ratio))
        assert temp-1 <= new_height <= temp+1, 'New height: "{}" gives wrong aspect ratio width: "{}"'.format(new_height, temp)
        assert new_width == max_width or new_height == max_height, 'New width: "{}" is not max_width: "{}" or new height "{}" is not max_height: "{}"'.format(new_width, max_width, new_height, max_height)

        # Update status, with options for enlarging or not
        status_msg = ' Resizing image  : "{}"'.format(image)
        if new_width > cur_width or new_height > cur_height:
            if not enlarge:
                print(' Not enlarging   : "{}"'.format(image))
                new_width  = cur_width
                new_height = cur_height
            else:
                print(status_msg)
        else:
            print(status_msg)

        # Resize it.
        new_img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # Create the output folder if necessary
        if not os.path.isdir(os.path.join(folder, 'resized')):
            setup_dir(os.path.join(folder, 'resized'))

        # Save it back to disk.
        new_img.save(os.path.join(folder, 'resized', image))

        # Close objects
        img.close()
        new_img.close()

    print('Batch processing complete.')

#%% Functions - convert_tif_to_jpg
def convert_tif_to_jpg(folder, max_width=8192, max_height=8192, replace=False, enlarge=False):
    r"""
    Converts *.tif images into *.jpg images.

    Parameters
    ----------
    folder : str
        Name of the folder to process
    max_width : int
        Maximum width for the resized photo
    max_height : int
        Maximum height for the resized photo
    replace : bool, optional
        Set to True to replace any existing *.jpg files
    enlarge : bool
        Enlarge smaller images to the max size (True), or only shrink large ones (False)

    Notes
    -----
    #.  Written by David C. Stauffer in August 2015.
    #.  Updated by David C. Stauffer in December 2015 to use QImage instead of PIL (which is no
        longer maintained).  QImage is more powerful, but harder to use.  This function could now be
        rewritten to use the better resize options of QImage if desired.
    #.  Updated by David C. Stauffer in June 2016 to go back to PIL now that it (finally) supports
        Python 3.X.

    Examples
    --------

    >>> from dstauffman import get_data_dir, convert_tif_to_jpg
    >>> folder = get_data_dir()
    >>> convert_tif_to_jpg(folder) # doctest: +ELLIPSIS
    Processing folder: "..."
     Skipping file   : "..."
    Batch processing complete.

    """
    # update status
    print('Processing folder: "{}"'.format(folder))

    # Iterate through every image given in the folder argument and resize it.
    for image in os.listdir(folder):
        image_fullpath = os.path.join(folder, image)
        # check if valid image tif file
        if os.path.isdir(image_fullpath):
            continue
        elif image_fullpath.split('.')[-1] not in {'tif','tiff'}:
            print(' Skipping file   : "{}"'.format(image))
            continue

        # get new name (handles *.tiff or *.tif)
        new_name = '.'.join(image_fullpath.split('.')[:-1]) + '.jpg'

        # determine if the file already exists
        if os.path.isfile(new_name) and not replace:
            print(' Skipping due to pre-existing jpg file: "{}"'.format(image))
            continue

        # Open and load the image file
        with open(image_fullpath, 'rb') as file:
            img = Image.open(file)
            img.load()

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
        elif cal_height == max_height and cal_width == max_width:
            new_width  = cal_width
            new_height = cal_height
        else:
            raise ValueError("You shouldn't be able to get here, check out the logic and fix!") # pragma: no cover

        # Assert that everything is as expected
        assert new_width <= max_width, 'New width: "{}" is not <= max_width: "{}"'.format(new_width, max_width)
        assert new_height <= max_height, 'New height: "{}" is not <= max_height: "{}"'.format(new_height, max_height)
        temp = int(round(aspect_ratio * new_height))
        assert temp-1 <= new_width <= temp+1, 'New width: "{}" gives wrong aspect ratio height: "{}"'.format(new_width, temp)
        temp = int(round(new_width / aspect_ratio))
        assert temp-1 <= new_height <= temp+1, 'New height: "{}" gives wrong aspect ratio width: "{}"'.format(new_height, temp)
        assert new_width == max_width or new_height == max_height, 'New width: "{}" is not max_width: "{}" or new height "{}" is not max_height: "{}"'.format(new_width, max_width, new_height, max_height)

        # Update status, with options for enlarging or not
        status_msg = ' Saving image    : "{}"'.format(new_name)
        if new_width > cur_width or new_height > cur_height:
            if not enlarge:
                print(' Saving (not enlarging) : "{}"'.format(new_name))
                new_width  = cur_width
                new_height = cur_height
            else:
                print(status_msg)
        else:
            print(status_msg)

        # Resize it.
        new_img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # Save it back to disk.
        new_img.save(new_name)

        # Close objects
        img.close()
        new_img.close()

    print('Batch processing complete.')

#%% number_files
def number_files(folder, prefix='Image ', start=1, digits=2, process_extensions=PROCESS_EXTENSIONS):
    r"""
    Numbers the files in the folder using the given prefix, starting value, and number of digits.

    Parameters
    ----------
    folder : str
        Name of the folder to process
    prefix : str
        Filename to use as prefix to each photo
    start  : int
        Number to use as start of counter, default is 1
    digits : int
        Number of digits to put in str, default is 2
    process_extensions : set of str, optional
        List of extensions to be processed within the folder

    Notes
    -----
    #.  Written by David C. Stauffer in August 2015.

    Examples
    --------
    >>> from dstauffman import number_files, get_data_dir
    >>> folder = get_data_dir()
    >>> prefix = 'Photo '
    >>> start  = 5
    >>> digits = 3
    >>> number_files(folder, prefix, start, digits) # doctest: +ELLIPSIS
    Processing folder: "..."
     Skipping file   : "..."
    Batch processing complete.

    """
    # update status
    print('Processing folder: "{}"'.format(folder))

    # initialize counter
    counter = start

    # build digits str command
    dig_str = '{:0' + '{}'.format(digits) + 'd}'

    # Iterate through every image given in the folder argument and resize it.
    for image in sorted(os.listdir(folder)):
        image_fullpath = os.path.join(folder, image)
        file_ext = image[-4:]
        # check if valid image file
        if os.path.isdir(image_fullpath):
            continue
        elif file_ext not in process_extensions:
            print(' Skipping file   : "{}"'.format(image))
            continue

        # create the new path
        new_name = prefix + dig_str.format(counter) + file_ext

        # rename the image
        print(' Renaming : "{}" to "{}"'.format(image, new_name))
        shutil.move(image_fullpath, os.path.join(folder, new_name))

        # increment counter
        counter = counter + 1

    print('Batch processing complete.')

#%% Unit test
if __name__ == '__main__':
    # run the tests
    unittest.main(module='tests.test_photos', exit=False)
    doctest.testmod(verbose=False)
