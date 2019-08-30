# -*- coding: utf-8 -*-
r"""
Generic functions related to managing repositories.

Notes
-----
#.  Split out of utils by David C. Stauffer in July 2019.

"""

#%% Imports
import doctest
import os
import shutil
import sys
import types
import unittest

from dstauffman.utils import line_wrap, read_text_file, setup_dir, write_text_file

#%% find_tabs
def find_tabs(folder, extensions=frozenset(('m', 'py')), *, list_all=False, trailing=False, \
              exclusions=None, check_eol=None, show_execute=False):
    r"""
    Find all the tabs in source code that should be spaces instead.

    Parameters
    ----------
    folder : str
        Folder path to search
    extensions : tuple of str
        File extensions to consider, default is ('m', 'py')
    list_all : bool, optional, default is False
        Whether to list all the files, or only those with tabs in them
    trailing : bool, optional, default is False
        Whether to consider trailing whitespace a problem, too
    exclusions : tuple of str
        Folders to ignore, default is empty
    check_eol : str
        If not None, then the line endings to check, such as '\r\n'

    Notes
    -----
    #.  This function will iterate over extensions and exclusions, so extensions='txt' will look for
        't' and 'x' instead of 'txt'.  Use extensions=('txt',) or ['txt'] instead.  Likewise for
        exclusions.

    Examples
    --------
    >>> from dstauffman import find_tabs, get_root_dir
    >>> folder = get_root_dir()
    >>> find_tabs(folder)

    """
    def _is_excluded(path, exclusions):
        if exclusions is None:
            return False
        for this_exclusion in exclusions:
            if path.startswith(this_exclusion):
                return True
        return False

    for (root, dirs, files) in os.walk(folder, topdown=True):
        dirs.sort()
        for name in sorted(files):
            fileparts = name.split('.')
            if extensions is None or fileparts[-1] in extensions:
                if _is_excluded(root, exclusions):
                    continue
                this_file = os.path.join(root, name)
                already_listed = list_all
                if already_listed:
                    print('Evaluating: "{}"'.format(this_file))
                if show_execute and os.access(this_file, os.X_OK):
                    print('File: "{}" has execute privileges.'.format(this_file))
                with open(this_file, encoding='utf8', newline='') as file:
                    bad_lines = False
                    try:
                        lines = file.readlines()
                    except UnicodeDecodeError: # pragma: no cover
                        print('File: "{}" was not a valid utf-8 file.'.format(this_file))
                    for (c, line) in enumerate(lines):
                        sline = line.rstrip('\n').rstrip('\r').rstrip('\n') # for all possible orderings
                        if line.count('\t') > 0:
                            if not already_listed:
                                print('Evaluating: "{}"'.format(this_file))
                                already_listed = True
                            print('    Line {:03}: '.format(c+1) + repr(line))
                        elif trailing and len(sline) >= 1 and sline[-1] == ' ':
                            if not already_listed:
                                print('Evaluating: "{}"'.format(this_file))
                                already_listed = True
                            print('    Line {:03}: '.format(c+1) + repr(line))
                        if check_eol is not None and c != len(lines)-1 and not line.endswith(check_eol) and not bad_lines:
                            line_ending = line[-(len(line) - len(sline)):]
                            print('File: "{}" has bad line endings of "{}".'.format(this_file, repr(line_ending)[1:-1]))
                            bad_lines = True

#%% Functions - delete_pyc
def delete_pyc(folder, recursive=True, print_progress=True):
    r"""
    Delete all the *.pyc files (Python Byte Code) in the specified directory.

    Parameters
    ----------
    folder : str
        Name of folder to delete the files from
    recursive : bool, optional
        Whether to delete files recursively
    print_progress: bool, optional
        Whether to display information about any deleted files

    Examples
    --------
    >>> from dstauffman import get_root_dir, delete_pyc
    >>> folder = get_root_dir()
    >>> delete_pyc(folder, print_progress=False) # doctest: +SKIP

    """
    def _remove_pyc(root, name):
        r"""Do the actual file removal."""
        # check for allowable extensions
        (_, file_ext) = os.path.splitext(name)
        if file_ext == '.pyc': # TODO: or file_ext == 'pyo' ???  Add file extension list?
            # remove this file
            if print_progress:
                print('Removing "{}"'.format(os.path.join(root, name)))
            os.remove(os.path.join(root, name))

    if recursive:
        # walk through folder
        for (root, _, files) in os.walk(folder):
            # go through files
            for name in files:
                # remove relevant files
                _remove_pyc(root, name)
    else:
        # list files in folder
        for name in os.listdir(folder):
            # check if it's a file
            if os.path.isfile(os.path.join(folder, name)):
                # remove relevant files
                _remove_pyc(folder, name)

#%% Functions - get_python_definitions
def get_python_definitions(text):
    r"""
    Get all public class and def names from the text of the file.

    Parameters
    ----------
    text : str
        The text of the python file

    Returns
    -------
    funcs : array_like, str
        List of functions within the text of the python file

    Examples
    --------
    >>> from dstauffman import get_python_definitions
    >>> text = 'def a():\n    pass\n'
    >>> funcs = get_python_definitions(text)
    >>> print(funcs)
    ['a']

    """
    cap_letters = frozenset('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    extended_letters = frozenset(cap_letters & {'_'})
    assert len(cap_letters) == 26
    funcs = []
    for line in text.split('\n'):
        if line.startswith('class ') and not line.startswith('class _'):
            temp = line[len('class '):].split('(')
            temp = temp[0].split(':') # for classes without arguments
            funcs.append(temp[0])
        if line.startswith('def ') and not line.startswith('def _'):
            temp = line[len('def '):].split('(')
            temp = temp[0].split(':') # for functions without arguments
            funcs.append(temp[0])
        if len(line) > 0 and line[0] in cap_letters and '=' in line and ' ' in line:
            temp = line.split(' ')[0]
            if len(extended_letters - set(temp)) == 0:
                funcs.append(temp)
    return funcs

#%% Functions - make_python_init
def make_python_init(folder, lineup=True, wrap=100, filename=''):
    r"""
    Make the Python __init__.py file based on the files/definitions found within the specified folder.

    Parameters
    ----------
    folder : str
        Name of folder to process

    Returns
    -------
    output : str
        Resulting text for __init__.py file

    Notes
    -----
    #.  This tool is written without using the dis library, such that the code does not have to be
        valid or importable into Python.  It can thus be used very early on in the development
        cycle. The files are read as text.

    Examples
    --------
    >>> from dstauffman import make_python_init, get_root_dir
    >>> folder = get_root_dir()
    >>> text = make_python_init(folder)
    >>> print(text[0:25])
    from .analysis     import

    """
    # exclusions
    exclusions = ['__init__.py']
    # initialize intermediate results
    results = {}
    # Loop through the contained files/folders
    for this_elem in os.listdir(folder):
        # alias the fullpath of this file element
        this_full_elem = os.path.join(folder, this_elem)
        # check if a folder or file
        if not os.path.isdir(this_full_elem):
            # get the file extension
            fileext = this_full_elem.split('.')
            # only process source *.py files
            if fileext[-1] == 'py':
                # exclude any existing '__init__.py' file
                if any([this_elem.startswith(exc) for exc in exclusions]):
                    continue
                # read the contents of the file
                text = read_text_file(this_full_elem)
                # get a list of definitions from the text file
                funcs = get_python_definitions(text)
                # append these results (if not empty)
                if len(funcs) > 0:
                    results[this_elem[:-3]] = funcs
    # check for duplicates
    all_funcs = [func for k in results for func in results[k]]
    if len(all_funcs) != len(set(all_funcs)):
        print('Uniqueness Problem: {funs} functions, but only {uni_funcs} unique functions'.format( \
            funs=len(all_funcs), uni_funcs=len(set(all_funcs))))
    dups = set([x for x in all_funcs if all_funcs.count(x) > 1])
    if dups:
        print('Duplicated functions:')
        print(dups)
    # get information about padding
    max_len   = max(len(x) for x in results)
    indent = len('from . import ') + max_len + 4
    # start building text output
    text = []
    # loop through results and build text output
    for key in sorted(results):
        pad = ' ' * (max_len - len(key)) if lineup else ''
        temp = ', '.join(results[key])
        header = 'from .' + key + pad + ' import '
        min_wrap = len(header)
        this_line = [header + temp]
        wrapped_lines = line_wrap(this_line, wrap=wrap, min_wrap=min_wrap, indent=indent)
        text += wrapped_lines
    # combined the text into a single string with newline characters
    output = '\n'.join(text)
    # optionally write the results to a file
    if filename:
        write_text_file(filename, output)
    return output

#%% Functions - reload_package
def reload_package(root_module, disp_reloads=True): # pragma: no cover
    r"""
    Force Python to reload all the items within a module.  Useful for interactive debugging in IPython.

    Parameters
    ----------
    root_module : module
        The module to force python to reload
    disp_reloads : bool
        Whether to display the modules that are reloaded

    Notes
    -----
    #.  Added to the library in June 2015 from:
        http://stackoverflow.com/questions/2918898/prevent-python-from-caching-the-imported-modules/2918951#2918951
    #.  Restarting the IPython kernel is by far a safer option, but for some debugging cases this is useful,
        however unit tests on this function will fail, because it reloads too many dependencies.

    Examples
    --------
    >>> from dstauffman import reload_package
    >>> import dstauffman as dcs
    >>> reload_package(dcs) # doctest: +ELLIPSIS
    loading dstauffman...

    """
    package_name = root_module.__name__

    # get a reference to each loaded module
    loaded_package_modules = dict([
        (key, value) for key, value in sys.modules.items()
        if key.startswith(package_name) and isinstance(value, types.ModuleType)])

    # delete references to these loaded modules from sys.modules
    for key in loaded_package_modules:
        del sys.modules[key]

    # load each of the modules again
    # make old modules share state with new modules
    for key in loaded_package_modules:
        if disp_reloads:
            print('loading {}'.format(key))
        newmodule = __import__(key)
        oldmodule = loaded_package_modules[key]
        oldmodule.__dict__.clear()
        oldmodule.__dict__.update(newmodule.__dict__)

#%% rename_module
def rename_module(folder, old_name, new_name, print_status=True):
    r"""
    Rename the given module from the old to new name.

    Parameters
    ----------
    folder : str
        Name of folder to operate within
    old_name : str
        The original name of the module
    new_name : str
        The new name of the module

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015 mostly for Matt Beck, because he doesn't
        like the name of my library.

    Examples
    --------
    >>> from dstauffman import rename_module, get_root_dir
    >>> import os
    >>> folder = os.path.split(get_root_dir())[0]
    >>> old_name = 'dstauffman'
    >>> new_name = 'dcs_tools'
    >>> rename_module(folder, old_name, new_name) # doctest: +SKIP

    """
    # hard-coded values
    folder_exclusions = {'.git'}
    file_exclusions = {'.pyc'}
    files_to_edit = {'.py', '.rst', '.bat', '.txt'}
    root_ix = len(folder)
    for (root, _, files) in os.walk(os.path.join(folder, old_name)):
        for skip in folder_exclusions:
            if root.endswith(skip) or root.find(skip + os.path.sep) >= 0:
                if print_status:
                    print('Skipping: {}'.format(root))
                break
        else:
            for name in files:
                (_, file_ext) = os.path.splitext(name)
                this_old_file = os.path.join(root, name)
                this_new_file = os.path.join(folder + root[root_ix:].replace(old_name, new_name), \
                    name.replace(old_name, new_name))
                if file_ext in file_exclusions:
                    if print_status:
                        print('Skipping: {}'.format(this_old_file))
                    continue # pragma: no cover (actually covered, optimization issue)
                # only create the new folder if not all files skipped
                new_folder = os.path.split(this_new_file)[0]
                if not os.path.isdir(new_folder):
                    setup_dir(new_folder)
                if file_ext in files_to_edit:
                    # edit files
                    if print_status:
                        print('Editing : {}'.format(this_old_file))
                        print('     To : {}'.format(this_new_file))
                    text = read_text_file(this_old_file)
                    text = text.replace(old_name, new_name)
                    write_text_file(this_new_file, text)
                else:
                    # copy file as-is
                    if print_status:
                        print('Copying : {}'.format(this_old_file))
                        print('     To : {}'.format(this_new_file))
                    shutil.copyfile(this_old_file, this_new_file)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_repos', exit=False)
    doctest.testmod(verbose=False)
