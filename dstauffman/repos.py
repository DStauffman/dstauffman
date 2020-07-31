r"""
Generic functions related to managing repositories.

Notes
-----
#.  Split out of utils by David C. Stauffer in July 2019.
"""

#%% Imports
import datetime
import doctest
import os
import sys
import unittest

import pytest
try:
    # TODO: this sometimes messes up the coverage tool, as the imports are done before it starts counting.
    # May have to move this function to an external tool to fix it.
    from coverage import Coverage
except ImportError:
    Coverage = None

from dstauffman.enums        import ReturnCodes
from dstauffman.paths        import get_tests_dir, list_python_files
from dstauffman.utils        import line_wrap, read_text_file, write_text_file
from dstauffman.utils_log    import setup_dir

#%% run_docstrings
def run_docstrings(files, verbose=False):
    r"""
    Runs all the docstrings in the given files.

    Parameters
    ----------
    files : list of str
        Files(s) to run tests from
    verbose : bool, optional, default is False
        Whether to print verbose information

    Returns
    -------
    return_code : class ReturnCodes
        Return code enum, 0 means clean

    Examples
    --------
    >>> from dstauffman import get_root_dir, list_python_files, run_docstrings
    >>> files = list_python_files(get_root_dir())
    >>> return_code = run_docstrings(files) # doctest: +SKIP

    """
    # disable plots from showing up
    from dstauffman.plotting import Plotter
    plotter = Plotter()
    plotter.set_plotter(False)
    # initialize failure status
    had_failure = False
    # loop through and test each file
    for file in files:
        if verbose:
            print('')
            print('******************************')
            print('******************************')
            print(f'Testing "{file}":')
        (failure_count, test_count) = doctest.testfile(file, report=True, verbose=verbose, module_relative=False)
        if failure_count > 0:
            had_failure = True
    return_code = ReturnCodes.test_failures if had_failure else ReturnCodes.clean
    return return_code

#%% run_unittests
def run_unittests(names, verbose=False):
    r"""
    Runs all the unittests with the given names using unittest.

    Parameters
    ----------
    names : str
        Names of the unit tests to run (discover through unittest library)
    verbose : bool, optional, default is False
        Whether to show verbose output to the screen

    Returns
    -------
    return_code : class ReturnCodes
        Return code enum, 0 means clean

    Examples
    --------
    >>> from dstauffman import run_unittests
    >>> names = 'dstauffman.tests'
    >>> return_code = run_unittests(names) # doctest: +SKIP

    """
    # disable plots from showing up
    from dstauffman.plotting import Plotter
    plotter = Plotter()
    plotter.set_plotter(False)
    # find the test cases
    test_suite = unittest.TestLoader().discover(names)
    # set the verbosity
    verbosity = 10 if verbose else 1
    # run the tests
    result = unittest.TextTestRunner(verbosity=verbosity).run(test_suite)
    return_code = ReturnCodes.clean if result.wasSuccessful() else ReturnCodes.test_failures
    return return_code

#%% run_pytests
def run_pytests(folder):
    r"""
    Runs all the unittests using pytest as the runner instead of unittest.

    Parameters
    ----------
    folder : str
        Folder to process for test cases
    names : str
        Names of the unit tests to run (discover through unittest library)

    Returns
    -------
    return_code : class ReturnCodes
        Return code enum, 0 means clean

    Examples
    --------
    >>> from dstauffman import run_pytests, get_root_dir
    >>> folder = get_root_dir()
    >>> return_code = run_pytests(folder) # doctest: +SKIP

    """
    # disable plots from showing up
    from dstauffman.plotting import Plotter
    plotter = Plotter()
    plotter.set_plotter(False)
    # Note: need to do this next part to keep GUI testing from closing the instance with sys.exit
    # open a qapp
    from PyQt5.QtWidgets import QApplication
    if QApplication.instance() is None:
        qapp = QApplication(sys.argv)
    else:
        qapp = QApplication.instance()
    # run tests using pytest
    exit_code = pytest.main([folder])
    # close the qapp
    qapp.closeAllWindows()
    return_code = ReturnCodes.clean if exit_code == 0 else ReturnCodes.test_failures
    return return_code

#%% run_coverage
def run_coverage(folder, *, names='tests', report=True):
    r"""
    Wraps the pytests with a Code Coverage report.

    Parameters
    ----------
    folder : str
        Folder to process for test cases
    names : str, optional
        Names of the unit tests to run (discover through unittest library)
    report : bool, optional, default is True
        Whether to generate the HTML report

    Returns
    -------
    return_code : class ReturnCodes
        Return code enum, 0 means clean

    Examples
    --------
    >>> from dstauffman import run_coverage, get_root_dir
    >>> folder = get_root_dir()
    >>> return_code = run_coverage(folder) # doctest: +SKIP

    """
    # check that coverage tool was imported
    if Coverage is None:
        print('coverage tool is not available, no report was generated.')
        return_code = ReturnCodes.no_coverage_tool
        return return_code

    # Get information on the test folder
    test_folder = get_tests_dir()
    data_file   = os.path.join(test_folder, '.coverage')
    config_file = os.path.join(test_folder, '.coveragerc')
    cov_folder  = os.path.join(test_folder, 'coverage_html_report')

    # Instantiate the coverage tool and start tracking
    cov = Coverage(data_file=data_file, config_file=config_file)
    cov.start()

    # Call test code
    return_code = run_pytests(folder, names=names)

    # Stop coverage tool and save results
    cov.stop()
    cov.save()

    # Generate the HTML report
    if report:
        cov.html_report(directory=cov_folder)

    return return_code

#%% find_repo_issues
def find_repo_issues(folder, extensions=frozenset(('m', 'py')), *, list_all=False, check_tabs=True, \
              trailing=False, exclusions=None, check_eol=None, show_execute=False):
    r"""
    Find all the tabs in source code that should be spaces instead.

    Parameters
    ----------
    folder : str
        Folder path to search
    extensions : tuple of str
        File extensions to consider, default is ('m', 'py')
    list_all : bool, optional, default is False
        Whether to list all the files, or only those with problems in them
    check_tabs : bool, optional, default is True
        Whether to include tabs as an issue to check
    trailing : bool, optional, default is False
        Whether to consider trailing whitespace a problem, too
    exclusions : tuple of str
        Folders to ignore, default is empty
    check_eol : str
        If not None, then the line endings to check, such as '\r\n'
    show_execute : bool
        Whether to show files that have execute permissions, default is False

    Returns
    -------
    is_clean : bool
        Whether the folder is clean, meaning nothing was found to report.

    Notes
    -----
    #.  This function will iterate over extensions and exclusions, so extensions='txt' will look for
        't' and 'x' instead of 'txt'.  Use extensions=('txt',) or ['txt'] instead.  Likewise for
        exclusions.

    Examples
    --------
    >>> from dstauffman import find_repo_issues, get_root_dir
    >>> folder = get_root_dir()
    >>> find_repo_issues(folder)
    True

    """
    def _is_excluded(path, exclusions):
        if exclusions is None:
            return False
        for this_exclusion in exclusions:
            if path.startswith(this_exclusion):
                return True
        return False

    # initialize output
    is_clean = True

    for (root, dirs, files) in os.walk(folder, topdown=True):
        dirs.sort()
        for name in sorted(files):
            fileparts = name.split('.')
            if extensions is None or fileparts[-1] in extensions:
                if _is_excluded(root, exclusions):
                    continue
                this_file = os.path.join(root, name)
                already_listed = False
                if list_all:
                    print(f'Evaluating: "{this_file}"')
                    already_listed = True
                if show_execute and os.access(this_file, os.X_OK):
                    print(f'File: "{this_file}" has execute privileges.')
                    is_clean = False
                with open(this_file, encoding='utf8', newline='') as file:
                    bad_lines = False
                    try:
                        lines = file.readlines()
                    except UnicodeDecodeError: # pragma: no cover
                        print(f'File: "{this_file}" was not a valid utf-8 file.')
                        is_clean = False
                    for (c, line) in enumerate(lines):
                        sline = line.rstrip('\n').rstrip('\r').rstrip('\n') # for all possible orderings
                        if check_tabs and line.count('\t') > 0:
                            if not already_listed:
                                print(f'Evaluating: "{this_file}"')
                                already_listed = True
                                is_clean = False
                            print('    Line {:03}: '.format(c+1) + repr(line))
                        elif trailing and len(sline) >= 1 and sline[-1] == ' ':
                            if not already_listed:
                                print(f'Evaluating: "{this_file}"')
                                already_listed = True
                                is_clean = False
                            print('    Line {:03}: '.format(c+1) + repr(line))
                        if check_eol is not None and c != len(lines)-1 and not line.endswith(check_eol) and not bad_lines:
                            line_ending = line[-(len(line) - len(sline)):]
                            print('File: "{}" has bad line endings of "{}".'.format(this_file, repr(line_ending)[1:-1]))
                            bad_lines = True
                            is_clean = False
    # end checks, return overall result
    return is_clean

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
def get_python_definitions(text, *, include_private=False):
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
        if line.startswith('class ') and (include_private or not line.startswith('class _')):
            temp = line[len('class '):].split('(')
            temp = temp[0].split(':') # for classes without arguments
            funcs.append(temp[0])
        if line.startswith('def ') and (include_private or not line.startswith('def _')):
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
    >>> print(text[0:22])
    from .classes   import

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

#%% write_unit_test_templates
def write_unit_test_templates(folder, output, *, author='unknown', exclude=None, recursive=True, \
        repo_subs=None, add_classification=False):
    r"""
    Writes template files for unit tests.  These can then be used with a diff tool to find what is missing.

    Parameters
    ----------

    Notes
    -----
    #.  Written by David C. Stauffer in July 2020.

    Examples
    --------
    >>> from dstauffman import write_unit_test_templates, get_root_dir, get_tests_dir
    >>> folder = get_root_dir()
    >>> output = get_tests_dir() + '_template'
    >>> author = 'David C. Stauffer'
    >>> exclude = get_tests_dir() # can also be tuple of exclusions
    >>> write_unit_test_templates(folder, output, author=author, exclude=exclude) # doctest: +SKIP

    """
    # hard-coded substitutions for imports
    _subs = {'dstauffman': 'dcs', 'dstauffman.aerospace': 'space', 'dstauffman.commands': 'commands', \
              'dstauffman.estimation': 'estm', 'dstauffman.health': 'health', 'dstauffman.plotting': 'plot'}
    if repo_subs is not None:
        _subs.update(repo_subs)
    # create the output location
    setup_dir(output)
    # get all the files
    files = list_python_files(folder, recursive=recursive)
    # save the starting point in the name
    num = len(folder) + 1
    # get the name of the repository
    repo_name = files[0][:num-1].replace('\\','/').split('/')[-1]
    # get date information
    now = datetime.datetime.now()
    month = now.strftime('%B')
    year = now.strftime('%Y')
    for file in files:
        # check for exclusions
        if file.startswith(exclude) or file.startswith(output):
            continue
        # read the contents of the file
        text = read_text_file(file)
        # get a list of definitions from the text file
        funcs = get_python_definitions(text, include_private=True)
        # get the name of the test file
        names = file[num:].replace('\\','/').split('/')
        # get the name of the repo or sub-repo
        sub_repo = '.'.join(names[:-1])
        this_repo = repo_name + ('.' + sub_repo if sub_repo else '')
        # create the text to write to the file
        text = ['r"""']
        text += [f'Test file for the `{names[-1][:-3]}` module of the "{this_repo}" library.']
        text += ['', 'Notes', '-----', f'#.  Written by {author} in {month} {year}.']
        if add_classification:
            text += ['', 'Classification', '--------------', 'TBD']
        text += ['"""', '', '#%% Imports', 'import unittest', '']
        import_text = 'import ' + this_repo
        if this_repo in _subs:
            import_text += ' as ' + _subs[this_repo]
        text += [import_text, '']
        for func in funcs:
            if func.startswith('_'):
                func = names[-1][:-3] + '.' + func
            func_name = sub_repo + '.' + func if sub_repo else func
            temp_name = func_name.replace('.', '_')
            text += [f'#%% {func_name}', f'class Test_{temp_name}(unittest.TestCase):', '    r"""']
            text += [f'    Tests the {func_name} function with the following cases:','        TBD']
            text += ['    """', '    pass # TODO: write this', '']

        text += ['#%% Unit test execution', "if __name__ == '__main__':", '    unittest.main(exit=False)', '']
        new_file = os.path.join(output, 'test_' + '_'.join(names))
        print(f'Writing: "{new_file}".')
        write_text_file(new_file, '\n'.join(text))

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_repos', exit=False)
    doctest.testmod(verbose=False)
