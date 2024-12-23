r"""
Generic functions related to managing repositories.

Notes
-----
#.  Split out of utils by David C. Stauffer in July 2019.
"""

# %% Imports
from __future__ import annotations

import datetime
import doctest
import os
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING
import unittest

from slog import ReturnCodes

from dstauffman.constants import HAVE_COVERAGE, HAVE_PYTEST
from dstauffman.paths import get_root_dir, get_tests_dir, list_python_files
from dstauffman.utils import line_wrap, read_text_file, write_text_file
from dstauffman.utils_log import setup_dir

if HAVE_COVERAGE:
    from coverage import Coverage
if HAVE_PYTEST:
    import pytest

if TYPE_CHECKING:
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtWidgets import QApplication

    assert QApplication  # type: ignore[truthy-function]


# %% run_docstrings
def run_docstrings(files: list[Path], verbose: bool = False) -> int:
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
    try:
        from dstauffman.plotting import suppress_plots  # pylint: disable=import-outside-toplevel
    except:
        pass
    else:
        suppress_plots()
    # initialize failure status
    had_failure = False
    # loop through and test each file
    for file in files:
        if verbose:
            print("")
            print("******************************")
            print("******************************")
            print(f'Testing "{file}":')
        (failure_count, _) = doctest.testfile(file, report=True, verbose=verbose, module_relative=False)  # type: ignore[arg-type, misc]
        if failure_count > 0:
            had_failure = True
    return_code = ReturnCodes.test_failures if had_failure else ReturnCodes.clean
    return return_code


# %% run_unittests
def run_unittests(names: str, verbose: bool = False) -> int:
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
    >>> names = "dstauffman.tests"
    >>> return_code = run_unittests(names) # doctest: +SKIP

    """
    # disable plots from showing up
    try:
        from dstauffman.plotting import suppress_plots  # pylint: disable=import-outside-toplevel
    except:
        pass
    else:
        suppress_plots()
    # find the test cases
    test_suite = unittest.TestLoader().discover(names)
    # set the verbosity
    verbosity = 10 if verbose else 1
    # run the tests
    result = unittest.TextTestRunner(verbosity=verbosity).run(test_suite)
    return_code: int = ReturnCodes.clean if result.wasSuccessful() else ReturnCodes.test_failures
    return return_code


# %% run_pytests
def run_pytests(folder: Path, *args: str, **kwargs: Any) -> int:
    r"""
    Runs all the unittests using pytest as the runner instead of unittest.

    Parameters
    ----------
    folder : class pathlib.Path
        Folder to process for test cases

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
    try:
        from dstauffman.plotting import suppress_plots  # pylint: disable=import-outside-toplevel
    except:
        pass
    else:
        suppress_plots()
    # Note: need to do this next part to keep GUI testing from closing the instance with sys.exit
    # open a qapp
    qapp: QApplication | QCoreApplication | None
    try:
        from qtpy.QtWidgets import QApplication  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        qapp = None
    except ImportError:
        qapp = None
    else:
        if QApplication.instance() is None:
            qapp = QApplication(sys.argv)
        else:
            qapp = QApplication.instance()
    # run tests using pytest
    exit_code = pytest.main([str(folder), "-rfEsP"] + list(*args), **kwargs)
    # close the qapp
    if qapp is not None:
        qapp.closeAllWindows()  # type: ignore[attr-defined]
    return_code = ReturnCodes.clean if exit_code == 0 else ReturnCodes.test_failures
    return return_code


# %% run_coverage
def run_coverage(folder: Path, *, report: bool = True, cov_file: Path | None = None) -> int:
    r"""
    Wraps the pytests with a Code Coverage report.

    Parameters
    ----------
    folder : class Path or str
        Folder to process for test cases
    report : bool, optional, default is True
        Whether to generate the HTML report
    cov_file : class Path or str, optional
        File to output the coverage results to

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
    if not HAVE_COVERAGE:
        print("coverage tool is not available, no report was generated.")
        return_code: int = ReturnCodes.no_coverage_tool
        return return_code

    # Get information on the test folder
    test_folder = get_tests_dir()
    data_file = test_folder / ".coverage" if cov_file is None else Path(cov_file)
    config_file = get_root_dir().parent / "pyproject.toml"
    cov_folder = test_folder / "coverage_html_report"

    # Instantiate the coverage tool and start tracking
    # TODO: work around temporary coverage bug?  Should definitely be able to pass pathlib.Path instead of str here
    cov = Coverage(data_file=str(data_file), config_file=str(config_file))
    cov.start()

    # Call test code
    return_code = run_pytests(folder)

    # Stop coverage tool and save results
    cov.stop()
    cov.save()

    # Generate the HTML report
    if report:
        cov.html_report(directory=str(cov_folder))

    return return_code


# %% find_repo_issues
def find_repo_issues(  # noqa: C901
    folder: Path,
    extensions: frozenset[str] | set[str] | tuple[str, ...] | str | None = frozenset((".m", ".py")),
    *,
    list_all: bool = False,
    check_tabs: bool = True,
    trailing: bool = False,
    exclusions: tuple[Path, ...] | Path | None = None,
    check_eol: str | None = None,
    show_execute: bool = False,
) -> bool:
    r"""
    Find all the tabs in source code that should be spaces instead.

    Parameters
    ----------
    folder : class pathlib.Path
        Folder path to search
    extensions : tuple of str
        File extensions to consider, default is (".m", ".py")
    list_all : bool, optional, default is False
        Whether to list all the files, or only those with problems in them
    check_tabs : bool, optional, default is True
        Whether to include tabs as an issue to check
    trailing : bool, optional, default is False
        Whether to consider trailing whitespace a problem, too
    exclusions : tuple of pathlib.Path
        Folders to ignore, default is empty
    check_eol : str
        If not None, then the line endings to check, such as "\r\n"
    show_execute : bool
        Whether to show files that have execute permissions, default is False

    Returns
    -------
    is_clean : bool
        Whether the folder is clean, meaning nothing was found to report.

    Examples
    --------
    >>> from dstauffman import find_repo_issues, get_root_dir
    >>> folder = get_root_dir()
    >>> is_clean = find_repo_issues(folder)
    >>> print(is_clean)
    True

    """

    def _is_excluded(path: Path, exclusions: tuple[Path, ...] | None) -> bool:
        if exclusions is None:
            return False
        for this_exclusion in exclusions:
            if this_exclusion == path or this_exclusion in path.parents:
                return True
        return False

    # initialize output
    is_clean = True

    if isinstance(extensions, str):
        extensions = {extensions,}  # fmt: skip
    if isinstance(exclusions, Path):
        exclusions = (exclusions,)

    for this_file in folder.rglob("*"):
        if not this_file.is_file():
            continue
        if extensions is None or this_file.suffix in extensions:
            if _is_excluded(folder, exclusions):
                continue
            already_listed = False
            if list_all:
                print(f'Evaluating: "{this_file}"')
                already_listed = True
            if show_execute and os.access(this_file, os.X_OK):
                print(f'File: "{this_file}" has execute privileges.')
                is_clean = False
            with open(this_file, encoding="utf8", newline="") as file:
                bad_lines = False
                try:
                    lines = file.readlines()
                except UnicodeDecodeError:  # pragma: no cover
                    print(f'File: "{this_file}" was not a valid utf-8 file.')
                    is_clean = False
                for c, line in enumerate(lines):
                    sline = line.rstrip("\n").rstrip("\r").rstrip("\n")  # for all possible orderings
                    if check_tabs and line.count("\t") > 0:
                        if not already_listed:
                            print(f'Evaluating: "{this_file}"')
                            already_listed = True
                            is_clean = False
                        print(f"    Line {c + 1:03}: " + repr(line))
                    elif trailing and len(sline) >= 1 and sline[-1] == " ":
                        if not already_listed:
                            print(f'Evaluating: "{this_file}"')
                            already_listed = True
                            is_clean = False
                        print(f"    Line {c + 1:03}: " + repr(line))
                    if check_eol is not None and c != len(lines) - 1 and not line.endswith(check_eol) and not bad_lines:
                        line_ending = line[-(len(line) - len(sline)) :]
                        print(f'File: "{this_file}" has bad line endings of "{repr(line_ending)[1:-1]}".')
                        bad_lines = True
                        is_clean = False
    # end checks, return overall result
    return is_clean


# %% Functions - delete_pyc
def delete_pyc(folder: Path, recursive: bool = True, *, print_progress: bool = True) -> None:
    r"""
    Delete all the *.pyc files (Python Byte Code) in the specified directory.

    Parameters
    ----------
    folder : class pathlib.Path
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

    def _remove_pyc(file: Path) -> None:
        r"""Do the actual file removal."""
        # check for allowable extensions
        # fmt: off
        assert file.suffix in {".pyc",}
        assert file.is_file()
        # fmt: on
        # remove this file
        if print_progress:
            print(f'Removing "{file}"')
        file.unlink(missing_ok=True)

    if recursive:
        # walk through folder
        for file in folder.rglob("*.pyc"):
            # remove relevant files
            _remove_pyc(file)
    else:
        # list files in folder
        for file in folder.glob("*.pyc"):
            # remove relevant files
            _remove_pyc(file)


# %% Functions - get_python_definitions
def get_python_definitions(text: str, *, include_private: bool = False) -> list[str]:  # noqa: C901
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
    >>> text = "def a():\n    pass\n"
    >>> funcs = get_python_definitions(text)
    >>> print(funcs)
    ['a']

    """
    cap_letters = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    extended_letters = frozenset(cap_letters & {"_"})
    assert len(cap_letters) == 26
    funcs: list[str] = []
    skip_next = False
    skip_strs = False
    for line in text.split("\n"):
        # check for @overload function definitions and skip them
        if skip_next:
            skip_next = False
            continue
        if skip_strs:
            if line.endswith('"""'):
                skip_strs = False
            continue
        if line == "@overload":
            skip_next = True
            continue
        if line.startswith('r"""') or line.startswith('"""'):
            skip_strs = True
        if line.startswith("class ") and (include_private or not line.startswith("class _")):
            temp = line[len("class ") :].split("(")
            temp = temp[0].split(":")  # for classes without arguments
            funcs.append(temp[0])
        if line.startswith("def ") and (include_private or not line.startswith("def _")):
            temp = line[len("def ") :].split("(")
            temp = temp[0].split(":")  # for functions without arguments
            funcs.append(temp[0])
        if len(line) > 0 and line[0] in cap_letters and "=" in line and " " in line:
            temp2 = line.split(" ")[0].split(":")[0]
            if len(extended_letters - set(temp2)) == 0:
                funcs.append(temp2)
    return funcs


# %% Functions - make_python_init
def make_python_init(folder: Path, *, lineup: bool = True, wrap: int = 100, filename: Path | None = None) -> str:
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
    from .binary    import

    """
    # exclusions
    exclusions = ["__init__.py"]
    # initialize intermediate results
    results = {}
    # Loop through the contained files/folders
    for this_elem in folder.glob("*"):
        # check if a folder or file
        if not this_elem.is_dir():
            # only process source *.py files
            if this_elem.suffix == ".py":
                # exclude any existing "__init__.py" file
                if any((exc in this_elem.parents for exc in exclusions)):
                    continue
                # read the contents of the file
                this_text = read_text_file(this_elem)
                # get a list of definitions from the text file
                funcs = get_python_definitions(this_text)
                # append these results (if not empty)
                if len(funcs) > 0:
                    results[this_elem.stem] = funcs
    # check for duplicates
    all_funcs = [func for v in results.values() for func in v]
    if len(all_funcs) != len(set(all_funcs)):
        print(f"Uniqueness Problem: {len(all_funcs)} functions, but only {len(set(all_funcs))} unique functions")
    dups = set((x for x in all_funcs if all_funcs.count(x) > 1))
    if dups:
        print("Duplicated functions:")
        print(dups)
    # get information about padding
    max_len = max(len(x) for x in results)
    indent = len("from . import ") + max_len + 4
    # start building text output
    text: list[str] = []
    # loop through results and build text output
    for key in sorted(results):
        pad = " " * (max_len - len(key)) if lineup else ""
        temp = ", ".join(results[key])
        header = "from ." + key + pad + " import "
        min_wrap = len(header)
        this_line = [header + temp]
        wrapped_lines = line_wrap(this_line, wrap=wrap, min_wrap=min_wrap, indent=indent)
        text += wrapped_lines
    # combined the text into a single string with newline characters
    output = "\n".join(text)
    # optionally write the results to a file
    if filename is not None:
        write_text_file(filename, output)
    return output


# %% write_unit_test_templates
def write_unit_test_templates(
    folder: Path,
    output: Path,
    *,
    author: str = "unknown",
    exclude: Path | tuple[Path, ...] | None = None,
    recursive: bool = True,
    repo_subs: dict[str, str] | None = None,
    add_classification: bool = False,
) -> None:
    r"""
    Writes template files for unit tests.  These can then be used with a diff tool to find what is missing.

    Parameters
    ----------
    folder : class pathlib.Path
        Folder location of files to write the unit tests for
    output : class pathlib.Path
        Folder location of the output unit tests
    author : str, optional
        Name of the author
    exclude : pathlib.Path or (pathlib.Path, ...), optional
        Names to exclude
    recursive : bool, optional
        Whether to process recursively
    repo_subs : dict[str, str], optional
        Repository names to replace
    add_classification : bool, optional
        Whether to add a classification to the headers

    Notes
    -----
    #.  Written by David C. Stauffer in July 2020.

    Examples
    --------
    >>> from dstauffman import write_unit_test_templates, get_root_dir, get_tests_dir
    >>> from pathlib import Path
    >>> folder = get_root_dir()
    >>> output = Path(str(get_tests_dir()) + "_template")
    >>> author = "David C. Stauffer"
    >>> exclude = get_tests_dir() # can also be tuple of exclusions
    >>> write_unit_test_templates(folder, output, author=author, exclude=exclude) # doctest: +SKIP

    """
    # hard-coded substitutions for imports
    _subs = {
        "dstauffman": "dcs",
        "dstauffman.aerospace": "space",
        "dstauffman.commands": "commands",
        "dstauffman.estimation": "estm",
        "dstauffman.health": "health",
        "dstauffman.plotting": "plot",
    }
    if repo_subs is not None:
        _subs.update(repo_subs)
    # create the output location
    setup_dir(output)
    # get all the files
    files = list_python_files(folder, recursive=recursive)
    # save the starting point in the name
    num = len(str(folder)) + 1
    # get the name of the repository
    repo_name = files[0].parent.name
    # get date information
    now = datetime.datetime.now()
    month = now.strftime("%B")
    year = now.strftime("%Y")
    for file in files:
        # check for exclusions
        if exclude is not None and exclude in file.parents or output in file.parents:
            continue
        # read the contents of the file
        this_text = read_text_file(file)
        # get a list of definitions from the text file
        funcs = get_python_definitions(this_text, include_private=True)
        # get the name of the test file
        names = str(file)[num:].replace("\\", "/").split("/")
        # get the name of the repo or sub-repo
        sub_repo = ".".join(names[:-1])
        this_repo = repo_name + ("." + sub_repo if sub_repo else "")
        # create the text to write to the file
        text = ['r"""']
        text += [f'Test file for the `{names[-1][:-3]}` module of the "{this_repo}" library.']
        text += ["", "Notes", "-----", f"#.  Written by {author} in {month} {year}."]
        if add_classification:
            text += ["", "Classification", "--------------", "TBD"]
        text += ['"""', "", "# %% Imports", "import unittest", ""]
        import_text = "import " + this_repo
        if this_repo in _subs:
            import_text += " as " + _subs[this_repo]
        text += [import_text, ""]
        for func in funcs:
            if func.startswith("_"):
                func = names[-1][:-3] + "." + func
            func_name = sub_repo + "." + func if sub_repo else func
            temp_name = func_name.replace(".", "_")
            text += [f"# %% {func_name}", f"class Test_{temp_name}(unittest.TestCase):", '    r"""']
            text += [f"    Tests the {func_name} function with the following cases:", "        TBD"]
            text += ['    """', "    pass  # TODO: write this", ""]

        text += ["# %% Unit test execution", 'if __name__ == "__main__":', "    unittest.main(exit=False)", ""]
        new_file = Path.joinpath(output, "test_" + "_".join(names))
        print(f'Writing: "{new_file}".')
        write_text_file(new_file, "\n".join(text))


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_repos", exit=False)
    doctest.testmod(verbose=False)
