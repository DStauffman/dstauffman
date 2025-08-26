r"""
Generic functions related to managing repositories.

Notes
-----
#.  Split out of utils by David C. Stauffer in July 2019.

"""

# %% Imports
from __future__ import annotations

import doctest
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING
import unittest

from slog import ReturnCodes

from dstauffman.constants import HAVE_COVERAGE, HAVE_PYTEST
from dstauffman.paths import get_root_dir, get_tests_dir

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
    >>> from dstauffman import get_root_dir, run_docstrings
    >>> from slog import list_python_files
    >>> files = list_python_files(get_root_dir(), recursive=True)
    >>> return_code = run_docstrings(files, verbose=True)  # doctest: +SKIP

    """
    # disable plots from showing up
    try:
        from dstauffman.plotting import suppress_plots  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-exception-caught
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
        (failure_count, _) = doctest.testfile(file, report=True, verbose=verbose, module_relative=False)  # type: ignore[arg-type]
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
    >>> return_code = run_unittests(names)  # doctest: +SKIP

    """
    # disable plots from showing up
    try:
        from dstauffman.plotting import suppress_plots  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-exception-caught
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
    >>> return_code = run_pytests(folder)  # doctest: +SKIP

    """
    # disable plots from showing up
    try:
        from dstauffman.plotting import suppress_plots  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-exception-caught
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
    >>> return_code = run_coverage(folder)  # doctest: +SKIP

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


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_repos", exit=False)
    doctest.testmod(verbose=False)
