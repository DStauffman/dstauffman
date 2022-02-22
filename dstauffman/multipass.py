r"""
Functions that make it easier to deal with the multiprocessing library.

Notes
-----
#.  Written by David C. Stauffer in November 2020.
"""

#%% Imports
from __future__ import annotations
import doctest
import logging
import multiprocessing
import sys
import traceback
from typing import Any, Callable, Iterable, List, Optional, Type, TYPE_CHECKING
import unittest
import warnings

from dstauffman.enums import LogLevel

if TYPE_CHECKING:
    from types import TracebackType

#%% Activate exception support for parallel code
try:
    import tblib.pickling_support
except ModuleNotFoundError:
    warnings.warn('tblib not found, so parallelized tracebacks will not work.')
else:
    # TODO: is there a downside to always doing this?  Should I rely on the scripts that call it instead?
    tblib.pickling_support.install()

#%% Globals
logger = logging.getLogger(__name__)

#%% Classes - MultipassExceptionWrapper
class MultipassExceptionWrapper(object):
    r"""Exception wrapper that can pass through multiprocessing calls and back to main."""

    def __init__(self, ee: Type[BaseException]):
        # save exception
        self.ee = ee
        # save traceback
        # Note: sys.exc_info: Union[Tuple[None, None, None], Tuple[Type[BaseException], Any, TracebackType]]
        self.tb: Optional[TracebackType] = sys.exc_info()[2]

    def re_raise(self) -> None:
        r"""Re-raise a previously saved exception and traceback."""
        raise self.ee.with_traceback(self.tb)  # type: ignore[call-arg, misc, type-var]


#%% Functions - parfor_wrapper
def parfor_wrapper(
    func: Callable,
    args: Iterable,
    *,
    results: Any = None,
    use_parfor: bool = True,
    max_cores: Optional[int] = -1,
    ignore_errors: bool = False
) -> Any:
    r"""
    Wrapper function for the code that you want to run in a parallelized fashion.

    Parameters
    ----------
    func : callable
        The model you want to run
    args : iterable
        The arguments to the model
    results : Any, optional, defaults to list
        Optional class to use to accumulate results, must have append method
    use_parfor : bool, optional
        Whether to run parallelized or not
    max_cores : int, optional
        Maximum number of cores to use
    ignore_errors : bool, optional, default is False
        Flag to collect and log, but otherwise ignore errors and continue on where possible

    Returns
    -------
    results : list of Any
        List of outputs from model

    Notes
    -----
    #.  Written by David C. Stauffer in November 2020.

    Examples
    --------
    >>> from dstauffman import parfor_wrapper
    >>> import numpy as np
    >>> def func(x, y):
    ...     return x + np.sin(x) + np.cos(y*2)
    >>> temp = np.linspace(0, 2*np.pi)
    >>> args = ((temp.copy(), temp.copy()), (temp + 2*np.pi, temp.copy()), (temp + 4*np.pi, temp.copy()))
    >>> max_cores = None
    >>> results = parfor_wrapper(func, args, max_cores=max_cores)
    >>> print(len(results))
    3

    """
    # initialize results
    if results is None:
        results = []
    # calculate the number of cores to use
    if max_cores is None:
        num_cores = 1
    elif max_cores == -1:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = min((multiprocessing.cpu_count(), max_cores))
    errors: List[MultipassExceptionWrapper] = []
    if use_parfor and num_cores > 1:
        # parallel loop
        with multiprocessing.get_context('spawn').Pool(num_cores) as pool:
            temp_results = list(pool.starmap(func, args))
        for result in temp_results:
            if isinstance(result, MultipassExceptionWrapper):
                if ignore_errors:
                    errors.append(result)
                else:
                    result.re_raise()
            else:
                results.append(result)
    else:
        # standard (non-parallel) loop
        it = iter(args)
        while True:
            try:
                this_args = next(it)
                result = func(*this_args)
                if isinstance(result, MultipassExceptionWrapper):
                    if ignore_errors:
                        errors.append(result)
                    else:
                        result.re_raise()
                else:
                    results.append(result)
            except StopIteration:
                break
    if ignore_errors and len(errors) > 0:
        logger.log(LogLevel.L2, 'There were %i error(s) in the processing.', len(errors))
        for (i, err) in enumerate(errors):
            logger.log(
                LogLevel.L6,
                'Error %i: %s\n%s',
                i + 1,
                err.ee.with_traceback(err.tb),  # type: ignore[call-arg, type-var]
                '\n'.join(traceback.format_tb(err.tb)),
            )
    return results


#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_multipass', exit=False)
    doctest.testmod(verbose=False)
