r"""
Functions that make it easier to deal with the multiprocessing library.

Notes
-----
#.  Written by David C. Stauffer in November 2020.
"""

#%% Imports
from __future__ import annotations
import doctest
import multiprocessing
import sys
from typing import Any, Callable, Iterable, Optional, Type, TYPE_CHECKING
import unittest

if TYPE_CHECKING:
    from types import TracebackType

#%% Activate exception support for parallel code
#try:
#    import tblib.pickling_support
#except ModuleNotFoundError:
#    pass
#else:
#    # TODO: is there a downside to always doing this?  Should I rely on the scripts that call it instead?
#    tblib.pickling_support.install()

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
        raise self.ee.with_traceback(self.tb)  # type: ignore[arg-type, call-arg, misc, type-var]

#%% Functions - parfor_wrapper
def parfor_wrapper(func: Callable, args: Iterable, *, results: Any = None, use_parfor: bool = True, \
        max_cores: int = -1) -> Any:
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
    if use_parfor and num_cores > 1:
        # parallel loop
        with multiprocessing.get_context('spawn').Pool(num_cores) as pool:
            temp_results = list(pool.starmap(func, args))
        for result in temp_results:
            if isinstance(result, MultipassExceptionWrapper):
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
                    result.re_raise()
                else:
                    results.append(result)
            except StopIteration:
                break
    return results

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_multipass', exit=False)
    doctest.testmod(verbose=False)
