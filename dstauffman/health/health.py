r"""
Contains more complex analysis related routines mostly specific to health care modeling.

Notes
-----
#.  Written by David C. Stauffer in October 2017.
"""

# %% Imports
from __future__ import annotations

import doctest
from typing import List, Literal, Optional, overload, Tuple, TYPE_CHECKING, Union
import unittest

from dstauffman.constants import HAVE_NUMPY, HAVE_PANDAS

if HAVE_NUMPY:
    import numpy as np
if HAVE_PANDAS:
    import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from dstauffman.plotting import Opts as _Opts

    _I = NDArray[np.int_]
    _N = NDArray[np.float64]


# %% Functions - dist_enum_and_mons
@overload
def dist_enum_and_mons(
    num: int,
    distribution: ArrayLike,
    prng: np.random.RandomState,
    *,
    max_months: Literal[None] = ...,
    start_num: int,
    alpha: float,
    beta: float,
) -> _I:
    ...


@overload
def dist_enum_and_mons(
    num: int,
    distribution: ArrayLike,
    prng: np.random.RandomState,
    *,
    max_months: Union[int, _I],
    start_num: int,
    alpha: float,
    beta: float,
) -> _I:
    ...


def dist_enum_and_mons(
    num: int,
    distribution: ArrayLike,
    prng: np.random.RandomState,
    *,
    max_months: Optional[Union[int, _I]] = None,
    start_num: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Union[_I, Tuple[_I, _I]]:
    r"""
    Create a distribution for an enumerated state with a duration (such as a disease status).

    Parameters
    ----------
    num : int
        Number of people in the population
    distribution : array_like
        Likelihood of being in each state (should cumsum to 100%)
    prng : class numpy.random.RandomState
        Pseudo-random number generator
    max_months : scalar or array_like, optional
        Maximum number of months for being in each state
    start_num : int, optional
        Number to start counting from, default is 1
    alpha : float, optional
        The alpha parameter for the beta distribution
    beta : float, optional
        The beta parameter for the beta distribution

    Returns
    -------
    state : ndarray
        Enumerated status for this month for everyone in the population
    mons : ndarray
        Number of months in this state for anyone with an infection

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.
    #.  Updated by David C. Stauffer in June 2015 to use a beta curve to distribute the number of
        months spent in each state.
    #.  Made into a generic function for the dstauffman library by David C. Stauffer in July 2015.
    #.  Updated by David C. Stauffer in November 2015 to change the inputs to allow max_months and
        mons output to be optional.
    #.  Updated by David C. Stauffer in April 2017 to only return state if desired, and to allow
        distribution to be a 2D matrix, so you can have age based distributions.

    Examples
    --------
    >>> from dstauffman.health import dist_enum_and_mons
    >>> import numpy as np
    >>> num = 100
    >>> distribution = np.array([0.10, 0.20, 0.30, 0.40])
    >>> max_months = np.array([5, 100, 20, 1])
    >>> start_num = 0
    >>> prng = np.random.RandomState()
    >>> (state, mons) = dist_enum_and_mons(num, distribution, prng, max_months=max_months, start_num=start_num)

    """
    # hard-coded values
    precision = 1e-12
    # create the cumulative distribution (allows different distribution per person if desired)
    cum_dist = np.cumsum(np.atleast_2d(distribution), axis=1)
    assert np.all(np.abs(cum_dist[:, -1] - 1) < precision), "Given distribution doesn't sum to 1."
    # do a random draw based on the cumulative distribution
    state = np.sum(prng.rand(num, 1) >= cum_dist, axis=1, dtype=int) + start_num
    # set the number of months in this state based on a beta distribution with the given
    # maximum number of months in each state
    if max_months is None:
        return state  # type: ignore[no-any-return]
    if np.isscalar(max_months):
        max_months = np.full(len(distribution), max_months)  # type: ignore[arg-type]
    mons = np.ceil(max_months[state - start_num] * prng.beta(alpha, beta, num)).astype(int)  # type: ignore[index]
    return (state, mons)


# %% Functions - icer
def icer(
    cost: ArrayLike,
    qaly: ArrayLike,
    names: Optional[List[str]] = None,
    baseline: Optional[int] = None,
    make_plot: bool = False,
    opts: Optional[_Opts] = None,
) -> Tuple[_N, _N, _N, _N, pd.DataFrame, Optional[Figure]]:
    r"""
    Calculate the incremental cost effectiveness ratios with steps to throw out dominated strategies.

    Summary
    -------
        In a loop, the code sorts by cost, throws out strongly dominated strategies (qaly doesn't
        improve despite higher costs), calculates an incremental cost, qaly and cost effectiveness
        ratio, then throws out weakly dominated strategies (icer doesn't improve over cheaper options)
        and finally returns the incremental cost, qaly and ratios for the remaining "frontier" options
        along with an order variable to map them back to the inputs.

    Parameters
    ----------
    cost : (N) array_like
        Cost of each strategy
    qaly : (N) array_like
        Quality adjusted life years (QALY) gained by each strategy
    names : (N) array_like, optional
        Names of the different strategies
    baseline : int, optional
        Index of baseline strategy to use for cost comparisons, if not nan
    make_plot : bool, optional
        True/false flag for whether to plot the data
    opts : class Opts, optional
        Plotting options

    Results
    -------
    inc_cost  : (M) ndarray
        incremental costs - see note 1
    inc_qaly  : (M) ndarray
        incremental QALYs gained
    icer_out  : (M) ndarray
        incremental cost effectiveness ratios
    order     : (N) ndarray
        order mapping to the original inputs, with NaNs for dominated strategies
    icer_data : (N) pandas dataframe
        ICER data as a pandas dataframe
    fig       : (object) figure handle or None
        Figure handle for any figure that was produced

    Notes
    -----
    #.  N may be smaller than M due to dominated strategies being removed.  The order variable
            will have (M - N) values set to NaN.

    Examples
    --------
    >>> from dstauffman.health import icer
    >>> cost = [250e3, 750e3, 2.25e6, 3.75e6]
    >>> qaly = [20., 30, 40, 80]
    >>> (inc_cost, inc_qaly, icer_out, order, icer_data, fig) = icer(cost, qaly)
    >>> print(inc_cost) # doctest: +NORMALIZE_WHITESPACE
    [ 250000. 500000. 3000000.]

    >>> print(inc_qaly) # doctest: +NORMALIZE_WHITESPACE
    [20. 10. 50.]

    >>> print(icer_out) # doctest: +NORMALIZE_WHITESPACE
    [12500. 50000. 60000.]

    >>> print(order) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 1. nan 2.]

    """
    # force inputs to be ndarrays
    cost = np.atleast_1d(np.asarray(cost))
    qaly = np.atleast_1d(np.asarray(qaly))
    fig = None

    # check inputs
    assert np.all(cost > 0), "Costs must be positive."  # type: ignore[operator]
    assert np.all(qaly > 0), "Qalys must be positive."  # type: ignore[operator]
    assert cost.shape == qaly.shape, "Cost and Qalys must have same size."  # type: ignore[union-attr]
    assert cost.size > 0, "Costs and Qalys cannot be empty."  # type: ignore[union-attr]

    # alias the number of strategies
    num = cost.size  # type: ignore[union-attr]

    # build an index order variable to keep track of strategies
    keep = list(range(num))

    # enter processing loop
    while True:
        # pull out current values based on evolving order mask
        this_cost = cost[keep]  # type: ignore[call-overload, index]
        this_qaly = qaly[keep]  # type: ignore[call-overload, index]

        # sort by cost
        ix_sort = np.argsort(this_cost)
        sorted_cost = this_cost[ix_sort]  # type: ignore[index]
        sorted_qaly = this_qaly[ix_sort]  # type: ignore[index]

        # check for strongly dominated strategies
        if not np.all(np.diff(sorted_qaly) >= 0):
            # find the first occurence (increment by one to find the one less effective than the last)
            bad = np.flatnonzero(np.diff(sorted_qaly) < 0) + 1
            if len(bad) == 0:
                raise ValueError("Index should never be empty, something unexpected happended.")  # pragma: no cover
            # update the mask and continue to next pass of while loop
            keep.pop(ix_sort[bad[0]])
            continue

        # calculate incremental costs
        inc_cost = np.hstack([sorted_cost[0], np.diff(sorted_cost)])  # type: ignore[index, list-item]
        inc_qaly = np.hstack([sorted_qaly[0], np.diff(sorted_qaly)])  # type: ignore[index, list-item]
        icer_out = inc_cost / inc_qaly

        # check for weakly dominated strategies
        if not np.all(np.diff(icer_out) >= 0):
            # find the first bad occurence
            bad = np.flatnonzero(np.diff(icer_out) < 0)
            if len(bad) == 0:
                raise ValueError("Index should never be empty, something unexpected happended.")  # pragma: no cover
            # update mask and continue to next pass
            keep.pop(ix_sort[bad[0]])
            continue

        # if no continue statements were reached, then another iteration is not necessary, so break out
        break

    # save the final ordering
    order = np.full(cost.shape, np.nan)  # type: ignore[union-attr]
    order[keep] = ix_sort

    # build an index to pull data out
    temp = np.flatnonzero(~np.isnan(order))
    ix = temp[order[~np.isnan(order)].astype(int)]

    # recalculate based on given baseline
    if baseline is not None:
        inc_cost = np.diff(np.hstack((cost[baseline], cost[ix])))  # type: ignore[arg-type, index]
        inc_qaly = np.diff(np.hstack((qaly[baseline], qaly[ix])))  # type: ignore[arg-type, index]
        icer_out = np.divide(inc_cost, inc_qaly, out=np.full(inc_cost.shape, np.nan), where=inc_qaly != 0)

    # output as dataframe
    # build a name list if not given
    if names is None:
        names = [f"Strategy {i + 1}" for i in range(num)]
    # preallocate some variables
    full_inc_costs = np.full((num), np.nan)
    full_inc_qalys = np.full((num), np.nan)
    full_icers = np.full((num), np.nan)
    # fill the calculations in where applicable
    full_inc_costs[ix] = inc_cost
    full_inc_qalys[ix] = inc_qaly
    full_icers[ix] = icer_out
    # make into dictionary with more explicit column names
    data: dict[str, list[str] | _N] = {}
    data["Strategy"] = names
    data["Cost"] = cost  # type: ignore[assignment]
    data["QALYs"] = qaly  # type: ignore[assignment]
    data["Increment_Costs"] = full_inc_costs
    data["Incremental_QALYs"] = full_inc_qalys
    data["ICER"] = full_icers
    data["Order"] = order

    # make the whole data set into a dataframe
    icer_data = pd.DataFrame.from_dict(data)
    icer_data.set_index("Strategy", inplace=True)

    # Make a plot
    if make_plot:
        # delayed import to eliminate circular imports
        from dstauffman.plotting import plot_icer  # pylint: disable=import-outside-toplevel

        fig = plot_icer(qaly, cost, ix, baseline=baseline, names=names, opts=opts)  # type: ignore[arg-type]

    return (inc_cost, inc_qaly, icer_out, order, icer_data, fig)


# %% Unit test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.ioff()
    unittest.main(module="dstauffman.tests.test_health_health", exit=False)
    doctest.testmod(verbose=False)
