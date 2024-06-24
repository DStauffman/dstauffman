r"""
Example script for running the Batch Parameter Estimation (BPE) portion of the DStauffman code.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""  # pylint: disable=redefined-outer-name

# %% Imports
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

import slog as lg

import dstauffman as dcs
import dstauffman.estimation as estm
import dstauffman.plotting as plot

if TYPE_CHECKING:
    _I = np.typing.NDArray[np.int_]
    _N = np.typing.NDArray[np.float64]


# %% Classes - SimParams
class SimParams(dcs.Frozen):
    r"""Simulation model parameters."""

    def __init__(self, time: _N, *, magnitude: float, frequency: float, phase: float):
        self.time = time
        self.magnitude = magnitude
        self.frequency = frequency
        self.phase = phase

    def __eq__(self, other: Any) -> bool:
        r"""Allows classes to compare contents to determine if equal."""
        if not isinstance(other, SimParams):
            return False
        for key in vars(self):
            if np.any(getattr(self, key) != getattr(other, key)):
                return False
        return True

    def __repr__(self) -> str:
        r"""Detailed string representation."""
        return f"mag={self.magnitude}, freq={self.frequency}, phs={self.phase}"


# %% Functions - _get_truth_index
def _get_truth_index(results_time: _N, truth_time: _N) -> tuple[_I, _I]:
    r"""Find the indices to the truth data from the results time."""
    # Hard-coded values
    precision = 1e-7
    # find the indices to truth
    ix_truth = np.flatnonzero((truth_time >= results_time[0] - precision) & (truth_time <= results_time[-1] + precision))
    # find the indices to results (in case truth isn't long enough)
    ix_results = np.flatnonzero(results_time <= truth_time[-1] + precision)
    # return the indices
    return (ix_truth, ix_results)


# %% Functions - sim_model
def sim_model(sim_params: SimParams) -> _N:
    r"""Run the simple example simulation model."""
    return sim_params.magnitude * np.sin(  # type: ignore[no-any-return]
        2 * np.pi * sim_params.frequency * sim_params.time / 1000 + sim_params.phase * np.pi / 180
    )


# %% Functions - truth
def truth(time: _N, magnitude: float = 5.0, frequency: float = 10.0, phase: float = 90.0) -> _N:
    r"""Return true values for simple example truth data."""
    return magnitude * np.sin(2 * np.pi * frequency * time / 1000 + phase * np.pi / 180)  # type: ignore[no-any-return]


# %% Functions - cost_wrapper
def cost_wrapper(  # pylint: disable=unused-argument
    results_data: _N, *, results_time: _N, truth_time: _N, truth_data: _N, sim_params: SimParams
) -> _N:
    r"""Calculate innovations (cost) for the model."""
    # Pull out overlapping time points and indices
    (ix_truth, ix_results) = _get_truth_index(results_time, truth_time)
    sub_truth = truth_data[ix_truth]
    sub_result = results_data[ix_results]

    # calculate the innovations
    innovs = sub_result - sub_truth
    return innovs


# %% Functions - get_parameter
def get_parameter(sim_params: SimParams, *, names: list[str]) -> Any:
    r"""Get the model parameters."""
    num = len(names)
    values = np.full(num, np.nan)
    for ix, name in enumerate(names):
        if hasattr(sim_params, name):
            values[ix] = getattr(sim_params, name)
        else:
            raise ValueError(f'Bad parameter name: "{name}".')
    return values


# %% Functions - set_parameter
def set_parameter(sim_params: SimParams, *, names: list[str], values: list[Any]) -> None:
    r"""Set the model parameters."""
    num = len(names)
    assert len(values) == num, "Names and Values must have the same length."
    for ix, name in enumerate(names):
        if hasattr(sim_params, name):
            setattr(sim_params, name, values[ix])
        else:
            raise ValueError(f'Bad parameter name: "{name}".')


# %% Script
if __name__ == "__main__":
    # Constants
    # fmt: off
    rerun      = True
    make_plots = True
    time       = np.arange(251.0)
    # fmt: on

    # Parameters
    sim_params = SimParams(time, magnitude=3.5, frequency=12.0, phase=180.0)

    # Truth data
    # fmt: off
    truth_time = np.arange(-10.0, 201.0)
    truth_data = truth(truth_time)
    truth      = plot.TruthPlotter(truth_time, truth_data)  # type: ignore[assignment]
    # fmt: on

    # Logger
    lg.activate_logging(lg.LogLevel.L8)

    # BPE Settings
    # fmt: off
    opti_opts                = estm.OptiOpts()
    opti_opts.model_func     = sim_model
    opti_opts.model_args     = {"sim_params": sim_params}
    opti_opts.cost_func      = cost_wrapper
    opti_opts.cost_args = {"results_time": time, "truth_time": truth_time, "truth_data": truth_data}
    opti_opts.get_param_func = get_parameter
    opti_opts.set_param_func = set_parameter
    opti_opts.output_folder  = dcs.get_output_dir() / datetime.datetime.now().strftime("%Y-%m-%d")
    opti_opts.output_results = Path("bpe_results.hdf5")
    opti_opts.params         = []

    # less common optimization settings
    opti_opts.slope_method    = "one_sided"  # or "two_sided"
    opti_opts.is_max_like     = False
    opti_opts.max_iters       = 10
    opti_opts.tol_cosmax_grad = 1e-4
    opti_opts.tol_delta_step  = 1e-7
    opti_opts.tol_delta_cost  = 1e-8
    opti_opts.step_limit      = 5
    opti_opts.x_bias          = 0.8
    opti_opts.grow_radius     = 2
    opti_opts.shrink_radius   = 0.5
    opti_opts.trust_radius    = 1.0
    opti_opts.max_cores       = None  # None means no parallelization, -1 means use all cores

    # Parameters to estimate
    opti_opts.params.append(estm.OptiParam("magnitude", best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
    opti_opts.params.append(estm.OptiParam("frequency", best=20, min_=1, max_=1000, typical=60, minstep=0.01))
    opti_opts.params.append(estm.OptiParam("phase", best=180, min_=0, max_=360, typical=100, minstep=0.1))
    # fmt: on

    # Run code
    if rerun:
        (bpe_results, results) = estm.run_bpe(opti_opts)
    else:
        bpe_results = estm.BpeResults.load(opti_opts.output_folder / opti_opts.output_results)
        results = sim_model(sim_params)  # just re-run, nothing is actually saved by this model

    # Plot results
    if make_plots:
        # build opts
        # fmt: off
        opts           = plot.Opts()
        opts.case_name = "Model Results"
        opts.save_path = dcs.get_output_dir()
        opts.save_plot = True
        # fmt: on

        # make model plots
        f0 = plot.plot_health_monte_carlo(time, results, "Output", opts=opts, truth=truth)  # type: ignore[arg-type]
        extra_plotter = lambda fig, ax: truth.plot_truth(ax[0], scale=1)  # type: ignore[attr-defined]  # pylint: disable=unnecessary-lambda-assignment
        f1 = plot.plot_time_history("Output vs. Time", time, results, opts=opts, extra_plotter=extra_plotter)

        # make BPE plots
        bpe_plots = {"innovs": True, "convergence": True, "correlation": True, "info_svd": True, "covariance": True}
        plot.plot_bpe_results(bpe_results, opts=opts, plots=bpe_plots)
