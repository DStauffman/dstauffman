# -*- coding: utf-8 -*-
r"""
Example script for running the Batch Parameter Estimation (BPE) portion of the DStauffman code.

Notes
-----
#.  Written by David C. Stauffer in May 2015.
"""
# pylint: disable=E1101, C0103, C0326

#%% Imports
from datetime import datetime
import numpy as np
import os
import dstauffman as dcs

#%% Classes - SimParams
class SimParams(dcs.Frozen):
    r"""Simulation model parameters."""
    def __init__(self, time, *, magnitude, frequency, phase):
        self.time      = time
        self.magnitude = magnitude
        self.frequency = frequency
        self.phase     = phase

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for key in vars(self):
            if np.any(getattr(self, key) != getattr(other, key)):
                return False
        return True

    def __repr__(self):
        return 'mag={}, freq={}, phs={}'.format(self.magnitude, self.frequency, self.phase)

#%% Functions - _get_truth_index
def _get_truth_index(results_time, truth_time):
    r"""
    Finds the indices to the truth data from the results time.
    """
    # Hard-coded values
    precision    = 1e-7
    # find the indices to truth
    ix_truth     = np.nonzero((truth_time >= results_time[0] - precision) & (truth_time <= \
        results_time[-1] + precision))[0]
    # find the indices to results (in case truth isn't long enough)
    ix_results   = np.nonzero(results_time <= truth_time[-1] + precision)[0]
    # return the indices
    return (ix_truth, ix_results)

#%% Functions - sim_model
def sim_model(sim_params):
    r"""Simple example simulation model."""
    return sim_params.magnitude * np.sin(2*np.pi*sim_params.frequency*sim_params.time/1000 + \
        sim_params.phase*np.pi/180)

#%% Functions - truth
def truth(time, magnitude=5, frequency=10, phase=90):
    r"""Simple example truth data."""
    return magnitude * np.sin(2*np.pi*frequency*time/1000 + phase*np.pi/180)

#%% Functions - cost_wrapper
def cost_wrapper(results_data, *, results_time, truth_time, truth_data, sim_params):
    r"""Example Cost wrapper for the model."""
    # Pull out overlapping time points and indices
    (ix_truth, ix_results) = _get_truth_index(results_time, truth_time)
    sub_truth  = truth_data[ix_truth]
    sub_result = results_data[ix_results]

    # calculate the innovations
    innovs = sub_result - sub_truth
    return innovs

#%% Functions - get_parameter
def get_parameter(sim_params, *, names):
    r"""Simple example parameter getter."""
    num = len(names)
    values = np.nan * np.ones(num)
    for (ix, name) in enumerate(names):
        if hasattr(sim_params, name):
            values[ix] = getattr(sim_params, name)
        else:
            raise ValueError('Bad parameter name: "{}".'.format(name))
    return values

#%% Functions - set_parameter
def set_parameter(sim_params, *, names, values):
    r"""Simple example parameter setter."""
    num = len(names)
    assert len(values) == num, 'Names and Values must have the same length.'
    for (ix, name) in enumerate(names):
        if hasattr(sim_params, name):
            setattr(sim_params, name, values[ix])
        else:
            raise ValueError('Bad parameter name: "{}".'.format(name))

#%% Script
if __name__=='__main__':
    # Constants
    rerun      = True
    make_plots = True
    time       = np.arange(251)

    # Parameters
    sim_params = SimParams(time, magnitude=3.5, frequency=12, phase=180)

    # Truth data
    truth_time = np.arange(-10, 201)
    truth_data = truth(truth_time)
    truth = dcs.TruthPlotter(truth_time, truth_data)

    # Logger
    dcs.Logger().set_level(10)

    # BPE Settings
    opti_opts = dcs.OptiOpts()
    opti_opts.model_func     = sim_model
    opti_opts.model_args     = {'sim_params': sim_params}
    opti_opts.cost_func      = cost_wrapper
    opti_opts.cost_args      = {'results_time': time, 'truth_time': truth_time, 'truth_data': truth_data}
    opti_opts.get_param_func = get_parameter
    opti_opts.set_param_func = set_parameter
    opti_opts.output_folder  = os.path.join(dcs.get_output_dir(), datetime.now().strftime('%Y-%m-%d'))
    opti_opts.output_results = 'bpe_results.hdf5'
    opti_opts.params         = []

    # less common optimization settings
    opti_opts.slope_method    = 'one_sided' # or 'two_sided'
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

    # Parameters to estimate
    opti_opts.params.append(dcs.OptiParam('magnitude', best=2.5, min_=-10, max_=10, typical=5, minstep=0.01))
    opti_opts.params.append(dcs.OptiParam('frequency', best=20, min_=1, max_=1000, typical=60, minstep=0.01))
    opti_opts.params.append(dcs.OptiParam('phase', best=180, min_=0, max_=360, typical=100, minstep=0.1))

    # Run code
    if rerun:
        (bpe_results, results) = dcs.run_bpe(opti_opts)
    else:
        bpe_results = dcs.BpeResults.load(os.path.join(opti_opts.output_folder, opti_opts.output_results))
        results     = sim_model(sim_params) # just re-run, nothing is actually saved by this model

    # Plot results
    if make_plots:
        # build opts
        opts           = dcs.Opts()
        opts.case_name = 'Model Results'
        opts.save_path = dcs.get_output_dir()
        opts.save_plot = True

        # make model plots
        dcs.plot_time_history(time, results, description='Output vs. Time', opts=opts, truth=truth)

        # make BPE plots
        bpe_plots = {'innovs': True, 'convergence': False, 'correlation': True, 'info_svd': True, \
            'covariance': False}
        dcs.plot_bpe_results(bpe_results, opti_opts, opts, plots=bpe_plots)
