"""Script to run the GND plots with different time and opts combinations."""

#%% Imports
from copy import deepcopy
import datetime

import numpy as np

from dstauffman import convert_date, get_output_dir
from dstauffman.aerospace import Kf,  quat_from_euler, quat_mult, quat_norm
from dstauffman.plotting import Opts, plot_attitude, plot_position, plot_innovations, plot_covariance, \
    plot_los, plot_states, save_figs_to_pdf

#%% Flags
plots = {}
plots['att'] = True
plots['pos'] = False
plots['inn'] = False
plots['cov'] = False
plots['los'] = False
plots['sts'] = False

#%% Initializations
q1 = quat_norm(np.array([0.1, -0.2, 0.3, 0.4]))
dq = quat_from_euler(1e-6*np.array([-300, 100, 200]), [3, 1, 2])
q2 = quat_mult(dq, q1)

date_zero = datetime.datetime(2020, 12, 19, 14, 20, 0)

num_points = 11
num_states = 6
num_axes   = 2
num_innovs = 11

#%% KF1
kf1      = Kf()
kf1.name = 'KF1'
kf1.time = np.arange(11)
kf1.att  = np.tile(q1[:, np.newaxis], (1, kf1.time.size))  # type: ignore[union-attr]
kf1.pos  = 1e6 * np.random.rand(3, 11)
kf1.vel  = 1e3 * np.random.rand(3, 11)
kf1.covar  = 1e-6 * np.tile(np.arange(1, num_states+1, dtype=float)[:, np.newaxis], (1, num_points))
kf1.active = np.array([1, 2, 3, 4, 8, 12])

kf1.innov.name  = 'Sensor 1'
kf1.innov.units = 'm'
kf1.innov.time  = np.arange(num_innovs, dtype=float)
kf1.innov.innov = 1e-6 * np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)
kf1.innov.norm  = np.ones((num_axes, num_innovs)) * np.sign(np.random.rand(num_axes, num_innovs) - 0.5)

#%% KF2
kf2      = Kf()
kf2.name = 'KF2'
kf2.time = np.arange(2, 13)
kf2.att  = np.tile(q2[:, np.newaxis], (1, kf2.time.size))  # type: ignore[union-attr]
kf2.att[3,4] += 50e-6  # type: ignore[index]
kf2.att = quat_norm(kf2.att)  # type: ignore[arg-type]
kf2.pos  = kf1.pos[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e5  # type: ignore[index]
kf2.vel  = kf1.vel[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 100  # type: ignore[index]
kf2.covar  = kf1.covar + 1e-9 * np.random.rand(*kf1.covar.shape)  # type: ignore[union-attr]
kf2.active = kf1.active

ix              = np.hstack((np.arange(7), np.arange(8, num_innovs)))
kf2.innov.name  = 'Sensor 1'
kf2.innov.time  = kf1.innov.time[ix]
kf2.innov.innov = kf1.innov.innov[:, ix] + 1e-8 * np.random.rand(num_axes, ix.size)
kf2.innov.norm  = kf1.innov.norm[:, ix] + 0.1 * np.random.rand(num_axes, ix.size)

#%% Opts
opts1 = Opts()
opts1.case_name = 'Test 1: sec-sec'
opts1.quat_comp = True
opts1.sub_plots = True
opts1.date_zero = date_zero
opts1.rms_xmin  = 4
opts1.rms_xmax  = 20

opts2 = Opts(opts1).convert_dates('numpy')
opts2.case_name = 'Test 2: dates-sec'

opts3 = Opts(opts1)
opts3.case_name = 'Test 3: sec-dates'

opts4 = Opts(opts2)
opts4.case_name = 'Test 4: dates-dates'

#%% Copies
kd1 = deepcopy(kf1)
kd1.time = convert_date(kf1.time, 'numpy', date_zero=date_zero)
kd1.innov.time = convert_date(kf1.innov.time, 'numpy', date_zero=date_zero)
kd2 = deepcopy(kf2)
kd2.time = convert_date(kf2.time, 'numpy', date_zero=date_zero)
kd2.innov.time = convert_date(kf2.innov.time, 'numpy', date_zero=date_zero)

#%% Plots
if plots['att']:
    f1 = plot_attitude(kf1, kf2, opts=opts1)
    f2 = plot_attitude(kd1, kd2, opts=opts2, second_units=('mrad', 1e3))
    f3 = plot_attitude(kf1, kf2, opts=opts3, leg_scale='milli')
    f4 = plot_attitude(kd1, kd2, opts=opts4, leg_scale='milli', second_units=('nrad', 1e9))

if plots['pos']:
    f1 = plot_position(kf1, kf2, opts=opts1)
    f2 = plot_position(kd1, kd2, opts=opts2, second_units=('Mm', 1e-6))
    f3 = plot_position(kf1, kf2, opts=opts3, leg_scale='mega')
    f4 = plot_position(kd1, kd2, opts=opts4, leg_scale='milli', second_units=('Mm', 1e-6))

if plots['inn']:
    f1 = plot_innovations(kf1.innov, kf2.innov, opts=opts1)
    f2 = plot_innovations(kd1.innov, kd2.innov, opts=opts2, second_units=('mm', 1e3))
    f3 = plot_innovations(kf1.innov, kf2.innov, opts=opts3, leg_scale='milli')
    f4 = plot_innovations(kd1.innov, kd2.innov, opts=opts4, leg_scale='milli', second_units=('nm', 1e9))

if plots['cov']:
    f1 = plot_covariance(kf1, kf2, opts=opts1)
    f2 = plot_covariance(kd1, kd2, opts=opts2, second_units=('mrad', 1e3))
    f3 = plot_covariance(kf1, kf2, opts=opts3, leg_scale='milli')
    f4 = plot_covariance(kd1, kd2, opts=opts4, leg_scale='milli', second_units=('nrad', 1e9))

if plots['los']:
    f = plot_los(kf1, kf2, opts=opts2, leg_scale='milli', second_units='micro')
if plots['sts']:
    f = plot_states(kd1, kd2, opts=opts1, leg_scale='mill', second_units=('nrad', 1e9))

# Test PDF saving
save_figs_to_pdf(f1 + f2 + f3 + f4, filename=get_output_dir() / 'GND_plots.pdf')
