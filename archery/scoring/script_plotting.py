# -*- coding: utf-8 -*-
# script_plotting makes the plots for Tulare and Vegas scores relative to season standard deviations

#%% Imports
import getpass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import pandas as pd
import dstauffman.archery.scoring as score
from dstauffman import Opts, setup_plots

#%% folder and file locations
username        = getpass.getuser()
folder          = os.path.join(r'C:\Users', username, r'Google Drive\Python\2015-16_Indoor_Scores')
xlsx_datafile   = os.path.join(folder, '2015-16 Indoor Scorecards.xlsx')

#%% opts settings for plots
opts = Opts()
opts.save_path = folder
opts.save_plot = True
opts.plot_type = 'png'

#%% process data
(scores, names, dates) = score.read_from_excel_datafile(xlsx_datafile)
(nfaa_score, usaa_score) = score.convert_data_to_scores(scores)

#%% Specific dates
inner10 = []
outer10 = []
tulare  = []
vegas   = []
for (ix, this_name) in enumerate(names):
    if 'Tulare' in this_name:
        tulare.append(usaa_score[ix])
    elif 'Vegas' in this_name:
        vegas.append(nfaa_score[ix])
    else:
        inner10.append(usaa_score[ix])
        outer10.append(nfaa_score[ix])
# manually add missing Vegas scores for now
vegas = [284, 289] + vegas

#%% make plots
# hard-coded values
num2per       = 100
perfect_score = 300
PLOT_LIMITS   = [250, 300]

# calculate mean and standard deviations, use pandas Series instead of numpy for N-1 definition of std.
outer10_mean   = np.mean(outer10)
inner10_mean   = np.mean(inner10)
outer10_std    = pd.Series(outer10).std()
inner10_std    = pd.Series(inner10).std()

# create score range to evaluate for plotting
dt          = 0.1
score_range = np.arange(0, perfect_score+dt, dt)

# create actuals for scores
act_range   = np.arange(PLOT_LIMITS[0], PLOT_LIMITS[1]+1)
outer10_acts   = np.empty(act_range.shape)
inner10_acts   = np.empty(act_range.shape)
num_scores  = len(outer10)
for (ix, this_score) in enumerate(act_range):
    outer10_acts[ix] = np.sum(outer10 == this_score) / num_scores
    inner10_acts[ix] = np.sum(inner10 == this_score) / num_scores

#%% Tulare
# create figure
fig = plt.figure()
fig.canvas.set_window_title('Score Distribution')
ax = fig.add_subplot(111)

# plot data
norm = ax.plot(score_range, num2per*score.normal_curve(score_range, inner10_mean, inner10_std), color='#396AB1', label='Normal')
acts = ax.bar(act_range, num2per*inner10_acts, color='#7293CB', label='Actuals')

# plot tournament scores
val = 1/len(inner10)
patches = []
for this_score in set(tulare):
    num_this_one = tulare.count(this_score)
    label = 'Tournament' if len(patches) == 0 else ''
    this_patch = Rectangle((this_score, 0), 0.6, num2per*val*num_this_one, facecolor='#D35E60', label=label)
    patches.append(this_patch)
    ax.add_patch(this_patch)

# add labels and legends
plt.xlabel('Score')
plt.ylabel('Distribution [%]')
plt.title(fig.canvas.get_window_title())
plt.xlim(265, 290)
plt.legend()
plt.grid(True)

# optionally save and format plot
opts.case_name = 'Tulare'
setup_plots(fig, opts, 'dist_no_y_scale')
plt.show(block=False)

#%% Vegas
# create figure
fig = plt.figure()
fig.canvas.set_window_title('Score Distribution')
ax = fig.add_subplot(111)

# plot data
ax.plot(score_range, num2per*score.normal_curve(score_range, outer10_mean, outer10_std), color='#396AB1', label='Normal')
ax.bar(act_range, num2per*outer10_acts, color='#7293CB', label='Actuals')

# plot tournament scores
val = 1/len(outer10)
patches = []
for this_score in set(vegas):
    num_this_one = vegas.count(this_score)
    label = 'Tournament' if len(patches) == 0 else ''
    this_patch = Rectangle((this_score, 0), 0.6, num2per*val*num_this_one, facecolor='#D35E60', label=label)
    patches.append(this_patch)
    ax.add_patch(this_patch)

# add labels and legends
plt.xlabel('Score')
plt.ylabel('Distribution [%]')
plt.title(fig.canvas.get_window_title())
plt.xlim(275, 300)
plt.legend()
plt.grid(True)

# optionally save and format plot
opts.case_name = 'Vegas'
setup_plots(fig, opts, 'dist_no_y_scale')
plt.show(block=False)
