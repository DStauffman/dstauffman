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
from dstauffman import Opts, setup_plots, get_output_dir

#%% folder and file locations
username        = getpass.getuser()
folder          = os.path.join(r'C:\Users', username, r'Google Drive\Python\2015-16_Indoor_Scores')
xlsx_datafile   = os.path.join(folder, '2015-16 Indoor Scorecards.xlsx')

#%% opts settings for plots
opts = Opts()
opts.save_path = get_output_dir()
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

# turn interactive plotting off
plt.ioff()

#%% Tulare
# create figure
fig = plt.figure(facecolor='w')
fig.canvas.set_window_title('Tulare Score Distribution vs. Expectation')
ax = fig.add_subplot(111)

# remove plot frame lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# set ticks to only bottom and left
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# plot data
norm = ax.plot(score_range, num2per*score.normal_curve(score_range, inner10_mean, inner10_std), \
    color='#396AB1', label='Normal', linewidth=3)
acts = ax.bar(act_range, num2per*inner10_acts, color='#7293CB', label='Actuals', width=0.9)

# plot tournament scores
val = 1/len(inner10)
patches = []
for this_score in tulare: # set(tulare)
    num_this_one = tulare.count(this_score)
    label = 'Tournament' if len(patches) == 0 else ''
    this_patch = Rectangle((this_score, 0), 0.8, num2per*val*num_this_one, facecolor='#D35E60', label=label)
    patches.append(this_patch)
    ax.add_patch(this_patch)
    plt.text(this_score+0.1, 0.25, 'Session {}'.format(len(patches)), rotation=90, ha='left', va='bottom', \
        fontsize=12, color='w', fontweight='bold')

# add labels and legends
plt.xlabel('Score', fontsize=14)
plt.ylabel('Distribution [%]', fontsize=14)
plt.title(fig.canvas.get_window_title(), fontsize=20)
plt.xlim(265, 290)
(handles, labels) = ax.get_legend_handles_labels()
order = [0, 2, 1]
ax.legend([handles[i] for i in order], [labels[i] for i in order])

# add more information
plt.text(266, 14.5, 'N = 21 samples from Oct 2015 to Jan 2016', fontsize=10)
plt.text(266, 13.8, 'Inner 10 scoring', fontsize=10)
plt.text(266, 13.1, 'Mean = {:1.1f}'.format(inner10_mean), fontsize=10)
plt.text(266, 12.4, 'Std Dev = {:1.1f}'.format(inner10_std), fontsize=10)

plt.text(283, 11.0, 'Goal: 276 ave. to beat 1103 PR', fontsize=10)
plt.text(283, 10.3, 'Stretch Goal: 280 ave. for 1120', fontsize=10)
plt.text(283,  9.6, 'Result: 1084.  Failed Miserably!', fontsize=10)

# optionally save and format plot
setup_plots(fig, opts, 'dist_no_y_scale')

#%% Vegas
# create figure
fig = plt.figure(facecolor='w')
fig.canvas.set_window_title('Vegas Score Distribution vs. Expectation')
ax = fig.add_subplot(111)

# remove plot frame lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# set ticks to only bottom and left
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# plot data
ax.plot(score_range, num2per*score.normal_curve(score_range, outer10_mean, outer10_std), \
    color='#396AB1', label='Normal', linewidth=3)
ax.bar(act_range, num2per*outer10_acts, color='#7293CB', label='Actuals', width=0.9)

# plot tournament scores
val = 1/len(outer10)
patches = []
for this_score in vegas: # set(vegas) to handle duplicates
    num_this_one = vegas.count(this_score)
    label = 'Tournament' if len(patches) == 0 else ''
    this_patch = Rectangle((this_score, 0), 0.8, num2per*val*num_this_one, facecolor='#D35E60', label=label)
    patches.append(this_patch)
    ax.add_patch(this_patch)
    plt.text(this_score+0.1, 0.25, 'Day {}'.format(len(patches)), rotation=90, ha='left', va='bottom', \
        fontsize=12, color='w', fontweight='bold')

# add labels and legends
plt.xlabel('Score', fontsize=14)
plt.ylabel('Distribution [%]', fontsize=14)
plt.title(fig.canvas.get_window_title(), fontsize=20)
plt.xlim(275, 300)
(handles, labels) = ax.get_legend_handles_labels()
order = [0, 2, 1]
ax.legend([handles[i] for i in order], [labels[i] for i in order])

# add more information
plt.text(275.5, 17.5, 'N = 21 samples from Oct 2015 to Jan 2016', fontsize=10)
plt.text(275.5, 16.8, 'Outer 10 scoring', fontsize=10)
plt.text(275.5, 16.1, 'Mean = {:1.1f}'.format(outer10_mean), fontsize=10)
plt.text(275.5, 15.4, 'Std Dev = {:1.1f}'.format(outer10_std), fontsize=10)

plt.text(293, 13.0, 'Goal: 290 average for 870', fontsize=10)
plt.text(293, 12.3, 'Stretch Goal: Beat PR.', fontsize=10)
plt.text(293, 11.6, '  (Needs 292+ ave. for >876)', fontsize=10)
plt.text(293, 10.9, 'Result: 865.  Won $65.  Meh.', fontsize=10)

# optionally save and format plot
setup_plots(fig, opts, 'dist_no_y_scale')
