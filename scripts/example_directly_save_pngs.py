r"""
Example script for generating plots, but having a flag to disable them from being interactive.
This allows you to use one script for development, but also run it on a server that might not have
a display, and then just save the results directly to disk.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import sys

import numpy as np

#%% Parse arguments
use_display = '-nodisp' not in sys.argv
use_plotting = '-noplot' not in sys.argv

if use_plotting:
    from dstauffman.plotting import Opts, plot_time_history, Plotter
    if not use_display:
        plotter = Plotter(show=False)

#%% Main function
if __name__ == '__main__':
    #%% Parse arguments
    no_display = '-nodisp' in sys.argv
    if use_plotting and not use_display:
        print('Not showing plots interactively, only saving to disk.')

    #%% Create some fake data
    time = np.arange(0., 20., 0.01)
    data = np.vstack((np.sin(time), np.cos(time) + 2.))
    description = 'Sin and Cosine plots'

    #%% create opts
    if use_plotting:
        opts = Opts()
        opts.save_plot = True
        opts.show_plot = True and use_display

    #%% Create the plots
    if use_plotting:
        fig = plot_time_history(description, time, data, opts=opts)
