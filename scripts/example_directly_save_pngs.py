r"""
Example script for generating plots, but having a flag to disable them from being interactive.
This allows you to use one script for development, but also run it on a server that might not have
a display, and then just save the results directly to disk.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import numpy as np

from dstauffman import process_command_line_options

flags = process_command_line_options()

if flags.use_plotting:
    from dstauffman.plotting import Opts, plot_time_history

#%% Main function
if __name__ == '__main__':
    #%% Create some fake data
    time = np.arange(0., 20., 0.01)
    data = np.vstack((np.sin(time), np.cos(time) + 2.))
    description = 'Sin and Cosine plots'

    #%% create opts
    if flags.use_plotting:
        opts = Opts()
        opts.save_plot = True
        opts.show_plot &= flags.use_display

    #%% Create the plots
    if flags.use_plotting:
        fig = plot_time_history(description, time, data, opts=opts)
