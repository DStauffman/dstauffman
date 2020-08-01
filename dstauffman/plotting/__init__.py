r"""
dstauffman code related to plotting data.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Added support and generic files in May 2020.
#.  Split into separate submodule to not require matplotlib in July 2020.
"""

#%% Imports
from .aerospace import make_quaternion_plot, plot_attitude, plot_los, plot_position, \
                           plot_velocity, plot_innovations, plot_covariance, plot_states
from .batch     import plot_bpe_convergence, plot_bpe_results
from .generic   import make_time_plot, make_error_bar_plot, make_difference_plot, \
                           make_categories_plot
from .health    import plot_health_time_history, plot_health_monte_carlo, plot_icer, \
                           plot_population_pyramid
from .plotting  import Opts, plot_time_history, plot_correlation_matrix, plot_bar_breakdown
from .support   import DEFAULT_COLORMAP, DEFAULT_CLASSIFICATION, Plotter, TruthPlotter, \
                           MyCustomToolbar, ColorMap, close_all, get_color_lists, \
                           ignore_plot_data, whitten, resolve_name, storefig, titleprefix, \
                           disp_xlimits, zoom_ylim, setup_plots, figmenu, rgb_ints_to_hex, \
                           get_screen_resolution, show_zero_ylim, plot_second_units_wrapper, \
                           plot_second_yunits, get_rms_indices, plot_vert_lines, plot_phases, \
                           get_classification, plot_classification, align_plots, z_from_ci

#%% Unittest
if __name__ == '__main__':
    pass
