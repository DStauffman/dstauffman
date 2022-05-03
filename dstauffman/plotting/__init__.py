r"""
dstauffman code related to plotting data.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Added support and generic files in May 2020.
#.  Split into separate submodule to not require matplotlib in July 2020.
"""

#%% Imports
# fmt: off
from .aerospace import make_quaternion_plot, plot_attitude, plot_los, plot_position, \
                           plot_velocity, plot_innovations, plot_innov_fplocs, plot_innov_hist, \
                           plot_covariance, plot_states
from .batch     import plot_bpe_convergence, plot_bpe_results
from .generic   import make_generic_plot, make_time_plot, make_error_bar_plot, \
                           make_difference_plot, make_categories_plot, make_bar_plot, \
                           make_connected_sets
from .health    import TruthPlotter, plot_health_time_history, plot_health_monte_carlo, \
                           plot_icer, plot_population_pyramid
from .plotting  import Opts, suppress_plots, unsuppress_plots, plot_time_history, \
                           plot_correlation_matrix, plot_bar_breakdown, plot_histogram, setup_plots
from .support   import DEFAULT_COLORMAP, DEFAULT_CLASSIFICATION, COLOR_LISTS, MyCustomToolbar, \
                           ColorMap, close_all, get_nondeg_colorlists, ignore_plot_data, whitten, \
                           get_figure_title, resolve_name, storefig, titleprefix, disp_xlimits, \
                           zoom_ylim, figmenu, rgb_ints_to_hex, get_screen_resolution, \
                           show_zero_ylim, plot_second_units_wrapper, plot_second_yunits, \
                           get_rms_indices, plot_vert_lines, plot_phases, get_classification, \
                           plot_classification, align_plots, z_from_ci, ci_from_z, \
                           save_figs_to_pdf, save_images_to_pdf, add_datashaders, fig_ax_factory
# fmt: on

#%% Unit test
if __name__ == "__main__":
    pass
