r"""
Functions related to version history.

Notes
-----
#.  Written by David C. Stauffer in December 2020.
"""

# %% Constants
version_info = (3, 8, 0)

# Below is data about the minor release history for potential use in deprecating older support.
# For inspiration, see: https://numpy.org/neps/nep-0029-deprecation_policy.html

data = """Nov 18, 2020: dstauffman 2.0
Jan 11, 2021: dstauffman 2.1
Feb 09, 2021: dstauffman 2.2
Mar 12, 2021: dstauffman 2.3
Apr 09, 2021: dstauffman 2.4
Jul 28, 2021: dstauffman 2.5
Feb 22, 2022: dstauffman 3.0
May 24, 2022: dstauffman 3.1
Aug 04, 2022: dstauffman 3.2
Aug 10, 2022: dstauffman 3.3
Jun 30, 2023: dstauffman 3.4
Oct 11, 2023: dstauffman 3.5
Jun 23, 2024: dstauffman 3.6
Nov 04, 2024: dstauffman 3.7
Apr 16, 2025: dstauffman 3.8
"""

# Historical notes:
# v2.0 Supported running with only core python and started consistent versioning system.
# v2.1 Improved support for plotting datetimes, tracking versions, and using numba.
# v2.2 Split numba tools into a submodule allowing better support with and without numba.
# v2.3 Changed the way unit conversions were handled on all the plots, including some of the API.
#      It also dropped support of Python 3.7 and now supports only 3.8+
# v2.4 Used pathlib.Path instead of strings for files and folders.
# v2.5 Added significant GPS and orbit ephemeris routines.
# v3.0 Uses fig_ax_factory for creating figures where needed and switched to qtpy instead of only PyQt5.
#      Split the slog portions of the code into a separate sub-repository.
# v3.1 Added support for using poetry as your dependency manager.
# v3.2 Change the way capture_output works internally and moved it to only be within slog.
# v3.3 Move nubs and slog to be their own module level dependencies.
# v3.4 Removed the option to save as pickle files, using only HDF5 instead.
# v3.5 Remove unneeded dependencies on requests/urllib3 and keep in sync with other work.
# v3.6 Only support Python v3.10+ using newer typing syntax for dict, tuple, list and | for Union.
# v3.7 Support keras (tensorflow/torch/jax) for quaternion routines. Add Python v3.13 support.
# v3.8 Ditch poetry entirely, use setuptools (and uv instead)
