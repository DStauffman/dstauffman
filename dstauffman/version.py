r"""
Functions related to version history.

Notes
-----
#.  Written by David C. Stauffer in December 2020.
"""

#%% Constants
version_info = (2, 4, 3)

# Below is data about the minor release history for potential use in deprecating older support.
# For inspiration, see: https://numpy.org/neps/nep-0029-deprecation_policy.html

data = """Nov 18, 2020: dstauffman 2.0
Jan 11, 2021: dstauffman 2.1
Feb 09, 2021: dstauffman 2.2
Mar 12, 2021: dstauffman 2.3
Apr 09, 2021: dstauffman 2.4
"""

# Historical notes:
# v2.0 Supported running with only core python and started consistent versioning system.
# v2.1 Improved support for plotting datetimes, tracking versions, and using numba.
# v2.2 Split numba tools into a submodule allowing better support with and without numba.
# v2.3 Changed the way unit conversions were handled on all the plots, including some of the API.
# v2.4 Used pathlib.Path instead of strings for files and folders.