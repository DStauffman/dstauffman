r"""
Support functions used to create LaTeX documentation.

Notes
-----
#.  Written by David C. Stauffer in Jan 2015, moved to separate file in Jan 2017.
"""

# %% Imports
from __future__ import annotations

import doctest
from math import isinf, isnan
from typing import Sequence, TYPE_CHECKING
import unittest

from nubs import prob_to_rate_opt

from dstauffman.units import MONTHS_PER_YEAR

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    _I = NDArray[np.int_]
    _N = NDArray[np.floating]


# %% Functions - make_preamble
def make_preamble(
    caption: str,
    label: str,
    cols: str,
    size: str = r"\small",
    *,
    use_mini: bool = False,
    short_cap: str | None = None,
    numbered: bool = True,
) -> list[str]:
    r"""
    Write the table header and preamble.

    Parameters
    ----------
    caption : str
        Table caption
    label : str
        LaTeX reference label for table
    cols : str
        LaTeX string describing columns
    size : str, optional, from {r"\tiny", r"\scriptsize", r"\footnotesize", r"\small",
        r"\normalsize", r"\large", r"\Large", r"\LARGE", r"\huge", r"\Huge"}
        Size of the text within the table, default is \small
    use_mini : bool, optional, default is False
        Whether to build the table as a minipage or not
    short_cap : str, optional
        If present, used as optional caption argument for List of Tables caption.

    Returns
    -------
    out : list of str
        LaTeX text to build the table header, where each entry in the list is a row of text in the
        document

    Examples
    --------
    >>> from dstauffman import make_preamble
    >>> out = make_preamble("Table Caption", "tab:this_label", "lcc")
    >>> print(out) # doctest: +ELLIPSIS
    ['\\begin{table}[H]', '    \\small', '    \\centering', '    \\caption{Table Caption}%', ...

    """
    # check that size is valid
    assert size in {
        r"\tiny",
        r"\scriptsize",
        r"\footnotesize",
        r"\small",
        r"\normalsize",
        r"\large",
        r"\Large",
        r"\LARGE",
        r"\huge",
        r"\Huge",
    }
    # create caption string
    if short_cap is None:
        if numbered:
            cap_str = r"    \caption{" + caption + r"}%"
        else:
            cap_str = r"    \caption*{" + caption + r"}%"
    else:
        assert numbered, "Only numbered captions can have short versions."
        cap_str = r"    \caption[" + short_cap + r"]{" + caption + r"}%"
    # build table based on minipage or not
    if not use_mini:
        out = [
            r"\begin{table}[H]",
            "    " + size,
            r"    \centering",
            cap_str,
            r"    \label{" + label + r"}",
            r"    \begin{tabular}{" + cols + r"}",
            r"        \toprule",
        ]
    else:
        out = [
            r"\begin{table}[H]",
            "    " + size,
            r"    \centering",
            cap_str,
            r"    \label{" + label + r"}",
            r"    \begin{minipage}{\linewidth}",
            r"        \centering",
            r"        \begin{tabular}{" + cols + r"}",
            r"            \toprule",
        ]
    return out


# %% Functions - make_conclusion
def make_conclusion(*, use_mini: bool = False) -> list[str]:
    r"""
    Write closing tags at the end of the table.

    Parameters
    ----------
    use_mini : bool, optional, default is False
        Whether to conclude the table as part of a minipage

    Returns
    -------
    out : list of str
        LaTeX text to build the table footer, where each entry in the list is a row of text in the
        document

    Examples
    --------
    >>> from dstauffman import make_conclusion
    >>> out = make_conclusion()
    >>> print(out)
    ['        \\bottomrule', '    \\end{tabular}', '\\end{table}', '']

    """
    if not use_mini:
        out = [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    else:
        out = [r"            \bottomrule", r"        \end{tabular}", r"    \end{minipage}", r"\end{table}", ""]
    return out


# %% Functions - bins_to_str_ranges
def bins_to_str_ranges(
    bins: Sequence[int | float | str] | _I | _N, dt: int | float = 1, cutoff: int | float = 1000
) -> list[str]:
    r"""
    Take a given bin vector, and returns a string representation with both boundaries.

    Parameters
    ----------
    bins : array_like
        Boundaries for the bins
    dt : numeric scalar
        Amount to subtract from the right side boundary, default is 1
    cutoff : numeric scalar
        Value at which to consider everything above it as unbounded

    Returns
    -------
    out : list of str
        String representations of the bins

    Notes
    -----
    #.  This function works on ages, years, CD4 bins or other similar things.

    Examples
    --------
    >>> from dstauffman import bins_to_str_ranges
    >>> import numpy as np
    >>> age_bins = np.array([0, 20, 40, 60, 100000], dtype=int)
    >>> age_strs = bins_to_str_ranges(age_bins)
    >>> print(age_strs)
    ['0-19', '20-39', '40-59', '60+']

    """
    # preallocate output
    out = []
    # loop through ages
    for r in range(len(bins) - 1):
        # alias the left boundary
        left = bins[r]
        # check for string values and just pass them through
        if isinstance(left, str):
            out.append(left)
            continue
        # alias the right boundary
        right = bins[r + 1] - dt  # type: ignore[operator]
        # check for large values, and replace appropriately
        if left == right:
            this_str = f"{left:g}"
        elif right < cutoff:
            this_str = f"{left:g}-{right:g}"
        else:
            this_str = f"{left:g}+"
        # save this result
        out.append(this_str)
    # return everything combined as a list
    return out


# %% Functions - latex_str
def latex_str(
    value: int | float | str, digits: int = -1, fixed: bool = False, cmp2ar: bool = False, capped: int = 1073741823
) -> str:  # 1073741823 = 2**30-1
    r"""
    Formats a given value for display in a LaTeX document.

    Parameters
    ----------
    value : int or float (or str)
        Value, strings are passed through with underscores escaped
    digits : int, optional
        Number of digits to use in string
    fixed : bool, optional, default is False
        Whether to alway return exactly the given number of digits, or truncate if possible
    cmp2ar : bool, optional, default is False
        Whether to convert a monthly probability into an annual rate
    capped : int, optional
        Number at which anything larger is considered infinity

    Returns
    -------
    value_str : str
        String used to represent the value in LaTeX

    Notes
    -----
    #.  The capped value of 1073741823 = 2**30-1

    Examples
    --------
    >>> from dstauffman import latex_str
    >>> value = 3.14159
    >>> digits = 3
    >>> value_str = latex_str(value, digits)
    >>> print(value_str)
    3.14

    """
    # check for string case, and if so, just do replacements
    if isinstance(value, str):
        return value.replace("_", r"\_")
    # determine digit method
    letter = "f" if fixed else "g"
    # build the formatter
    formatter = "{:." + str(digits) + letter + "}" if digits >= 0 else "{}"
    # potentially convert units
    if cmp2ar:
        value = prob_to_rate_opt(value, 1 / MONTHS_PER_YEAR)  # type: ignore[assignment]
    if isnan(value):
        value_str = "NaN"
    elif isinf(value) or value > capped:
        value_str = r"$\infty$"
    else:
        # format the string
        value_str = formatter.format(value)
        # convert underscores
        value_str.replace("_", r"\_")
    return value_str


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_latex", exit=False)
    doctest.testmod(verbose=False)
