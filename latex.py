# -*- coding: utf-8 -*-
r"""
Support functions used to create LaTeX documentation.

Notes
-----
#.  Written by David C. Stauffer in Jan 2015, moved to separate file in Jan 2017.
"""
#%% Imports
import doctest
import unittest

#%% Functions - make_preamble
def make_preamble(caption, label, cols, size=r'\small', *, use_mini=False, short_cap=None):
    r"""
    Writes the table header and preamble.

    Parameters
    ----------
    caption : str
        Table caption
    label : str
        LaTeX reference label for table
    cols : str
        LaTeX string describing columns
    size : str, optional, from {r'\tiny', r'\scriptsize', r'\footnotesize', r'\small',
        r'\normalsize', r'\large', r'\Large', r'\LARGE', r'\huge', r'\Huge'}
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
    >>> out = make_preamble('Table Caption', 'tab:this_label', 'lcc')
    >>> print(out) # doctest: +ELLIPSIS
    ['\\begin{table}[H]', '    \\small', '    \\centering', '    \\caption{Table Caption}%', ...

    """
    # check that size is valid
    assert size in {r'\tiny', r'\scriptsize', r'\footnotesize', r'\small', r'\normalsize',
    r'\large', r'\Large', r'\LARGE', r'\huge', r'\Huge'}
    # create caption string
    if short_cap is None:
        cap_str = r'    \caption{' + caption + r'}%'
    else:
        cap_str = r'    \caption[' + short_cap + r']{' + caption + r'}%'
    # build table based on minipage or not
    if not use_mini:
        out = [r'\begin{table}[H]', '    '+size, r'    \centering', cap_str, \
            r'    \label{' + label + r'}', r'    \begin{tabular}{' + cols + r'}', r'        \toprule']
    else:
        out = [r'\begin{table}[H]', '    '+size, r'    \centering', cap_str, \
            r'    \label{' + label + r'}', r'    \begin{minipage}{\linewidth}', r'        \centering', \
            r'        \begin{tabular}{' + cols + r'}', r'            \toprule']
    return out

#%% Functions - make_conclusion
def make_conclusion(*, use_mini=False):
    r"""
    Writes closing tags at the end of the table.

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
        out = [r'        \bottomrule', r'    \end{tabular}', r'\end{table}', '']
    else:
        out = [r'            \bottomrule', r'        \end{tabular}', r'    \end{minipage}', r'\end{table}', '']
    return out

#%% Functions - bins_to_str_ranges
def bins_to_str_ranges(bins, dt=1, cutoff=1000):
    r"""
    Takes a given bin vector, and returns a string representation with both boundaries.

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
    for r in range(len(bins)-1):
        # alias the boundaries
        value1 = bins[r]
        value2 = bins[r+1]-dt
        # check for large values, and replace appropriately
        if value2 < cutoff:
            this_str = '{}-{}'.format(value1, value2)
        else:
            this_str = '{}+'.format(value1)
        # save this result
        out.append(this_str)
    # return everything combined as a list
    return out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_latex', exit=False)
    doctest.testmod(verbose=False)
