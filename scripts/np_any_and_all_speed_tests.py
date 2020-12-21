r"""Runs speed comparisons for the any and all commands."""

#%% Imports
from IPython import get_ipython
import numpy as np
import pandas as pd

from dstauffman import np_any, np_all

#%% Results
text = \
r"""i = None, any(x)
111 ns ± 1.81 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
i = None, np.any(x)
3.74 µs ± 47.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = None, np_any(x)
342 ns ± 2.11 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 0, any(x)
116 ns ± 3.65 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
i = 0, np.any(x)
6.55 µs ± 58.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 0, np_any(x)
339 ns ± 3.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 10, any(x)
255 ns ± 5.41 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 10, np.any(x)
6.52 µs ± 75.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 10, np_any(x)
340 ns ± 3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 100, any(x)
1.5 µs ± 9.82 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 100, np.any(x)
7.14 µs ± 1.01 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 100, np_any(x)
387 ns ± 6.03 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 1000, any(x)
14.3 µs ± 146 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 1000, np.any(x)
6.57 µs ± 54.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 1000, np_any(x)
783 ns ± 5.75 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 10000, any(x)
140 µs ± 660 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
i = 10000, np.any(x)
6.65 µs ± 126 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 10000, np_any(x)
4.86 µs ± 81.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 50000, any(x)
703 µs ± 7.79 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
i = 50000, np.any(x)
6.54 µs ± 64.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 50000, np_any(x)
26.5 µs ± 2 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
i = None, all(x)
111 ns ± 3.14 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
i = None, np.all(x)
3.89 µs ± 205 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = None, np_all(x)
333 ns ± 6.14 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 0, all(x)
111 ns ± 3.07 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
i = 0, np.all(x)
6.18 µs ± 41.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 0, np_all(x)
334 ns ± 3.92 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 10, all(x)
250 ns ± 1.81 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 10, np.all(x)
6.25 µs ± 57 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 10, np_all(x)
340 ns ± 4.21 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 100, all(x)
1.52 µs ± 19.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 100, np.all(x)
6.18 µs ± 74 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 100, np_all(x)
423 ns ± 35.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 1000, all(x)
14 µs ± 100 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 1000, np.all(x)
6.26 µs ± 71.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 1000, np_all(x)
775 ns ± 6.27 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
i = 10000, all(x)
140 µs ± 642 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
i = 10000, np.all(x)
6.55 µs ± 396 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 10000, np_all(x)
4.9 µs ± 56.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 50000, all(x)
717 µs ± 36.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
i = 50000, np.all(x)
6.3 µs ± 66.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
i = 50000, np_all(x)
22.5 µs ± 96 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
"""

#%% Functions - parse_results
def parse_results(text):
    r"""Parses the output into a more succient table."""
    lines = text.split('\n')
    table = pd.DataFrame(index=['None', '0', '10', '100', '1000', '10000', '50000'], \
        columns=['any(x)', 'np.any(x)', 'np_any(x)', 'all(x)', 'np.all(x)', 'np_all(x)'])
    for (ix, line) in enumerate(lines):
        if line.startswith('i = '):
            parts = line.split(', ')
            value = parts[0].split(' = ')[1]
            func = parts[1]
            next_line = lines[ix+1]
            speed = next_line.split(' ± ')[0]
            table.loc[value][func] = speed
        else:
            continue
    return table

#%% Script
if __name__ == '__main__':
    ipython = get_ipython()
    for i in [None, 0, 10, 100, 1000, 10000, 50000]:
        x = np.zeros(100000, dtype=bool)
        if x is not None:
            x[i] = True
        print(f'i = {i}, any(x)')
        ipython.magic('timeit any(x)')
        print(f'i = {i}, np.any(x)')
        ipython.magic('timeit np.any(x)')
        print(f'i = {i}, np_any(x)')
        ipython.magic('timeit np_any(x)')

    for i in [None, 0, 10, 100, 1000, 10000, 50000]:
        x = np.ones(100000, dtype=bool)
        if x is not None:
            x[i] = False
        print(f'i = {i}, all(x)')
        ipython.magic('timeit all(x)')
        print(f'i = {i}, np.all(x)')
        ipython.magic('timeit np.all(x)')
        print(f'i = {i}, np_all(x)')
        ipython.magic('timeit np_all(x)')

    #table = parse_results(text)
    #print(table)

    # Gives:
    #         any(x) np.any(x) np_any(x)   all(x) np.all(x) np_all(x)
    # None    111 ns   3.74 µs    342 ns   111 ns   3.89 µs    333 ns
    # 0       116 ns   6.55 µs    339 ns   111 ns   6.18 µs    334 ns
    # 10      255 ns   6.52 µs    340 ns   250 ns   6.25 µs    340 ns
    # 100     1.5 µs   7.14 µs    387 ns  1.52 µs   6.18 µs    423 ns
    # 1000   14.3 µs   6.57 µs    783 ns    14 µs   6.26 µs    775 ns
    # 10000   140 µs   6.65 µs   4.86 µs   140 µs   6.55 µs    4.9 µs
    # 50000   703 µs   6.54 µs   26.5 µs   717 µs    6.3 µs   22.5 µs
