r"""Runs speed comparisons for the any and all commands."""

# %% Imports
from IPython import get_ipython  # type: ignore[import-not-found]
import numpy as np
import pandas as pd

import nubs as nubs  # pylint: disable=unused-import  # noqa: F401, RUF100

# %% Results
TEXT = r"""i = None, any(x)
96 ns ± 5.88 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
i = None, np.any(x)
4.73 μs ± 242 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = None, np_any(x)
434 ns ± 32.6 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 0, any(x)
96 ns ± 3.08 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
i = 0, np.any(x)
5.93 μs ± 365 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 0, np_any(x)
387 ns ± 34.4 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 10, any(x)
240 ns ± 27.1 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 10, np.any(x)
5.17 μs ± 266 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 10, np_any(x)
395 ns ± 25 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 100, any(x)
1.43 μs ± 54.4 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 100, np.any(x)
5.58 μs ± 415 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 100, np_any(x)
430 ns ± 25.5 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 1000, any(x)
13.4 μs ± 293 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 1000, np.any(x)
5.45 μs ± 439 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 1000, np_any(x)
751 ns ± 31.9 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 10000, any(x)
135 μs ± 6.02 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
i = 10000, np.any(x)
5.51 μs ± 289 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 10000, np_any(x)
4.19 μs ± 151 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 50000, any(x)
687 μs ± 29.2 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
i = 50000, np.any(x)
5.22 μs ± 363 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 50000, np_any(x)
20.4 μs ± 1.93 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
i = None, all(x)
90 ns ± 3.68 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
i = None, np.all(x)
4.23 μs ± 370 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = None, np_all(x)
425 ns ± 33.2 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 0, all(x)
109 ns ± 4.84 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
i = 0, np.all(x)
5.68 μs ± 624 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 0, np_all(x)
424 ns ± 30.5 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 10, all(x)
275 ns ± 25.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 10, np.all(x)
5.92 μs ± 398 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 10, np_all(x)
412 ns ± 16.9 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 100, all(x)
1.62 μs ± 54.6 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 100, np.all(x)
5.66 μs ± 266 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 100, np_all(x)
482 ns ± 24.8 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 1000, all(x)
14.6 μs ± 1.45 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
i = 1000, np.all(x)
6 μs ± 214 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 1000, np_all(x)
949 ns ± 53.9 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
i = 10000, all(x)
142 μs ± 11.7 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
i = 10000, np.all(x)
6 μs ± 379 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 10000, np_all(x)
4.77 μs ± 219 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 50000, all(x)
725 μs ± 45.7 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
i = 50000, np.all(x)
7.07 μs ± 1.17 μs per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
i = 50000, np_all(x)
21.4 μs ± 1.66 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
"""


# %% Functions - parse_results
def parse_results(text: str) -> pd.DataFrame:
    r"""Parses the output into a more succient table."""
    lines = text.split("\n")
    table = pd.DataFrame(
        index=["None", "0", "10", "100", "1000", "10000", "50000"],
        columns=["any(x)", "np.any(x)", "np_any(x)", "all(x)", "np.all(x)", "np_all(x)"],
    )
    for ix, line in enumerate(lines):
        if line.startswith("i = "):
            parts = line.split(", ")
            value = parts[0].split(" = ")[1]
            func = parts[1]
            next_line = lines[ix + 1]
            speed = next_line.split(" ± ")[0]
            table.loc[value, func] = speed
        else:
            continue
    return table


# %% Script
if __name__ == "__main__":
    ipython = get_ipython()
    assert ipython is not None
    for i in [None, 0, 10, 100, 1000, 10000, 50000]:
        x = np.zeros(100000, dtype=bool)
        if x is not None:
            x[i] = True
        print(f"i = {i}, any(x)")
        ipython.run_line_magic("timeit", "any(x)")
        print(f"i = {i}, np.any(x)")
        ipython.run_line_magic("timeit", "np.any(x)")
        print(f"i = {i}, np_any(x)")
        ipython.run_line_magic("timeit", "nubs.np_any(x)")

    for i in [None, 0, 10, 100, 1000, 10000, 50000]:
        x = np.ones(100000, dtype=bool)
        if x is not None:
            x[i] = False
        print(f"i = {i}, all(x)")
        ipython.run_line_magic("timeit", "all(x)")
        print(f"i = {i}, np.all(x)")
        ipython.run_line_magic("timeit", "np.all(x)")
        print(f"i = {i}, np_all(x)")
        ipython.run_line_magic("timeit", "nubs.np_all(x)")

    # table = parse_results(TEXT)
    # print(table)

    # Gives:
    #         any(x) np.any(x) np_any(x)   all(x) np.all(x) np_all(x)
    # None     96 ns   4.73 μs    434 ns    90 ns   4.23 μs    425 ns
    # 0        96 ns   5.93 μs    387 ns   109 ns   5.68 μs    424 ns
    # 10      240 ns   5.17 μs    395 ns   275 ns   5.92 μs    412 ns
    # 100    1.43 μs   5.58 μs    430 ns  1.62 μs   5.66 μs    482 ns
    # 1000   13.4 μs   5.45 μs    751 ns  14.6 μs      6 μs    949 ns
    # 10000   135 μs   5.51 μs   4.19 μs   142 μs      6 μs   4.77 μs
    # 50000   687 μs   5.22 μs   20.4 μs   725 μs   7.07 μs   21.4 μs
