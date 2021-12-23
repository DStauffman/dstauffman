# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:57:30 2021

@author: e182918
"""
import logging
import time

import dstauffman as dcs

#%% Globals
logger = logging.getLogger(__name__)


#%% Functions
def func(num: int) -> dict[str, int]:
    """Long-running function that sometimes errors."""
    if num in {5, 15}:
        raise RuntimeError(f'Bad things happened at {num}.')
    out = {'time': 1, 'data': num}
    logger.log(dcs.LogLevel.L5, 'Ran "%i"', num)
    time.sleep(0.1)
    return out


#%% Script
if __name__ == '__main__':
    log_file = None
    dcs.activate_logging(dcs.LogLevel.L8, filename=log_file)
    args = zip(range(5))
    results = dcs.parfor_wrapper(func, args, use_parfor=True, max_cores=4, ignore_errors=True)

    print(results)
