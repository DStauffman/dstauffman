r"""
dstauffman code related to health policy simulations.

Notes
-----
#.  Written by David C. Stauffer in October 2017.
#.  Moved into subfolder by David C. Stauffer in July 2020.
"""

#%% Imports
from .health import dist_enum_and_mons, icer
from .latex  import make_preamble, make_conclusion, bins_to_str_ranges, latex_str
from .stats  import convert_annual_to_monthly_probability, convert_monthly_to_annual_probability, \
                        ca2mp, cm2ap, prob_to_rate, rate_to_prob, \
                        annual_rate_to_monthly_probability, monthly_probability_to_annual_rate, \
                        ar2mp, mp2ar, combine_sets, bounded_normal_draw, rand_draw, ecdf

#%% Unittest
if __name__ == '__main__':
    pass
