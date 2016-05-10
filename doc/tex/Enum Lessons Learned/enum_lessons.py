# -*- coding: utf-8 -*-
r"""
Enum lessons learned examples.

Notes
-----
#.  Written by David C. Stauffer in April 2015.
"""

#%% Imports
from enum import IntEnum, Enum, unique, EnumMeta
import numpy as np

#%% Meta Class
class _EnumMetaPlus(EnumMeta):
    r"""
    Overrides the repr/str methods of the EnumMeta class to display all possible
    values.
    """
    def __repr__(cls):
        text = [repr(field) for field in cls]
        return '\n'.join(text)
    def __str__(cls):
        text = [str(field) for field in cls]
        return '\n'.join(text)

#%% Extened IntEnum class
@unique
class _IntEnumPlus(int, Enum, metaclass=_EnumMetaPlus):
    r"""
    Custom IntEnum class based on _EnumMetaPlus metaclass to get more details from
    repr/str.

    Also forces all values to be unique.
    """
    def __str__(self):
        return '{}.{}: {}'.format(self.__class__.__name__, self.name, self.value)

#%% TB Status
class TbStatus(_IntEnumPlus):
    r"""
    Enumerator definitions for the possible Tuberculosis infection status.

    Notes
    -----
    #.  Negative values are uninfected, positive values are infected, zero
        is undefined.
    """
    null           =  0 # not set, used for preallocation
    uninfected     = -1 # never been infected
    recovered      = -2 # currently uninfected, but have been infected in the past
    latent_recent  =  1 # recently infected (<2 years)
    latent_remote  =  2 # immune stabilized infection
    active_untreat =  3 # active TB, not on treatment, or on ineffective treatment
    active_treated =  4 # active TB, on effective treatment

class TbStatus2(IntEnum):
    r"""
    Standard Enumerator
    """
    null           =  0
    uninfected     = -1
    recovered      = -2
    latent_recent  =  1
    latent_remote  =  2
    active_untreat =  3
    active_treated =  4

#%% Functions
def get_those_infected(tb_status):
    r"""
    Finds anyone who is infected with TB.
    """
    ix_infected = (tb_status == TbStatus.latent_recent) | (tb_status == \
        TbStatus.latent_remote) | (tb_status == TbStatus.active_treated) | \
        (tb_status == TbStatus.active_untreat)
    return ix_infected

def get_those_uninfected(tb_status):
    r"""
    Finds anyone who is not infected with TB.
    """
    ix_uninfected = (tb_status == TbStatus.uninfected) | \
        (tb_status == TbStatus.recovered)
    return ix_uninfected

#%% Example usage
if __name__ == '__main__':
    num = 100
    tb_status = TbStatus.null * np.ones(num, dtype=int)
    ix = np.random.rand(num)
    tb_status[ix >= 0.5] = TbStatus.active_treated
    tb_status[ix <  0.5] = TbStatus.uninfected


    ix_infected1 = tb_status > 0
    ix_infected2 = get_those_infected(tb_status)

    ix_uninfected1 = tb_status < 0
    ix_uninfected2 = get_those_uninfected(tb_status)

    np.testing.assert_equal(ix_infected1, ix_infected2)
    np.testing.assert_equal(ix_uninfected1, ix_uninfected2)

    # normal Enums
    print('Normal')
    print(TbStatus2.uninfected)
    print(repr(TbStatus2.uninfected))
    print(TbStatus2)
    print(repr(TbStatus2))

    # extended Enums
    print('Extended')
    print(TbStatus.uninfected)
    print(repr(TbStatus.uninfected))
    print(TbStatus)
    print(repr(TbStatus))