r"""
Support module file for the dstauffman.estimation code.

Defines the supporting code to use within the more generic batch parameter estimator included in the library.

Notes
-----
#.  Written by David C. Stauffer in April 2016.
#.  Pulled into dstauffman by David C. Stauffer in July 2020.
"""

# %% Imports
import doctest
from typing import Any
import unittest

from dstauffman import HAVE_NUMPY

if HAVE_NUMPY:
    import numpy as np


# %% _get_sub_level
def _get_sub_level(this_sub: Any, part: str) -> Any:
    r"""
    Gets the subfield level of parameters, with options to index by number into a list.

    Parameters
    ----------
    this_sub : object
        This sublevel object
    part : str
        Name of the next field to try and get

    Returns
    -------
    this_value : object
        Value of the lower level field item

    Examples
    --------
    >>> from dstauffman.estimation.support import _get_sub_level
    >>> from dstauffman.tests.test_estimation_support import _Parameters
    >>> param = _Parameters()
    >>> model = param.models[0]
    >>> this_sub = param
    >>> part = "models[0]"
    >>> model2 = _get_sub_level(this_sub, part)
    >>> print(model is model2)
    True

    >>> this_sub = param.models[0]
    >>> part = "field3['b'][1]"
    >>> value = _get_sub_level(this_sub, part)
    >>> print(value)
    2.5

    """
    # simple case
    if "[" not in part:
        return getattr(this_sub, part)

    # not so simple case
    sub_parts = part.split("[")
    temp_sub = getattr(this_sub, sub_parts[0])
    this_index = sub_parts[1].split("]")[0]
    try:
        this_key: int | str = int(this_index)
    except:
        this_key = this_index.replace('"', "").replace("'", "")
    this_value = temp_sub[this_key]
    if len(sub_parts) > 2:
        ix = sub_parts[2].split("]")[0]
        this_value = this_value[int(ix)]
    return this_value


# %% _check_valid_param_name
def _check_valid_param_name(param: Any, name: str) -> bool:
    r"""
    Checks whether the specified name actually exists.

    Parameters
    ----------
    param : class Parameters
        Model parameters
    names : str
        Name of the parameters you want to check

    Returns
    -------
    is_valid : bool
        Whether the name is valid or not

    Examples
    --------
    >>> from dstauffman.estimation.support import _check_valid_param_name
    >>> from dstauffman.tests.test_estimation_support import _Parameters
    >>> param = _Parameters()
    >>> names = ["param.config.log_level", "param.config.made_up_field_name", \
    ...     "param.models[1].bad_name['beta']", "param.models[0].field3['a']"]
    >>> is_valid = [_check_valid_param_name(param, name) for name in names]
    >>> print(is_valid)
    [True, False, False, True]

    """
    # split the name into parts
    parts = name.split(".")
    # check that the first part is param
    if parts[0] != "param":
        return False
    level = 1
    this_sub = param
    while level < len(parts):
        try:
            this_sub = _get_sub_level(this_sub, parts[level])
        except:
            return False
        level += 1
    # if we got here, then all the subpieces exist
    return True


# %% get_parameter
def get_parameter(param: Any, names: list[str]) -> list[Any]:
    r"""
    Gets the desired parameter by name.

    Parameters
    ----------
    param : class Parameters
        Model parameters
    names : list of str
        Names of the parameters you want values of

    Returns
    -------
    values : list
        Desired values from within param

    Examples
    --------
    >>> from dstauffman.estimation import get_parameter
    >>> from dstauffman.tests.test_estimation_support import _Parameters
    >>> param = _Parameters()
    >>> names = ["param.config.log_level", "param.models[0].field1", "param.models[1].field2[2]", "param.models[1].field3['b'][1]"]
    >>> values = get_parameter(param, names)
    >>> print([x for x in values]) #doctest: +NORMALIZE_WHITESPACE
    [20, 100, 300, 2.5]

    """
    # initialized the output values
    values = [np.nan for _ in range(len(names))]
    # loop through the names
    for ix, name in enumerate(names):
        # check that this name is valid
        is_valid = _check_valid_param_name(param, name)
        if not is_valid:
            raise ValueError(f'Bad name "{name}"')
        # split the name into parts
        parts = name.split(".")
        # loop through sublevels until you get the last value and add it to a set
        level = 1
        this_sub = param
        while level < len(parts):
            this_sub = _get_sub_level(this_sub, parts[level])
            level += 1
        values[ix] = this_sub
    return values


# %% set_parameter
def set_parameter(param: Any, names: list[str], values: list[Any]) -> None:
    r"""
    Sets the desired parameter by a given name.

    Parameters
    ----------
    param : class Parameters
        Model parameters
    names : list of str
        Names of the parameters you want to set
    values : list
        Values of the parameters you want to set

    Examples
    --------
    >>> from dstauffman.estimation import set_parameter
    >>> from dstauffman.tests.test_estimation_support import _Parameters
    >>> param = _Parameters()
    >>> names = ["param.config.log_level", "param.models[0].field1", "param.models[1].field2[2]", "param.models[1].field3['b'][1]"]
    >>> values = [-100, -2, -3, 44.]
    >>> print(param.models[0].field1)
    100

    >>> set_parameter(param, names, values)
    >>> print(param.models[0].field1)
    -2

    """
    # loop through the name/value pairs
    for name, value in zip(names, values):
        # check that this name is valid
        is_valid = _check_valid_param_name(param, name)
        if not is_valid:
            raise ValueError(f'Bad name "{name}"')
        # split the name into parts
        parts = name.split(".")
        # loop through sublevels until you get to the appropriate level
        level = 1
        this_sub = param
        while level < len(parts) - 1:
            this_sub = _get_sub_level(this_sub, parts[level])
            level += 1
        # find the key for this last level
        if "[" not in parts[level]:
            setattr(this_sub, parts[level], value)
            continue
        sub_parts = parts[level].split("[")
        this_index = sub_parts[1].split("]")[0]
        try:
            this_key: int | str = int(this_index)
        except:
            this_key = this_index.replace('"', "").replace("'", "")
        # set the value once on the last level
        if len(sub_parts) > 2:
            ix = sub_parts[2].split("]")[0]
            getattr(this_sub, sub_parts[0])[this_key][int(ix)] = value
        else:
            getattr(this_sub, sub_parts[0])[this_key] = value


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_estimation_support", exit=False)
    doctest.testmod(verbose=False)
