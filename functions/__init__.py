"""
Collection of functions.
"""
from oned_synth import oned_functions
from twod_synth import twod_functions

all_functions = oned_functions + twod_functions

def get_function_info(f_name):
    """Get function based on name and return Namespace object."""
    to_return = None
    for f_info in all_functions:
        if f_info.name.lower() == f_name:
            to_return = f_info
            break
    if to_return is None:
        raise ValueError('No function name %s' % f_name)
    return to_return
