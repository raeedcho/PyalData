import numpy as np
import pandas as pd

import functools
import warnings


def copy_td(func):
    """
    Call copy on the first argument of the function and work on the copied value
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # if there are no positional arguments
        if len(args) == 0:
            df = kwargs["trial_data"]

            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"first argument of {func.__name__} has to be a pandas DataFrame")

            kwargs["trial_data"] = df.copy()
            return func(**kwargs)
        else:
            # dataframe is the first positional argument
            if not isinstance(args[0], pd.DataFrame):
                raise ValueError(f"first argument of {func.__name__} has to be a pandas DataFrame")

            return func(args[0].copy(), *args[1:], **kwargs)

    return wrapper


def remove_suffix(text, suffix):
    """
    Remove suffix from the end of text

    Parameters
    ----------
    text : str
        text from which to remove the suffix
    suffix : str
        suffix to remove from the end of text

    Returns
    -------
    text : str
        text with suffix removed if text ends with suffix
        text untouched if text doesn't end with suffix
    """
    if text.endswith(suffix):
        return text[:-len(suffix)]
    else:
        warnings.warn(f"{text} doesn't end with {suffix}. Didn't remove anything.")
    return text


  
def get_time_varying_fields(trial_data, ref_field=None):
    """
    Identify time-varying fields in the dataset


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    time_fields : list of str
        list of fieldnames that store time-varying signals
    """
    # identify candidates in each trial and take union
    time_field_sets_per_trial= [set(get_time_varying_fields_in_trial(trial, ref_field=ref_field))
                            for _,trial in trial_data.iterrows()]
    time_fields = set().union(*time_field_sets_per_trial)

    # check if all trials have the same time-varying fields
    # Note: in some cases, this check will fail,
    # specifically if the time varying fields on one or more trials happens to be the same length as the number of neurons
    for trialnum,time_fields_in_trial in enumerate(time_field_sets_per_trial):
        set_diff = set(time_fields) - set(time_fields_in_trial)
        assert len(set_diff)==0, f"{set_diff} on trial {trial_data['trial_id'].values[trialnum]} have bad lengths"

    return list(time_fields)

def get_time_varying_fields_in_trial(trial, ref_field=None):
    """
    Get the names of time-varying fields in the trial

    Parameters
    ----------
    trial : pd.Series
        trial to check
    ref_field : str, optional
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    time_fields : list of str
    """
    # identify candidates based on trial length
    T = get_trial_length(trial, ref_field=ref_field)
    time_fields = []
    for col in trial.index:
        try:
            if trial[col].shape[0] == T:
                time_fields.append(col)
        except:
            pass

    return time_fields

def get_array_fields(trial_data):
    """
    Get the names of columns that contain numpy arrays

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    Returns
    -------
    columns that have array values : list of str
    """
    return [col for col in trial_data.columns if all([isinstance(el, np.ndarray) for el in trial_data[col]])]


def get_string_fields(trial_data):
    """
    Get the names of columns that contain string data

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    Returns
    -------
    columns that have string values : list of str
    """
    return [col for col in trial_data.columns if all([isinstance(el, str) for el in trial_data[col]])]


def get_trial_length(trial, ref_field=None):
    """
    Get the number of time points in the trial

    Parameters
    ----------
    trial : pd.Series
        trial to check
    ref_field : str, optional
        time-varying field to use for identifying the length
        if not given, the first field that ends with "spikes" is used

    Returns
    -------
    length : int
    """
    if ref_field is None:
        ref_field = [col for col in trial.index.values if col.endswith("spikes") or col.endswith("rates")][0]

    return np.size(trial[ref_field], axis=0)

