import os
import json
from sklearn.metrics import roc_auc_score


def generate_dtype_dict(features_cat: list = None,
                        features_num: list = None) -> dict:
    """
    Sets the metrical (numerical) predictors to float type and the nominal (categorical)
    predictors to object type.
    @param features_cat: List of categorical features
    @param features_num: List of numerical features
    @return: Dictionary containing the predictor names as key and their
    set type (float or object) as value.
    """
    d = {}
    if features_cat is not None:
        d.update(__generate_dtype_dict(features_cat, object))
    if features_num is not None:
        d.update(__generate_dtype_dict(features_num, float))
    return d


def __generate_dtype_dict(col_names: list, t: type) -> dict:
    """
    Private function which takes an non-empty list of column (predictor) names and
    a python builtin type and creates a dictionary with the column name as key and
    the type t as value.
    @param col_names: List of column (predictor) names.
    @param t: Required type of the columns listed in col_names.
    @return: Dictionary containing the column names as keys and type t as value.
    """
    if not col_names:
        raise ValueError('Column list cannot be empty.')
    d = {}
    for i, c in enumerate(col_names):
        d[c] = t
    return d


def check_fn(fn: str) -> tuple:
    """
    Checks if the file path is actually a file.
    @param fn: Path to a file.
    @return: Full path with filename and path to the directory where it is stored.
    """
    if os.path.isfile(fn):
        abs_fn = os.path.abspath(fn)
        abs_dir = os.path.dirname(abs_fn)
    else:
        raise ValueError('Cannot find the file with path = ' + fn)
    return abs_fn, abs_dir


def filename_to_dict(fn: str) -> dict:
    """
    Takes a string path to a json file, loads the json file and deserializes it as
    a dictionary.
    @param fn: String path to a json file which should be loaded.
    @return: Deserialized json file as dictionary.
    """
    abs_fn, abs_dir = check_fn(fn)
    if abs_fn:
        with open(abs_fn, 'r') as f:
            json_conf = json.load(f)
    return json_conf


def prefix_dict_key(d: dict, prefix: str) -> dict:
    """
    Concatenates a prefix to the keys of a dictionary, not changing the values.
    @param d: Dictionary for which the keys need to be changed
    @param prefix: Prefix for the key names
    @return: New dictionary whose keys are in the form of prefix + original key
    """
    return dict((prefix + key, value) for (key, value) in d.items())


def custom_roc_auc_score(y_true,
                         y_score,
                         average='macro',
                         sample_weight=None,
                         max_fpr=None,
                         multi_class='raise',
                         labels=None):
    return roc_auc_score(y_true=y_true,
                         y_score=y_score,
                         average='macro',
                         sample_weight=sample_weight,
                         max_fpr=max_fpr,
                         multi_class='ovo',
                         labels=labels)
