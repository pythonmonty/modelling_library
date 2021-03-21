import os
import json
from datetime import datetime

from sklearn.utils import Bunch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from acceleration_lib.defaults import DefaultValues
from acceleration_lib.skl_permutation_importance import permutation_importance
from matplotlib.backends.backend_pdf import PdfPages
import time


def generate_dtype_dict(predictors_nomi: list = None,
                        predictors_metr: list = None) -> dict:
    """
    Sets the metrical (numerical) predictors to float type and the nominal (categorical)
    predictors to object type.

    @param predictors_nomi: List of nominal (categorical) predictors.
    @param predictors_metr: List of metrical (numerical) predictors.

    @return: Dictionary containing the predictor names as key and their
    set type (float or object) as value.
    """
    d = {}
    if predictors_nomi is not None:
        d.update(__generate_dtype_dict(predictors_nomi, object))
    if predictors_metr is not None:
        d.update(__generate_dtype_dict(predictors_metr, float))
    return d


def check_fn(fn: str):
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