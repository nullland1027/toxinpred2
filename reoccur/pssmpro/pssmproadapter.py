# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2022/9/14
__project__ = EvolutionaryFeature
Fix the Problem, Not the Blame.
'''

# from .pssmpro import get_feature, get_all_features
import os
import re

from collections import OrderedDict
import numpy as np
from typing import Union, Dict

import pandas as pd

from common import logger
from functools import lru_cache

from .pssmpro import *

UNCOMMON_AA_MAP = {
    "U": "C"
}


def get_feature(pssm_result: str, algo_type="aac_pssm") -> np.ndarray:
    """Input is a pssm result, not a dictionary"""
    pssm_mat = read_pssm_matrix(pssm_result)  # read pssm
    # replace uncommon aa
    pssm_mat[:, 0] = pd.Series(pssm_mat[:, 0]).map(lambda x: UNCOMMON_AA_MAP.get(x, x)).values
    features = np.array(eval(algo_type)(pssm_mat))  # calculate
    return features


@lru_cache(maxsize=128)
def get_all_features(pssm_result: str) -> Dict[str, np.ndarray]:
    """Get all features from possum"""
    res = OrderedDict()
    for enc in all_encoders:
        res[enc] = get_feature(pssm_result, enc)
    return res
