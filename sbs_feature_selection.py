import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import math
import os
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")




def sbs(df, input_cols, target = ['content'], n_features = 'auto'):

    sfs_selector = SequentialFeatureSelector(
        estimator = LGBMRegressor(
            random_state = 66,
            objective = 'rmse',
            min_delta = 1e-3
        ),
        n_features_to_select = n_features,
        cv = 5,
        scoring = 'neg_root_mean_squared_error',
        tol = 5e-4 if n_features == 'auto' else None,
        direction ='backward'
    )

    sfs_selector.fit(df[input_cols], df[target])

    return df[input_cols].columns[sfs_selector.get_support()]