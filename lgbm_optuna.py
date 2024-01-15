import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import optuna
from optuna.integration import LightGBMPruningCallback
import warnings

warnings.filterwarnings("ignore")

path = '/stack_out_data/'
data_name = 'xxx.csv'
df = pd.read_csv(path + data_name, index_col = 0)

input_cols = ['summary_length', 'splling_err_num',
       'prompt_length',
       'length_ratio', 'word_overlap_count', 'bigram_overlap_count',
       'bigram_overlap_ratio', 'trigram_overlap_count',
       'trigram_overlap_ratio', 'quotes_count', 'model_output1',
       'emo_1', 'emo_2', 'emo_3',
       'emo_4', 'emo_5', 'emo_6', 'emo_7', 'emo_8', 'emo_9', 'emo_10',
       'emo_11', 'emo_12', 'emo_13', 'emo_14', 'emo_15', 'emo_16', 'emo_17',
       'emo_18', 'emo_19', 'emo_20', 'emo_21', 'emo_22', 'emo_23', 'emo_24',
       'emo_25', 'emo_26', 'emo_27', 'emo_28', 'model_output0',
       'model_output1', 'prompt_title']

target = ['content','wording']

def root_mean_squred_errors(y_true, y_pred):
    sample_size = y_true.shape[0]
    loss = np.sqrt(np.sum((y_true - y_pred)**2)/sample_size)
    return loss

def objective(trial, 
              x = df[input_cols], 
              y = df[[target[0]]]):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 30000, step = 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5),
        'num_leaves': trial.suggest_int('num_leaves', 5, 5000, step = 3),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.00001, 5, log = True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.00001, 5, log = True),
        'extra_tree': trial.suggest_categorical('extra_tree', [True, False]),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 0.5),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.00001, 0.1,log = True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0, 1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8, step = 0.02),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1., step = 0.02)
    }

    fold_loader = KFold(n_splits = 5, random_state = 66, shuffle = True)
    loss_li = []
    for i, (train_idx, val_idx) in enumerate(fold_loader.split(x, y)):
        lgb = LGBMRegressor(early_stopping_round = 30,
                            random_state = 66,
                            objective = 'rmse',
                            categorical_feature = -1,
                            **params)

        lgb.fit(x.iloc[train_idx],
                y.iloc[train_idx],
                eval_set = (x.iloc[val_idx], y.iloc[val_idx]),
                eval_metric = 'rmse', #gbm_metric,
                #categorical_feature = ['prompt_title'],
                callbacks = [LightGBMPruningCallback(trial, 'rmse')]
               )

        tr_pre = lgb.predict(x.iloc[train_idx])
        tr_loss = root_mean_squred_errors(y.iloc[train_idx].values.reshape(-1), tr_pre)

        val_pre = lgb.predict(x.iloc[val_idx])
        val_loss = root_mean_squred_errors(y.iloc[val_idx].values.reshape(-1), val_pre)
        loss_li.append(val_loss)

    return sum(loss_li) / 5

def go_tuning(df, 
              input_cols, 
              target):
    
    study = optuna.create_study(direction="minimize", study_name="LGBM Regressor")
    func = lambda trial: objective(trial, x = df[input_cols], y = df[[target[0]]])
    study.optimize(func, n_trials=500)
    return study.best_params