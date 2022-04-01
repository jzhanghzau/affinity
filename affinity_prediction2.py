from Bio.Seq import transcribe
import numpy as np
import pandas as pd
from scipy.sparse.construct import rand
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
from sklearn import model_selection



def spearmanr(y_pred, y_true):

    diff_pred, diff_true = y_pred - np.mean(y_pred), y_true - np.mean(y_true)

    return np.sum(diff_pred * diff_true) / np.sqrt(np.sum(diff_pred **2) * np.sum(diff_true **2))
# =------------------------------------------------------------------------------------------------------------------------------------------------
# train_data = pd.read_pickle('temp.pkl')
# x_train = train_data.drop(['delta_g'], axis=1)
# y_train = train_data['delta_g']
# print(train_data)
# NFOLDS = 5
# folds = KFold(n_splits=NFOLDS,shuffle = False)
# X1 = x_train
# y1 = y_train

# #columns = local_df.columns
# splits = folds.split(X1, y1)
# score = 0
# spearmancor = list()
# # training_start_time = time()

# for fold_n, (train_index, valid_index) in enumerate(splits):
#     X_train, X_valid = X1.iloc[train_index], X1.iloc[valid_index]
#     y_train, y_valid = y1.iloc[train_index], y1.iloc[valid_index]

#     clf = RandomForestRegressor(n_estimators=200).fit(X_train, y_train)
#     y_pred = clf.predict(X_valid)
#     spearmancor.append(spearmanr(y_pred, y_valid))

# #     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
# print('-' * 30)
# print('Training has finished.')
# #     print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
# print('Mean AUC:', np.mean(spearmancor))
# print('-' * 30)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# train_data = pd.read_pickle('temp.pkl')
# x_train = train_data.drop(['delta_g'], axis=1)
# y_train = train_data['delta_g']


# params = {'num_leaves': 100,
#           'min_child_weight': 0.03454472573214212,
#           'feature_fraction': 0.3797454081646243,
#           'bagging_fraction': 0.4181193142567742,
#           'min_data_in_leaf': 106,
#           'objective': 'binary',
#           'max_depth': 3,
#           'learning_rate': 0.02,
#           "boosting_type": "gbdt",
#           "bagging_seed": 11,
#           "metric": 'auc',
#           "verbosity": -1,
#           'reg_alpha': 0.3899927210061127,
#           'reg_lambda': 0.6485237330340494,
#           'random_state': 47,
#          }

# NFOLDS = 5
# folds = KFold(n_splits=NFOLDS,shuffle = True)
# X1 = x_train
# y1 = y_train

# #columns = local_df.columns
# splits = folds.split(X1, y1)
# score = 0
# spearmancor = list()
# # training_start_time = time()
# for fold_n, (train_index, valid_index) in enumerate(splits):
#     X_train, X_valid = X1.iloc[train_index], X1.iloc[valid_index]
#     y_train, y_valid = y1.iloc[train_index], y1.iloc[valid_index]

#     clf = LGBMRegressor(n_estimators=100).fit(X_train, y_train)
#     y_pred = clf.predict(X_valid)
#     spearmancor.append(spearmanr(y_pred, y_valid))

# #     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
# print('-' * 30)
# print('Training has finished.')
# #     print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
# print('Mean sparmancor:', np.mean(spearmancor))
# print('-' * 30)

# # best_iter = clf.best_iteration
# # clf = lgb.LGBMRegressor(**params, num_boost_round=best_iter)
# # clf.fit(X1, y1)
# # sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
# # sub['isFraud'] = clf.predict_proba(test)[:, 1]
# # sub.to_csv('submission_cis_fraud_detection_v2.csv', index=False)

# cdr3_train = pd.read_pickle('cdr3_train.pkl')
# cdr3_sabdab  = pd.read_pickle('cdr3_sabdab.pkl')
# cdr3_external = pd.read_pickle('cdr3_external.pkl')
# cdr3_sabdab = cdr3_sabdab.drop(list(set(cdr3_sabdab.columns) - set(cdr3_train.columns)), axis=1)
# train_data = pd.concat([cdr3_train, cdr3_external, cdr3_sabdab],axis=0).reset_index(drop=True)
# -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
train_data = pd.read_pickle('transformed_trainB.pkl')

# -----------------------------------------------------------------------------------------------------
def objective(trial):
    
    params = {
          'n_estimators':trial.suggest_int('n_estimators',400,1200), 
          'num_leaves': trial.suggest_int('num_leaves',30, 100),
          'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
          #'feature_fraction': trial.suggest_float('feature_fraction',0, 0.8, step=0.1),
          #'bagging_fraction': trial.suggest_float('bagging_fraction',0, 0.9, step=0.1),
          #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 10000, step=50),
          'max_depth': trial.suggest_int('max_depth', 4, 7),
          'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.12),
          #"bagging_seed": trial.suggest_int('bagging_seed', 0, 100),
          "max_bin" : trial.suggest_int('max_bin', 20, 120),
          "reg_alpha": trial.suggest_loguniform('reg_alpha', 1e-3, 10),
          "reg_lambda": trial.suggest_loguniform('reg_lambda', 1e-3, 10),
          "colsample_bytree": trial.suggest_float('colsample_bytree', 0.1, 1),
          'subsample': trial.suggest_float('subsample', 0.1, 1),
          'cat_smooth':trial.suggest_int('cat_smooth', 1, 75)
         }

    regressor_object = LGBMRegressor(**params)
    

    # train_data1 = pd.read_pickle('temp.pkl')
    # #train_data1 = train_data1.drop(train_data1.filter(regex='antibody_seq_b').columns, axis=1)
    # train_data2 = pd.read_pickle('external_data2.pkl')
    # train_data = pd.concat([train_data1, train_data2]).reset_index(drop=True)

  

    #train_data = pd.read_pickle('temp.pkl')
    x = train_data.drop(['delta_g'], axis=1)
    y = train_data['delta_g']
    

    # X_train, X_val, y_train, y_val = model_selection.train_test_split(x, y,random_state=None,test_size=0.2)
    # regressor_object.fit(X_train, y_train)
    # y_pred = regressor_object.predict(X_val)
    # spearmancor = spearmanr(y_pred, y_val)
    
    
    # res = []
    # for i in range(0, 5):
    #     X_train, X_val, y_train, y_val = model_selection.train_test_split(x, y,random_state=None,test_size=0.2)
    #     #regressor_object.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2", early_stopping_rounds=100)
    
    #     regressor_object.fit(X_train, y_train)
    #     y_pred = regressor_object.predict(X_val)
    #     spearmancor = spearmanr(y_pred, y_val)
    #     res.append(spearmancor)
    #     print("spearman{}".format(spearmancor))
    
 
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    res = np.empty(4)
    for idx, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train, X_val = x.iloc[train_index], x.iloc[test_index]
        y_train, y_val = y[train_index], y[test_index]
        regressor_object.fit(X_train, y_train)
        y_pred = regressor_object.predict(X_val)
        spearmancor = spearmanr(y_pred, y_val)
        print("spearman{}".format(spearmancor))
        res[idx] = spearmancor
    
    return np.mean(res)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=6000)
print(study.best_trial.params)
for k, v in study.best_params.items():
    print(f"\t\t{k}:{v}")

# n_estimators:799
# num_leaves:385
# min_child_samples:45
# max_depth:10
# max_bin:427
# cas_smooth:230
# learning_rate:0.02908072567148931
# reg_alpha:0.0015561900640463705
# reg_lambda:2.3764669196245944
# colsample_bytree:0.3572271041712845
# subsample:0.509630753602541
# -----------------------------------------------------------------------------------------------
# train_data = pd.read_pickle('temp.pkl')
# x = train_data.drop(['delta_g'], axis=1)
# y = train_data['delta_g']
# params = {
#         'n_estimators':799, 
#         'num_leaves': 385,
#         'min_child_samples': 45,
#         'max_depth': 10,
#         'learning_rate': 0.02908072567148931,
#         "max_bin" : 427,
#         "reg_alpha": 0.0015561900640463705,
#         "reg_lambda": 2.3764669196245944,
#         "colsample_bytree": 0.3572271041712845,
#         'subsample': 0.509630753602541,
#         'cat_smooth':230
#         }

# regressor_object = LGBMRegressor(**params)
# regressor_object.fit(x, y)

# x_val = pd.read_pickle('test.pkl')
# y_pred = regressor_object.predict(x_val)

# res = pd.DataFrame({'Id':list(range(1, len(x_val)+1)), 'deltaG':y_pred})
# print(res)

# res.to_csv('res.csv', index=False)

# ----------------------------------------------------------------------------------------------
# train_data = pd.read_pickle('temp.pkl')
# print(train_data.columns)

# ----------------------------------------------------------------------------------------------


# sabdab_cdr3 = pd.read_csv('protein_sabdab_CDR3.tsv', sep='\t')
# sabdab_cdr3['Y'] = sabdab_cdr3['Y'].apply(lambda x: 0.001989*293*np.log(x))

# ----------------------------------------------------------------------

#Trial 235 finished with value: 0.8305403701319055 and parameters: {'n_estimators': 1694, 'num_leaves': 255, 'min_child_samples': 185, 'max_depth': 5, 'learning_rate': 0.06573918899909405, 'max_bin': 143, 'reg_alpha': 0.01714756032161254, 'reg_lambda': 0.09026417107715128, 'colsample_bytree': 0.43439348469365885, 'subsample': 0.5110952855389279, 'cas_smooth': 27}. Best is trial 235 with value: 0.8305403701319055.