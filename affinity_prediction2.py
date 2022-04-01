import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import lightgbm as lgb


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

train_data = pd.read_pickle('temp.pkl')
x_train = train_data.drop(['delta_g'], axis=1)
y_train = train_data['delta_g']


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

NFOLDS = 5
folds = KFold(n_splits=NFOLDS,shuffle = True)
X1 = x_train
y1 = y_train

#columns = local_df.columns
splits = folds.split(X1, y1)
score = 0
spearmancor = list()
# training_start_time = time()
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X1.iloc[train_index], X1.iloc[valid_index]
    y_train, y_valid = y1.iloc[train_index], y1.iloc[valid_index]

    clf = LGBMRegressor(n_estimators=100).fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    spearmancor.append(spearmanr(y_pred, y_valid))

#     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
#     print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean sparmancor:', np.mean(spearmancor))
print('-' * 30)

# best_iter = clf.best_iteration
# clf = lgb.LGBMRegressor(**params, num_boost_round=best_iter)
# clf.fit(X1, y1)
# sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
# sub['isFraud'] = clf.predict_proba(test)[:, 1]
# sub.to_csv('submission_cis_fraud_detection_v2.csv', index=False)
