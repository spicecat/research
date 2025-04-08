################################################################################
# Description:                                                                 #
# Follow Gu, Kelly, and Xiu (2020)                                             #
# Use machine learning methods to predict stock returns                        #
# Compare out-of-sample R2                                                     #
#                                                                              #
# Methods:                                                                     #
# Ordinary Least Squares (OLS)                                                 #
# Random Forest (RF)                                                           #
# Gradient Boosting Regression Tree (GBRT)                                     #
# Neural Networks (NN)                                                         #
#                                                                              #
# Author: Xudong Wen                                                           #
################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

################################################################################
# Characteristics
# Variables are constructed using Green, Hand, and Zhang (2017)'s sas code
# absacc  : Absolute accruals
# beta    : Beta
# cfp     : Cash flow to price ratio
# chpmia  : Industry-adjusted change in profit margin
# ep      : Earnings to price
# gma     : Gross profitability
# herf    : Industry sales concentration
# idiovol : Idiosyncratic return volatility
# lev     : Leverage
# mom12m  : 12-month momentum
# mom6m   : 6-month momentum
# nincr   : Number of earnings increases
# pchdepr : % change in depreciation
# ps      : Financial statements score
# roavol  : Earnings volatility
# roeq    : Return on equity
# salecash: Sales to cash
# stdcf   : Cash flow volatility
# sue     : Unexpected quarterly earnings
# tang    : Debt capacity/firm tangibility
CharsVars = ['absacc', 'beta', 'cfp', 'chpmia', 'ep', 'gma', 'herf', 'idiovol', 'lev', 'mom12m',
             'mom6m', 'nincr', 'pchdepr', 'ps', 'roavol', 'roeq', 'salecash', 'stdcf', 'sue', 'tang']

# Sample split
# Training sample  : 2000 ~ 2009 (10 years)
# Validation sample: 2010 ~ 2014 (5 years)
# Testing sample   : 2015 ~ 2019 (5 years)
# ym = (year - 1960) * 12 + (month - 1)
ym_train_st = (2000 - 1960) * 12
ym_train_ed = (2009 - 1960) * 12 + 11
ym_valid_st = (2010 - 1960) * 12
ym_valid_ed = (2014 - 1960) * 12 + 11
ym_test_st = (2015 - 1960) * 12
ym_test_ed = (2019 - 1960) * 12 + 11

# Number of selected models
# Select 5 models with the highest validation R2
# Then average these models' outputs in testing sample
navg = 5

################################################################################
# Function: winsorize
def winsorize(X, q):
    X = X.copy()
    q_l = np.nanquantile(X, q)
    q_u = np.nanquantile(X, 1 - q)
    X[X < q_l] = q_l
    X[X > q_u] = q_u
    return X

# Function: calculate R2
# Follow Gu, Kelly, and Xiu (2020), the denominator is without demeaning
def cal_r2(y_true, y_pred):
    return 100 * (1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true)))

################################################################################
# Load data
# Remove microcap stocks whose market capitalization fall below the 20th NYSE cutpoint
# lme   : 1-month-lagged market equity
# retadj: stock monthly return (adjust for delisting)
# exret : excess return (= retadj - rf)
# rf    : risk-free rate
retdata = pd.read_stata('D:/Research/Data/ML_Methods/ML_sample.dta')
retdata['ym'] = (retdata['year'] - 1960) * 12 + (retdata['month'] - 1)
retdata = retdata.astype({'permno': 'int', 'year': 'int', 'month': 'int', 'ym': 'int'})
retdata = retdata[['permno', 'year', 'month', 'ym', 'lme', 'retadj', 'exret'] + CharsVars]
retdata = retdata.sort_values(by=['ym', 'permno'], ascending=True).reset_index(drop=True)
assert not retdata.duplicated(subset=['ym', 'permno']).any()

# Winsorize
# The winsorized return is only used in training sample
retdata['exret_winsor'] = retdata.groupby('ym')['exret'].transform(winsorize, q=0.01)

# Time index
ym_dc = {}
for i, ym_i in enumerate(retdata['ym']):
    if ym_dc.get(ym_i) is None:
        ym_dc[ym_i] = [i]
    else:
        ym_dc[ym_i].append(i)
ym_ls = sorted(ym_dc.keys())

# Cross-sectional standardization
# Fill missing with cross-sectional mean (0 after standardization)
CMat = retdata[CharsVars].values.copy()
for ym_i in ym_ls:
    idx_i = ym_dc[ym_i]
    CMat[idx_i, :] = preprocessing.scale(CMat[idx_i, :], axis=0)
CMat[np.isnan(CMat)] = 0
retdata[CharsVars] = CMat
del CMat

################################################################################
# Sample index
idx_train_st = min(ym_dc[ym_train_st])
idx_train_ed = max(ym_dc[ym_train_ed])
idx_valid_st = min(ym_dc[ym_valid_st])
idx_valid_ed = max(ym_dc[ym_valid_ed])
idx_test_st = min(ym_dc[ym_test_st])
idx_test_ed = max(ym_dc[ym_test_ed])

# Training & validation & testing sample
X_train = retdata.loc[idx_train_st:idx_train_ed, CharsVars].values
Y_train = retdata.loc[idx_train_st:idx_train_ed, 'exret_winsor'].values
X_valid = retdata.loc[idx_valid_st:idx_valid_ed, CharsVars].values
Y_valid = retdata.loc[idx_valid_st:idx_valid_ed, 'exret'].values
X_test = retdata.loc[idx_test_st:idx_test_ed, CharsVars].values
Y_test = retdata.loc[idx_test_st:idx_test_ed, 'exret'].values
del retdata

################################################################################
# Ordinary Least Squares (OLS)
OLS = linear_model.LinearRegression()
OLS.fit(X_train, Y_train)
Y_pred = OLS.predict(X_test)
R2_OLS = cal_r2(Y_test, Y_pred)

################################################################################
# Random Forest (RF)
# Hyper-parameter list
hp_ls = []
for max_depth in [1, 2, 3]:
    for n_estimators in [30, 50]:
        for max_features in [5, 10]:
            hp_ls.append((max_depth, n_estimators, max_features))

# Training & validation
RF_ls = []
n_hp_ls = len(hp_ls)
R2_valid_ls = np.full(n_hp_ls, np.nan)
for i, hp_i in enumerate(hp_ls):
    print('Training (RF):', i + 1, '/', n_hp_ls)
    RF = RandomForestRegressor(max_depth=hp_i[0], n_estimators=hp_i[1], max_features=hp_i[2], n_jobs=4, random_state=i)
    RF.fit(X_train, Y_train)
    Y_pred = RF.predict(X_valid)
    R2_valid_ls[i] = cal_r2(Y_valid, Y_pred)
    RF_ls.append(RF)

# Testing
avg_models_id = np.argsort(-R2_valid_ls)[:navg]
for i, mid in enumerate(avg_models_id):
    if i == 0:
        Y_pred = RF_ls[mid].predict(X_test) / navg
    else:
        Y_pred = Y_pred + RF_ls[mid].predict(X_test) / navg
R2_RF = cal_r2(Y_test, Y_pred)

################################################################################
# Gradient Boosting Regression Tree (GBRT)
# Hyper-parameter list
hp_ls = []
for max_depth in [1, 2, 3]:
    for n_estimators in [5, 10]:
        for lr in [0.01, 0.05]:
            hp_ls.append((max_depth, n_estimators, lr))

# Training & validation
GBRT_ls = []
n_hp_ls = len(hp_ls)
R2_valid_ls = np.full(n_hp_ls, np.nan)
for i, hp_i in enumerate(hp_ls):
    print('Training (GBRT):', i + 1, '/', n_hp_ls)
    GBRT = GradientBoostingRegressor(max_depth=hp_i[0], n_estimators=hp_i[1], learning_rate=hp_i[2], max_features='sqrt', random_state=i)
    GBRT.fit(X_train, Y_train)
    Y_pred = GBRT.predict(X_valid)
    R2_valid_ls[i] = cal_r2(Y_valid, Y_pred)
    GBRT_ls.append(GBRT)

# Testing
avg_models_id = np.argsort(-R2_valid_ls)[:navg]
for i, mid in enumerate(avg_models_id):
    if i == 0:
        Y_pred = GBRT_ls[mid].predict(X_test) / navg
    else:
        Y_pred = Y_pred + GBRT_ls[mid].predict(X_test) / navg
R2_GBRT = cal_r2(Y_test, Y_pred)

################################################################################
# Neural Networks (NN)
# Hyper-parameter list
hp_ls = []
for dropout in [0.05, 0.10]:
    for lr in [0.01, 0.05]:
        for batch_size in [1000, 5000]:
            for epoch in [20, 50]:
                hp_ls.append((dropout, lr, batch_size, epoch))

# Transform numpy to tensor
Y_train_tf = tf.cast(Y_train.reshape([-1, 1]), tf.float32)
X_train_tf = tf.cast(X_train, tf.float32)
X_valid_tf = tf.cast(X_valid, tf.float32)
X_test_tf = tf.cast(X_test, tf.float32)

# Training & validation
NN_ls = []
n_hp_ls = len(hp_ls)
R2_valid_ls = np.full(n_hp_ls, np.nan)
for i, hp_i in enumerate(hp_ls):
    print('Training (NN):', i + 1, '/', n_hp_ls)
    tf.random.set_seed(i)
    NN = tf.keras.models.Sequential([
        tf.keras.layers.Dropout(hp_i[0]),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    NN.compile(optimizer=tf.keras.optimizers.Adam(hp_i[1]), loss='mse')
    NN.fit(x=X_train_tf, y=Y_train_tf, batch_size=hp_i[2], epochs=hp_i[3], verbose=0)
    Y_pred = np.reshape(NN(X_valid_tf, training=False).numpy(), [-1])
    R2_valid_ls[i] = cal_r2(Y_valid, Y_pred)
    NN_ls.append(NN)

# Testing
avg_models_id = np.argsort(-R2_valid_ls)[:navg]
for i, mid in enumerate(avg_models_id):
    if i == 0:
        Y_pred = np.reshape(NN_ls[mid](X_test_tf, training=False).numpy(), [-1]) / navg
    else:
        Y_pred = Y_pred + np.reshape(NN_ls[mid](X_test_tf, training=False).numpy(), [-1]) / navg
R2_NN = cal_r2(Y_test, Y_pred)

################################################################################
# Print
print('R2 - OLS  (%):', R2_OLS)
print('R2 - RF   (%):', R2_RF)
print('R2 - GBRT (%):', R2_GBRT)
print('R2 - NN   (%):', R2_NN)
