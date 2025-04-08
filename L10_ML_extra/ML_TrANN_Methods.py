################################################################################
# Description:                                                                 #
# Follow Gu, Kelly, and Xiu (2020)                                             #
# Use machine learning methods to predict stock returns                         #
# Compare out-of-sample R2                                                      #
#                                                                              #
# Methods:                                                                     #
# Original Methods in Gu, Kelly, and Xiu (2020):                              #
# - Ordinary Least Squares (OLS)                                               #
# - Random Forest (RF)                                                         #
# - Gradient Boosting Regression Tree (GBRT)                                   #
# - Neural Networks (NN)                                                       #
#                                                                              #
# Added Method:                                                                #
# - Transformed Artificial Neural Network (TREENN1)                            #
#   This method provides explainable AI capabilities by transforming           #
#   the neural network architecture to extract interpretable features          #
#                                                                              #
# Authors:                                                                     #
# Dr. Salih Tutun                                                             #
# Based on Xudong Wen's implementation                                        #
################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import sys
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import random

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
# Quick test function with simulated data
def run_quick_test():
    """
    Run a quick test of all methods with minimal hyperparameter search.
    """
    print("Generating synthetic data for quick testing...")
    np.random.seed(0)
    
    # Generate synthetic data with nonlinear relationships
    n_samples = 10000
    n_features = 20
    
    # Generate features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Generate target with nonlinear relationships and noise
    y = (
        0.5 * X[:, 0] + 
        0.2 * X[:, 1]**2 + 
        0.1 * X[:, 2] * X[:, 3] + 
        0.5 * np.sin(X[:, 4]) + 
        0.5 * np.random.normal(0, 1, n_samples)
    )
    
    # Split into train, validation, and test sets
    X_train = X[:6000]
    X_valid = X[6000:8000]
    X_test = X[8000:]
    y_train = y[:6000]
    y_valid = y[6000:8000]
    y_test = y[8000:]
    
    # OLS
    print("\nTraining OLS...")
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_ols = cal_r2(y_test, y_pred)
    print("OLS R² score:", r2_ols)
    results = {'OLS': r2_ols}
    
    # Random Forest
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_rf = cal_r2(y_test, y_pred)
    print("RF R² score:", r2_rf)
    results['RF'] = r2_rf
    
    # GBRT
    print("\nTraining GBRT...")
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_gbrt = cal_r2(y_test, y_pred)
    print("GBRT R² score:", r2_gbrt)
    results['GBRT'] = r2_gbrt
    
    # Neural Network
    print("\nTraining Neural Network...")
    # Convert data to TensorFlow tensors
    X_train_tf = tf.cast(X_train, tf.float32)
    y_train_tf = tf.cast(y_train.reshape(-1, 1), tf.float32)
    X_test_tf = tf.cast(X_test, tf.float32)
    
    # Define a simple neural network
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.1),  # Dropout for regularization
        tf.keras.layers.Dense(4, activation='relu'),  # Hidden layer with 4 neurons
        tf.keras.layers.Dense(4, activation='relu'),  # Second hidden layer with 4 neurons
        tf.keras.layers.Dense(1, activation='linear')  # Output layer
    ])
    
    # Compile the model
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    
    # Train the model
    nn_model.fit(X_train_tf, y_train_tf, 
                epochs=50, 
                batch_size=1000, 
                verbose=0)
    
    # Predict and calculate R²
    y_pred = nn_model.predict(X_test_tf).flatten()
    r2_nn = cal_r2(y_test, y_pred)
    print("NN R² score:", r2_nn)
    results['NN'] = r2_nn
    
    # TREENN1 - Neural Network with optimized single tree prediction as feature
    print("\nTraining TREENN1 (Neural Network with Decision Tree feature)...")
    
    # Scale targets for better tree learning
    y_train_scaled = (y_train - np.mean(y_train)) / np.std(y_train)
    
    # Train an optimized decision tree
    tree = DecisionTreeRegressor(
        max_depth=4,                 # Slightly deeper tree for better signal
        min_samples_split=20,        # Require at least 20 samples to consider a split
        min_samples_leaf=10,         # Minimum samples in each leaf
        min_weight_fraction_leaf=0.0001,  # Small minimum weight as regularization
        max_features=0.7,            # Use 70% of features for each split
        random_state=0
    )
    tree.fit(X_train, y_train_scaled)
    
    # Get tree predictions
    tree_train_pred = tree.predict(X_train) * np.std(y_train) + np.mean(y_train)
    tree_test_pred = tree.predict(X_test) * np.std(y_train) + np.mean(y_train)
    
    # Evaluate tree performance independently
    tree_train_r2 = cal_r2(y_train, tree_train_pred)
    tree_test_r2 = cal_r2(y_test, tree_test_pred)
    
    print(f"Tree alone - Train R²: {tree_train_r2:.4f}, Test R²: {tree_test_r2:.4f}")
    print(f"Tree structure - Nodes: {tree.tree_.node_count}, Max depth: {tree.tree_.max_depth}")
    
    # Correlation between prediction and true value
    tree_pred_corr = np.corrcoef(tree_train_pred, y_train)[0,1]
    print(f"Tree prediction correlation with target: {tree_pred_corr:.4f}")
    
    # Augment the data with tree predictions
    X_train_augmented = np.hstack((X_train, tree_train_pred.reshape(-1, 1)))
    X_test_augmented = np.hstack((X_test, tree_test_pred.reshape(-1, 1)))
    
    # Convert to TensorFlow tensors
    X_train_aug_tf = tf.cast(X_train_augmented, tf.float32)
    X_test_aug_tf = tf.cast(X_test_augmented, tf.float32)
    
    # Create a simple model to check importance of tree prediction
    feature_importance_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='linear')
    ])
    feature_importance_model.compile(optimizer='adam', loss='mse')
    feature_importance_model.fit(X_train_aug_tf, y_train_tf, epochs=10, verbose=0)
    
    # Check importance of tree prediction
    weights = feature_importance_model.layers[0].get_weights()[0]
    tree_weight = weights[-1][0]
    print(f"Linear weight assigned to tree prediction: {tree_weight:.6f}")
    
    # Define TREENN1 model with same architecture as NN
    treenn1_model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.1),  # Same dropout as NN
        tf.keras.layers.Dense(4, activation='relu'),  # Hidden layer with 4 neurons
        tf.keras.layers.Dense(4, activation='relu'),  # Second hidden layer with 4 neurons
        tf.keras.layers.Dense(1, activation='linear')  # Output layer
    ])
    
    # Compile the model
    treenn1_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    
    # Train the model
    treenn1_model.fit(X_train_aug_tf, y_train_tf, 
                    epochs=50, 
                    batch_size=1000, 
                    verbose=0)
    
    # Predict and calculate R²
    y_pred = treenn1_model.predict(X_test_aug_tf).flatten()
    r2_treenn1 = cal_r2(y_test, y_pred)
    print("TREENN1 R² score:", r2_treenn1)
    results['TREENN1'] = r2_treenn1
    
    # Get first layer weights to analyze tree importance
    weights = treenn1_model.layers[1].get_weights()[0]  # Skip dropout layer
    mean_feature_weight = np.mean(np.abs(weights[:-1, :]))
    tree_feature_weight = np.mean(np.abs(weights[-1, :]))
    print(f"Mean abs weight - Original features: {mean_feature_weight:.6f}")
    print(f"Mean abs weight - Tree prediction: {tree_feature_weight:.6f}")
    
    # Print summary
    print("\nSUMMARY OF RESULTS (R² scores):")
    for method, r2 in results.items():
        print(f"{method}: {r2:.4f}")
    
    # Improvement of TREENN1 over NN
    improvement = (r2_treenn1 - r2_nn) / r2_nn * 100
    print(f"\nTREENN1 improvement over NN: {improvement:.2f}%")
    
    return results

################################################################################
# Main function to run the asset pricing analysis with all models
def run_asset_pricing_with_trann(data_path='ML_sample.dta', seed=42):
    # Set global seeds for reproducibility
    SEED = seed
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    
    print("Loading and preparing data...")
    
    # Load data
    # Remove microcap stocks whose market capitalization fall below the 20th NYSE cutpoint
    # lme   : 1-month-lagged market equity
    # retadj: stock monthly return (adjust for delisting)
    # exret : excess return (= retadj - rf)
    # rf    : risk-free rate
    retdata = pd.read_stata(data_path)
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
    
    print(f"Data shapes: X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"Data shapes: X_valid: {X_valid.shape}, Y_valid: {Y_valid.shape}")
    print(f"Data shapes: X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    
    del retdata

    results = {}

    ################################################################################
    # Ordinary Least Squares (OLS)
    print("\nTraining OLS model...")
    OLS = linear_model.LinearRegression()
    OLS.fit(X_train, Y_train)
    Y_pred = OLS.predict(X_test)
    R2_OLS = cal_r2(Y_test, Y_pred)
    results['OLS'] = R2_OLS
    print('R2 - OLS  (%):', R2_OLS)

    ################################################################################
    # Random Forest (RF)
    print("\nTraining RF models...")
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
        print(f'Training (RF): {i + 1} / {n_hp_ls}')
        np.random.seed(SEED + i)  # Different seed for each RF model
        RF = RandomForestRegressor(max_depth=hp_i[0], n_estimators=hp_i[1], max_features=hp_i[2], n_jobs=4, random_state=SEED + i)
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
    results['RF'] = R2_RF
    print('R2 - RF   (%):', R2_RF)

    ################################################################################
    # Gradient Boosting Regression Tree (GBRT)
    print("\nTraining GBRT models...")
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
        print(f'Training (GBRT): {i + 1} / {n_hp_ls}')
        np.random.seed(SEED + i)  # Different seed for each GBRT model
        GBRT = GradientBoostingRegressor(max_depth=hp_i[0], n_estimators=hp_i[1], learning_rate=hp_i[2], max_features='sqrt', random_state=SEED + i)
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
    results['GBRT'] = R2_GBRT
    print('R2 - GBRT (%):', R2_GBRT)

    ################################################################################
    # Neural Networks (NN)
    print("\nTraining NN models...")
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
        print(f'Training (NN): {i + 1} / {n_hp_ls}')
        tf.random.set_seed(SEED + i)  # Different seed for each NN model
        np.random.seed(SEED + i)  # Also set numpy seed for consistency
        
        # Create NN model with the same architecture as in the original paper
        # Two hidden layers with 4 neurons each, followed by a linear output layer
        NN = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(hp_i[0]),  # Dropout for regularization
            tf.keras.layers.Dense(4, activation='relu'),  # First hidden layer: 4 neurons
            tf.keras.layers.Dense(4, activation='relu'),  # Second hidden layer: 4 neurons
            tf.keras.layers.Dense(1, activation='linear')  # Output layer: linear activation
        ])
        
        # Compile with Adam optimizer and MSE loss
        NN.compile(optimizer=tf.keras.optimizers.Adam(hp_i[1]), loss='mse')
        
        # Train the model
        NN.fit(x=X_train_tf, y=Y_train_tf, batch_size=hp_i[2], epochs=hp_i[3], verbose=0)
        
        # Validate on validation set
        Y_pred = np.reshape(NN(X_valid_tf, training=False).numpy(), [-1])
        R2_valid_ls[i] = cal_r2(Y_valid, Y_pred)
        NN_ls.append(NN)

    # Testing with ensemble of best models (select top navg models based on validation R2)
    avg_models_id = np.argsort(-R2_valid_ls)[:navg]
    for i, mid in enumerate(avg_models_id):
        if i == 0:
            Y_pred = np.reshape(NN_ls[mid](X_test_tf, training=False).numpy(), [-1]) / navg
        else:
            Y_pred = Y_pred + np.reshape(NN_ls[mid](X_test_tf, training=False).numpy(), [-1]) / navg
    R2_NN = cal_r2(Y_test, Y_pred)
    results['NN'] = R2_NN
    print('R2 - NN   (%):', R2_NN)

    ################################################################################
    # TrANN1: Tree Neural Network Type 1 (Tree at Input Layer)
    print("\nTraining TREENN1 models...")
    
    # First, train an optimized decision tree
    print("Training optimized decision tree for TREENN1 input layer...")
    
    # Train a decision tree with carefully tuned parameters
    np.random.seed(SEED)  # Reset seed for tree
    tree = DecisionTreeRegressor(
        max_depth=2,                 # Keep tree simple
        min_samples_split=100,       # Require many samples for splits
        min_samples_leaf=50,         # Require many samples in leaves
        max_features=0.5,            # Use 50% of features for each split
        random_state=SEED
    )
    
    # Train the tree
    tree.fit(X_train, Y_train)
    
    # Get tree predictions
    tree_preds_train = tree.predict(X_train)
    tree_preds_valid = tree.predict(X_valid)
    tree_preds_test = tree.predict(X_test)
    
    # Print tree feature importance
    feature_importance = pd.DataFrame({
        'feature': CharsVars,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTREENN1 Tree Feature Importance:")
    print(feature_importance)
    
    # Normalize tree predictions to [0,1] range
    tree_preds_train_norm = (tree_preds_train - np.min(tree_preds_train)) / (np.max(tree_preds_train) - np.min(tree_preds_train))
    tree_preds_valid_norm = (tree_preds_valid - np.min(tree_preds_train)) / (np.max(tree_preds_train) - np.min(tree_preds_train))
    tree_preds_test_norm = (tree_preds_test - np.min(tree_preds_train)) / (np.max(tree_preds_train) - np.min(tree_preds_train))
    
    # Augment the datasets with normalized tree predictions
    X_train_augmented = np.hstack((X_train, tree_preds_train_norm.reshape(-1, 1)))
    X_valid_augmented = np.hstack((X_valid, tree_preds_valid_norm.reshape(-1, 1)))
    X_test_augmented = np.hstack((X_test, tree_preds_test_norm.reshape(-1, 1)))
    
    # Transform augmented numpy to tensor
    X_train_augmented_tf = tf.cast(X_train_augmented, tf.float32)
    X_valid_augmented_tf = tf.cast(X_valid_augmented, tf.float32)
    X_test_augmented_tf = tf.cast(X_test_augmented, tf.float32)
    
    # Use the same hyperparameter grid as the NN model
    hp_ls = []
    for dropout in [0.05, 0.10]:
        for lr in [0.01, 0.05]:
            for batch_size in [1000, 5000]:
                for epoch in [20, 50]:
                    hp_ls.append((dropout, lr, batch_size, epoch))
    
    # Training & validation
    TREENN1_ls = []
    n_hp_ls = len(hp_ls)
    R2_valid_ls = np.full(n_hp_ls, np.nan)
    
    # Initialize best validation R²
    best_valid_r2 = -float('inf')
    
    for i, hp_i in enumerate(hp_ls):
        print(f"Training (TREENN1): {i + 1} / {n_hp_ls}")
        
        # Set TensorFlow seed for reproducibility
        tf.random.set_seed(SEED + i)
        
        # Create TREENN1 model with same architecture as NN
        TREENN1 = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(hp_i[0]),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile with Adam optimizer and MSE loss
        TREENN1.compile(optimizer=tf.keras.optimizers.Adam(hp_i[1]), loss='mse')
        
        # Train the model
        TREENN1.fit(x=X_train_augmented_tf, y=Y_train_tf,
                   batch_size=hp_i[2], epochs=hp_i[3], verbose=0)
        
        # Validate
        Y_pred = np.reshape(TREENN1(X_valid_augmented_tf, training=False).numpy(), [-1])
        r2 = cal_r2(Y_valid, Y_pred)
        R2_valid_ls[i] = r2
        TREENN1_ls.append(TREENN1)
        
        if r2 > best_valid_r2:
            best_valid_r2 = r2
            print(f"  New best validation R²: {r2:.4f}%")
    
    # Testing with ensemble of best models
    avg_models_id = np.argsort(-R2_valid_ls)[:navg]
    for i, mid in enumerate(avg_models_id):
        if i == 0:
            Y_pred = np.reshape(TREENN1_ls[mid](X_test_augmented_tf, training=False).numpy(), [-1]) / navg
        else:
            Y_pred = Y_pred + np.reshape(TREENN1_ls[mid](X_test_augmented_tf, training=False).numpy(), [-1]) / navg
    
    R2_TREENN1 = cal_r2(Y_test, Y_pred)
    results['TREENN1'] = R2_TREENN1
    print('R2 - TREENN1 (%):', R2_TREENN1)
    
    ################################################################################
    # FONN1: Forest of Trees Neural Network Type 1 (Ensemble of Trees at Input Layer)
    print("\nTraining FONN1 models...")
    
    # Train 5 different trees with different seeds
    print("Training ensemble of 5 trees for FONN1 input layer...")
    
    # Initialize list to store trees and their predictions
    trees = []
    tree_preds_train = np.zeros((X_train.shape[0], 5))
    tree_preds_valid = np.zeros((X_valid.shape[0], 5))
    tree_preds_test = np.zeros((X_test.shape[0], 5))
    
    # Train 5 trees with different seeds
    for i in range(5):
        np.random.seed(SEED + i)  # Different seed for each tree
        tree = DecisionTreeRegressor(
            max_depth=2,                 # Keep tree simple
            min_samples_split=100,       # Require many samples for splits
            min_samples_leaf=50,         # Require many samples in leaves
            max_features=0.5,            # Use 50% of features for each split
            random_state=SEED + i
        )
        
        # Train the tree
        tree.fit(X_train, Y_train)
        trees.append(tree)
        
        # Get predictions
        tree_preds_train[:, i] = tree.predict(X_train)
        tree_preds_valid[:, i] = tree.predict(X_valid)
        tree_preds_test[:, i] = tree.predict(X_test)
        
        # Print tree feature importance
        feature_importance = pd.DataFrame({
            'feature': CharsVars,
            'importance': tree.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nFONN1 Tree {i+1} Feature Importance:")
        print(feature_importance)
    
    # Normalize tree predictions to [0,1] range for each tree
    for i in range(5):
        min_val = np.min(tree_preds_train[:, i])
        max_val = np.max(tree_preds_train[:, i])
        tree_preds_train[:, i] = (tree_preds_train[:, i] - min_val) / (max_val - min_val)
        tree_preds_valid[:, i] = (tree_preds_valid[:, i] - min_val) / (max_val - min_val)
        tree_preds_test[:, i] = (tree_preds_test[:, i] - min_val) / (max_val - min_val)
    
    # Augment the datasets with normalized tree predictions
    X_train_augmented = np.hstack((X_train, tree_preds_train))
    X_valid_augmented = np.hstack((X_valid, tree_preds_valid))
    X_test_augmented = np.hstack((X_test, tree_preds_test))
    
    # Transform augmented numpy to tensor
    X_train_augmented_tf = tf.cast(X_train_augmented, tf.float32)
    X_valid_augmented_tf = tf.cast(X_valid_augmented, tf.float32)
    X_test_augmented_tf = tf.cast(X_test_augmented, tf.float32)
    
    # Use the same hyperparameter grid as the NN model
    hp_ls = []
    for dropout in [0.05, 0.10]:
        for lr in [0.01, 0.05]:
            for batch_size in [1000, 5000]:
                for epoch in [20, 50]:
                    hp_ls.append((dropout, lr, batch_size, epoch))
    
    # Training & validation
    FONN1_ls = []
    n_hp_ls = len(hp_ls)
    R2_valid_ls = np.full(n_hp_ls, np.nan)
    
    # Initialize best validation R²
    best_valid_r2 = -float('inf')
    
    for i, hp_i in enumerate(hp_ls):
        print(f"Training (FONN1): {i + 1} / {n_hp_ls}")
        
        # Set TensorFlow seed for reproducibility
        tf.random.set_seed(SEED + i)
        
        # Create FONN1 model with same architecture as NN
        FONN1 = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(hp_i[0]),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile with Adam optimizer and MSE loss
        FONN1.compile(optimizer=tf.keras.optimizers.Adam(hp_i[1]), loss='mse')
        
        # Train the model
        FONN1.fit(x=X_train_augmented_tf, y=Y_train_tf,
                 batch_size=hp_i[2], epochs=hp_i[3], verbose=0)
        
        # Validate
        Y_pred = np.reshape(FONN1(X_valid_augmented_tf, training=False).numpy(), [-1])
        r2 = cal_r2(Y_valid, Y_pred)
        R2_valid_ls[i] = r2
        FONN1_ls.append(FONN1)
        
        if r2 > best_valid_r2:
            best_valid_r2 = r2
            print(f"  New best validation R²: {r2:.4f}%")
            
            # Print importance of tree predictions
            weights = FONN1.layers[1].get_weights()[0]  # Skip dropout layer
            tree_weights = np.mean(np.abs(weights[-5:, :]), axis=1)
            print("\nTree Prediction Weights:")
            for j, w in enumerate(tree_weights):
                print(f"Tree {j+1}: {w:.6f}")
    
    # Testing with ensemble of best models
    avg_models_id = np.argsort(-R2_valid_ls)[:navg]
    for i, mid in enumerate(avg_models_id):
        if i == 0:
            Y_pred = np.reshape(FONN1_ls[mid](X_test_augmented_tf, training=False).numpy(), [-1]) / navg
        else:
            Y_pred = Y_pred + np.reshape(FONN1_ls[mid](X_test_augmented_tf, training=False).numpy(), [-1]) / navg
    
    R2_FONN1 = cal_r2(Y_test, Y_pred)
    results['FONN1'] = R2_FONN1
    print('R2 - FONN1 (%):', R2_FONN1)
    
    # Compare with other models
    fonn1_vs_nn = (R2_FONN1 - R2_NN) / R2_NN * 100
    fonn1_vs_treenn1 = (R2_FONN1 - R2_TREENN1) / R2_TREENN1 * 100
    print(f"FONN1 improvement over NN: {fonn1_vs_nn:.2f}%")
    print(f"FONN1 improvement over TREENN1: {fonn1_vs_treenn1:.2f}%")
    
    ################################################################################
    # TREENN3: Tree Neural Network Type 3 (Tree at Output Layer)
    print("\nTraining TREENN3 models...")
    
    # First, train the neural network part
    print("Training neural network component...")
    
    # Use the same hyperparameter grid as the NN model
    hp_ls = []
    for dropout in [0.05, 0.10]:
        for lr in [0.01, 0.05]:
            for batch_size in [1000, 5000]:
                for epoch in [20, 50]:
                    hp_ls.append((dropout, lr, batch_size, epoch))
    
    # Training & validation
    TREENN3_ls = []
    n_hp_ls = len(hp_ls)
    R2_valid_ls = np.full(n_hp_ls, np.nan)
    
    # Track best model and performance
    best_valid_r2 = -float('inf')
    best_model_idx = -1
    
    for i, hp_i in enumerate(hp_ls):
        print(f"Training (TREENN3): {i + 1} / {n_hp_ls}")
        
        # Set seeds for reproducibility
        tf.random.set_seed(SEED + i)
        np.random.seed(SEED + i)
        
        # Create the neural network part with same architecture as NN
        nn_model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(hp_i[0]),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile and train the neural network
        nn_model.compile(optimizer=tf.keras.optimizers.Adam(hp_i[1]), loss='mse')
        nn_model.fit(x=X_train_tf, y=Y_train_tf, batch_size=hp_i[2], epochs=hp_i[3], verbose=0)
        
        # Get neural network predictions
        nn_train_pred = nn_model.predict(X_train_tf).flatten()
        nn_valid_pred = nn_model.predict(X_valid_tf).flatten()
        nn_test_pred = nn_model.predict(X_test_tf).flatten()
        
        # Train a decision tree on NN predictions
        tree = DecisionTreeRegressor(
            max_depth=2,                 # Keep tree simple
            min_samples_split=100,       # Require many samples for splits
            min_samples_leaf=50,         # Require many samples in leaves
            max_features=0.5,            # Use 50% of features for each split
            random_state=SEED + i
        )
        
        # Train the tree on NN predictions
        tree.fit(nn_train_pred.reshape(-1, 1), Y_train)
        
        # Get tree predictions
        tree_train_pred = tree.predict(nn_train_pred.reshape(-1, 1))
        tree_valid_pred = tree.predict(nn_valid_pred.reshape(-1, 1))
        tree_test_pred = tree.predict(nn_test_pred.reshape(-1, 1))
        
        # Calculate R² using tree predictions
        r2 = cal_r2(Y_valid, tree_valid_pred)
        R2_valid_ls[i] = r2
        
        # Store models
        TREENN3_ls.append((nn_model, tree))
        
        # Track best model
        if r2 > best_valid_r2:
            best_valid_r2 = r2
            best_model_idx = i
            print(f"  New best validation R²: {r2:.4f}%")
            print(f"  NN R²: {cal_r2(Y_valid, nn_valid_pred):.4f}%")
            print(f"  Tree on NN output R²: {r2:.4f}%")
    
    # Testing with ensemble of best models
    avg_models_id = np.argsort(-R2_valid_ls)[:navg]
    for i, mid in enumerate(avg_models_id):
        nn_model, tree = TREENN3_ls[mid]
        
        # Get predictions
        nn_test_pred = nn_model.predict(X_test_tf).flatten()
        # Tree processes NN output
        tree_test_pred = tree.predict(nn_test_pred.reshape(-1, 1))
        
        if i == 0:
            Y_pred = tree_test_pred / navg
        else:
            Y_pred = Y_pred + tree_test_pred / navg
    
    # Calculate final R2 for TREENN3
    R2_TREENN3 = cal_r2(Y_test, Y_pred)
    results['TREENN3'] = R2_TREENN3
    print('R2 - TREENN3 (%):', R2_TREENN3)
    
    # Compare with other models
    treenn3_vs_nn = (R2_TREENN3 - R2_NN) / R2_NN * 100
    treenn3_vs_treenn1 = (R2_TREENN3 - R2_TREENN1) / R2_TREENN1 * 100
    print(f"TREENN3 improvement over NN: {treenn3_vs_nn:.2f}%")
    print(f"TREEN3 improvement over TREENN1: {treenn3_vs_treenn1:.2f}%")
    
    ################################################################################
    # FONN3: Forest of Trees Neural Network Type 3 (Ensemble of Trees at Output Layer)
    print("\nTraining FONN3 models...")
    
    # First, train the neural network part
    print("Training neural network component...")
    
    # Use the same hyperparameter grid as the NN model
    hp_ls = []
    for dropout in [0.05, 0.10]:
        for lr in [0.01, 0.05]:
            for batch_size in [1000, 5000]:
                for epoch in [20, 50]:
                    hp_ls.append((dropout, lr, batch_size, epoch))
    
    # Training & validation
    FONN3_ls = []
    n_hp_ls = len(hp_ls)
    R2_valid_ls = np.full(n_hp_ls, np.nan)
    
    # Track best model and performance
    best_valid_r2 = -float('inf')
    best_model_idx = -1
    
    for i, hp_i in enumerate(hp_ls):
        print(f"Training (FONN3): {i + 1} / {n_hp_ls}")
        
        # Set seeds for reproducibility
        tf.random.set_seed(SEED + i)
        np.random.seed(SEED + i)
        
        # Create the neural network part with same architecture as NN
        nn_model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(hp_i[0]),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile and train the neural network
        nn_model.compile(optimizer=tf.keras.optimizers.Adam(hp_i[1]), loss='mse')
        nn_model.fit(x=X_train_tf, y=Y_train_tf, batch_size=hp_i[2], epochs=hp_i[3], verbose=0)
        
        # Get neural network predictions
        nn_train_pred = nn_model.predict(X_train_tf).flatten()
        nn_valid_pred = nn_model.predict(X_valid_tf).flatten()
        nn_test_pred = nn_model.predict(X_test_tf).flatten()
        
        # Train 5 different trees on NN predictions with different seeds
        trees = []
        tree_preds_train = np.zeros((X_train.shape[0], 5))
        tree_preds_valid = np.zeros((X_valid.shape[0], 5))
        tree_preds_test = np.zeros((X_test.shape[0], 5))
        
        # Store individual tree R² scores for weighting
        tree_r2_scores = np.zeros(5)
        
        for j in range(5):
            # Train a decision tree on NN predictions
            tree = DecisionTreeRegressor(
                max_depth=2,                 # Keep tree simple
                min_samples_split=100,       # Require many samples for splits
                min_samples_leaf=50,         # Require many samples in leaves
                max_features=0.5,            # Use 50% of features for each split
                random_state=SEED + i + j    # Different seed for each tree
            )
            
            # Train the tree on NN predictions
            tree.fit(nn_train_pred.reshape(-1, 1), Y_train)
            trees.append(tree)
            
            # Get tree predictions
            tree_preds_train[:, j] = tree.predict(nn_train_pred.reshape(-1, 1))
            tree_preds_valid[:, j] = tree.predict(nn_valid_pred.reshape(-1, 1))
            tree_preds_test[:, j] = tree.predict(nn_test_pred.reshape(-1, 1))
            
            # Calculate individual tree R² score
            tree_r2_scores[j] = cal_r2(Y_valid, tree_preds_valid[:, j])
        
        # Calculate weights based on R² scores (softmax to ensure weights sum to 1)
        weights = np.exp(tree_r2_scores) / np.sum(np.exp(tree_r2_scores))
        
        # Weighted average of tree predictions
        tree_train_pred = np.sum(tree_preds_train * weights, axis=1)
        tree_valid_pred = np.sum(tree_preds_valid * weights, axis=1)
        tree_test_pred = np.sum(tree_preds_test * weights, axis=1)
        
        # Calculate R² using weighted tree predictions
        r2 = cal_r2(Y_valid, tree_valid_pred)
        R2_valid_ls[i] = r2
        
        # Store models and weights
        FONN3_ls.append((nn_model, trees, weights))
        
        # Track best model
        if r2 > best_valid_r2:
            best_valid_r2 = r2
            best_model_idx = i
            print(f"  New best validation R²: {r2:.4f}%")
            print(f"  NN R²: {cal_r2(Y_valid, nn_valid_pred):.4f}%")
            print(f"  Weighted Forest of Trees on NN output R²: {r2:.4f}%")
            
            # Print individual tree R² scores and weights
            print("\nIndividual Tree Performance:")
            for j in range(5):
                print(f"Tree {j+1}: R² = {tree_r2_scores[j]:.4f}%, Weight = {weights[j]:.4f}")
    
    # Testing with ensemble of best models
    avg_models_id = np.argsort(-R2_valid_ls)[:navg]
    for i, mid in enumerate(avg_models_id):
        nn_model, trees, weights = FONN3_ls[mid]
        
        # Get predictions
        nn_test_pred = nn_model.predict(X_test_tf).flatten()
        
        # Get predictions from all trees and combine with weights
        tree_preds_test = np.zeros((X_test.shape[0], 5))
        for j, tree in enumerate(trees):
            tree_preds_test[:, j] = tree.predict(nn_test_pred.reshape(-1, 1))
        tree_test_pred = np.sum(tree_preds_test * weights, axis=1)
        
        if i == 0:
            Y_pred = tree_test_pred / navg
        else:
            Y_pred = Y_pred + tree_test_pred / navg
    
    # Calculate final R2 for FONN3
    R2_FONN3 = cal_r2(Y_test, Y_pred)
    results['FONN3'] = R2_FONN3
    print('R2 - FONN3 (%):', R2_FONN3)
    
    # Compare with other models
    fonn3_vs_nn = (R2_FONN3 - R2_NN) / R2_NN * 100
    fonn3_vs_treenn1 = (R2_FONN3 - R2_TREENN1) / R2_TREENN1 * 100
    fonn3_vs_fonn1 = (R2_FONN3 - R2_FONN1) / R2_FONN1 * 100
    print(f"FONN3 improvement over NN: {fonn3_vs_nn:.2f}%")
    print(f"FONN3 improvement over TREENN1: {fonn3_vs_treenn1:.2f}%")
    print(f"FONN3 improvement over FONN1: {fonn3_vs_fonn1:.2f}%")
    
    # Print summary of all results
    print("\nFinal Results Summary:")
    for model, r2 in results.items():
        print(f"R2 - {model} (%): {r2:.4f}")
    
    return results

# Run the analysis if executed directly
if __name__ == "__main__":
    print("=" * 80)
    print("ASSET PRICING WITH MACHINE LEARNING METHODS INCLUDING TREENN1")
    print("=" * 80)
    
    try:
        data_path = 'ML_sample.dta'  # Adjust this path to your data file location
        if os.path.exists(data_path):
            print(f"Found data file at {data_path}")
            
            # Define 5 different random seeds
            seeds = [42, 123, 456, 789, 1011]
            
            # Store all results
            all_results = {}
            
            # Run analysis for each seed
            for seed in seeds:
                print(f"\n{'='*40}")
                print(f"Running analysis with seed: {seed}")
                print(f"{'='*40}")
                
                # Set the seed for all random number generators
                np.random.seed(seed)
                tf.random.set_seed(seed)
                random.seed(seed)
                
                # Run the analysis
                results = run_asset_pricing_with_trann(data_path, seed)
                all_results[seed] = results
                
                # Print results for this seed
                print(f"\nResults for seed {seed}:")
                for method, r2 in results.items():
                    print(f"{method}: {r2:.4f}%")
            
            # Save results to a report file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_filename = f"ML_Results_Report_{timestamp}.txt"
            
            with open(report_filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("ASSET PRICING WITH MACHINE LEARNING METHODS - DETAILED RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Report generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
                
                # Write results for each seed
                for seed in seeds:
                    f.write(f"\n{'='*40}\n")
                    f.write(f"RESULTS FOR SEED {seed}\n")
                    f.write(f"{'='*40}\n\n")
                    
                    results = all_results[seed]
                    for method, r2 in results.items():
                        f.write(f"{method}: {r2:.4f}%\n")
                    
                    # Add improvement metrics
                    if 'TREENN1' in results and 'NN' in results:
                        improvement = (results['TREENN1'] - results['NN']) / results['NN'] * 100
                        f.write(f"\nTREENN1 improvement over NN: {improvement:.2f}%\n")
                    
                    f.write("\n" + "-"*40 + "\n")
                
                # Add summary of improvements
                f.write("\nSUMMARY OF IMPROVEMENTS\n")
                f.write("=" * 40 + "\n")
                for seed in seeds:
                    results = all_results[seed]
                    if 'TREENN1' in results and 'NN' in results:
                        improvement = (results['TREENN1'] - results['NN']) / results['NN'] * 100
                        f.write(f"Seed {seed}: TREENN1 improvement over NN: {improvement:.2f}%\n")
            
            print(f"\nDetailed results have been saved to: {report_filename}")
            
        else:
            print(f"Data file not found at {data_path}")
            print("Please make sure the data file exists at the specified path.")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check the error and try again.") 