# Asset Pricing with Machine Learning Methods Including TREENN1

This repository contains an implementation of various machine learning methods for asset pricing, including TREENN1 (Transformed Artificial Neural Network), extending the approaches used in the paper "Empirical Asset Pricing via Machine Learning" by Gu, Kelly, and Xiu (2020).

## Overview

The original research by Gu, Kelly, and Xiu (2020) compared several machine learning methods for predicting stock returns:
- Ordinary Least Squares (OLS)
- Random Forest (RF)
- Gradient Boosting Regression Tree (GBRT)
- Neural Networks (NN)

This project implements all these methods and extends them by adding the TREENN1 (Transformed Artificial Neural Network) model that combines the strengths of tree-based models and neural networks.

## What's Implemented

1. **All Original Methods**:
   - Ordinary Least Squares (OLS)
   - Random Forest (RF)
   - Gradient Boosting Regression Tree (GBRT)
   - Neural Networks (NN) - Using the exact architecture from the original paper

2. **Added Method**:
   - TREENN1: The original NN model enhanced with an optimized decision tree prediction

3. **Comparison Framework**:
   - ML_TrANN_Methods.py contains all models
   - compare_ml_methods.py visualizes and compares the results
   - Quick test mode with simulated data

## TREENN1 Model: Tree-Enhanced Neural Network

TREENN1 represents an innovative approach to combining tree-based learning with neural networks:

1. **Train a carefully tuned decision tree** on the input features
   - Use target variable scaling to improve tree learning
   - Apply optimized parameters to prevent overfitting
   - Balance model complexity with generalization ability

2. **Generate predictions** from the decision tree for all data points

3. **Augment the feature set** by adding the tree prediction as an additional feature

4. **Use the exact same neural network model** as in the original NN implementation

```python
# 1. Scale target variables for better tree learning
y_train_scaled = (y_train - np.mean(y_train)) / np.std(y_train)

# 2. Train an optimized decision tree
tree = DecisionTreeRegressor(
    max_depth=4,                 # Carefully chosen depth
    min_samples_split=20,        # Require at least 20 samples to consider a split
    min_samples_leaf=10,         # Ensure leaf stability
    min_weight_fraction_leaf=0.0001,  # Slight regularization
    max_features=0.7,            # Use 70% of features for each split
    random_state=0
)
tree.fit(X_train, y_train_scaled)

# 3. Get tree predictions and unscale them
tree_preds = (tree.predict(X) * np.std(y_train) + np.mean(y_train)).reshape(-1, 1)

# 4. Augment features
X_augmented = np.hstack((X, tree_preds))

# 5. Feed into same NN as original model
model = tf.keras.Sequential([
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
```

## The Optimized Decision Tree Approach

Our implementation carefully tunes a single decision tree to provide high-quality predictions:

1. **Target Variable Scaling**: By standardizing the target variable before tree training, we improve the tree's ability to capture the underlying patterns
2. **Parameter Optimization**: We carefully tune the tree's parameters to balance complexity with generalization 
3. **Feature Subset Selection**: Using a subset of features (70%) for each split helps prevent overfitting
4. **Leaf Stabilization**: We require a minimum number of samples in each leaf to ensure robust predictions

We also include diagnostic measurements to verify that the tree's predictions are providing useful information:

```python
# Evaluate tree performance independently
tree_train_r2 = cal_r2(y_train, tree_train_pred)
tree_valid_r2 = cal_r2(y_valid, tree_valid_pred)
tree_test_r2 = cal_r2(y_test, tree_test_pred)

# Check tree structure
print(f"Tree structure - Nodes: {tree.tree_.node_count}, Max depth: {tree.tree_.max_depth}")
print(f"Top 5 important features: {np.argsort(-tree.feature_importances_)[:5]}")

# Verify prediction quality
print(f"Correlation between tree pred and actual: {np.corrcoef(tree_preds.flatten(), y)[0,1]}")
```

## Repository Structure

- `ML_TrANN_Methods.py`: Implementation of all models (OLS, RF, GBRT, NN, TREENN1) for asset pricing
- `compare_ml_methods.py`: Script to run and compare all methods
- `ML_sample.dta`: Sample dataset for asset pricing (from Gu, Kelly, and Xiu)
- `TrANN/`: Original TrANN implementation (reference only)

## How to Run

### Requirements

First, install the required packages:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn pyreadstat
```

### Running a Quick Test with Simulated Data

To run a quick test with simulated data (useful for verifying that all models are working):

```bash
python ML_TrANN_Methods.py --quick-test
```

This will:
1. Generate simulated data with non-linear relationships
2. Run all models (OLS, RF, GBRT, NN, TREENN1) with reduced parameters
3. Compare the out-of-sample R² values
4. Display a summary of results

### Running the Full Comparison

To run the full comparison with the real dataset:

```bash
python compare_ml_methods.py
```

This will:
1. Load the ML_sample.dta dataset
2. Run all ML methods (OLS, RF, GBRT, NN, TREENN1)
3. Compare the out-of-sample R² values
4. Generate visualizations and tables of results
5. Create an explainability analysis document

### Running the Models Directly

If you want to run the models directly:

```bash
python ML_TrANN_Methods.py
```

## Benefits of TREENN1 for Asset Pricing

### 1. Enhanced Performance Potential

TREENN1 has the potential to achieve better predictive performance than the base neural network by incorporating the tree's ability to capture non-linear relationships and interactions.

### 2. Explainability

A key advantage of TREENN1 is its enhanced explainability compared to standard neural networks:

- The decision tree component provides feature importance rankings
- The tree structure can be visualized and interpreted
- We can analyze how the neural network utilizes the tree's predictions

### 3. Complementary Strengths

TREENN1 combines the complementary strengths of two different modeling approaches:
- Decision trees excel at capturing threshold effects and interactions
- Neural networks excel at learning complex patterns from high-dimensional data
- The combination can potentially outperform either approach alone

## Next Steps

- Run the full comparison with the asset pricing dataset
- Add more TrANN variants (TREENN2, TREENN3, FONN1, FONN2, FONN3)
- Analyze feature importance in the TREENN1 model
- Create visualizations to demonstrate the explainability of TREENN1

## Citation

If you use this code in your research, please cite:

```
Original Research:
Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223-2273.

TREENN1 Extension:
[Your citation information]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work builds upon the research and code from Gu, Kelly, and Xiu (2020)
- The TREENN1 implementation is inspired by research on combining tree-based models with neural networks 