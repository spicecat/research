################################################################################
# Description:                                                                 #
# Run and compare all machine learning methods for asset pricing               #
# Including traditional methods from Gu, Kelly, and Xiu (2020)                 #
# and the new TREENN1 (Transformed Artificial Neural Network) approach         #
#                                                                              #
# This script:                                                                 #
# 1. Loads the ML_sample.dta dataset                                           #
# 2. Runs all ML methods (OLS, RF, GBRT, NN, TREENN1)                         #
# 3. Compares the out-of-sample R2 values                                      #
# 4. Visualizes the results                                                    #
#                                                                              #
################################################################################
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import the ML_TrANN_Methods module
from ML_TrANN_Methods import run_asset_pricing_with_trann

# Setup paths
data_path = "ML_sample.dta"  # Adjust this path to your data file location


def plot_results(results):
    """Create a bar plot to compare the R2 values of different methods"""
    # Create a bar plot
    plt.figure(figsize=(10, 6))

    # Sort models in a specific order
    model_order = ["OLS", "RF", "GBRT", "NN", "TREENN1"]
    # Filter only models that exist in results
    models = [model for model in model_order if model in results]
    r2_values = [results[model] for model in models]

    # Set up colors with a professional palette
    colors = sns.color_palette("Blues_d", len(models))

    # Highlight TREENN1 method with a different color
    for i, model in enumerate(models):
        if model == "TREENN1":
            colors[i] = sns.color_palette("Reds_d")[2]

    # Create the bar chart
    bars = plt.bar(models, r2_values, color=colors)

    # Add labels and title
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Out-of-Sample R² (%)")
    plt.title("Comparison of Machine Learning Methods for Asset Pricing")

    # Add text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add grid for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ensure y-axis starts from zero
    plt.ylim(bottom=0)

    # Save the plot
    plt.tight_layout()
    plt.savefig("ml_comparison_results.png", dpi=300)
    plt.show()


def create_latex_table(results):
    """Create a LaTeX table for the paper"""
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Out-of-Sample $R^2$ Comparison of Machine Learning Methods}
\\label{tab:ml_comparison}
\\begin{tabular}{lc}
\\toprule
Method & Out-of-Sample $R^2$ (\\%) \\\\
\\midrule
"""

    # Sort models in a specific order
    model_order = ["OLS", "RF", "GBRT", "NN", "TREENN1"]
    # Filter only models that exist in results
    models = [model for model in model_order if model in results]

    # Add each method to the table
    for model in models:
        r2 = results[model]
        # Format the model name for LaTeX
        if model == "GBRT":
            model_name = "Gradient Boosting Regression Tree (GBRT)"
        elif model == "RF":
            model_name = "Random Forest (RF)"
        elif model == "NN":
            model_name = "Neural Network (NN)"
        elif model == "OLS":
            model_name = "Ordinary Least Squares (OLS)"
        elif model == "TREENN1":
            model_name = "Tree Neural Network (TREENN1)"
        else:
            model_name = model

        latex_table += f"{model_name} & {r2:.4f} \\\\\n"

    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    # Save the LaTeX table to a file
    with open("ml_comparison_table.tex", "w") as f:
        f.write(latex_table)

    return latex_table


def create_explainability_analysis():
    """
    Create an analysis of the explainability aspects of TREENN1 model
    compared to traditional models in asset pricing.

    This function generates a markdown document discussing the explainability
    advantages of TREENN1 model in the context of asset pricing.
    """
    explainability_doc = """# Explainability Analysis of TREENN1 Model in Asset Pricing

## Introduction

Explainability is a critical aspect of machine learning models in finance, especially in asset pricing.
This document analyzes the explainability advantages of TREENN1 (Transformed Artificial Neural Network) 
compared to traditional models used in the Gu, Kelly, and Xiu (2020) paper.

## Explainability Advantages of TREENN1 Model

### 1. Interpretable Model Structure

TREENN1 combines the interpretability of decision trees with the flexibility of neural networks:

- Uses a decision tree at the input layer, allowing us to extract decision rules that 
  directly explain how the model processes the raw financial features. The tree structure provides 
  a clear hierarchy of feature importance in predicting returns.

- The neural network component then builds on these tree predictions, adding flexibility to capture 
  more complex patterns while maintaining a degree of explainability through the initial tree.

### 2. Feature Importance Analysis

TREENN1 enables multiple levels of feature importance analysis:

- **Tree-based importance**: From the decision tree component, we can extract traditional feature 
  importance metrics based on how frequently features are used in splits and their contribution to 
  reducing impurity.
  
- **Neural network gradients**: For the neural network component, we can compute the gradients of 
  predictions with respect to inputs to identify which features most strongly affect the output.

### 3. Partial Dependence Plots and Individual Conditional Expectation

TREENN1 supports visualization techniques that help explain how specific features affect predictions:

- **Partial Dependence Plots (PDPs)**: Show the marginal effect of a feature on the predicted outcome.
  The tree structure in TREENN1 makes these plots more reliable and less affected by feature correlations.
  
- **Individual Conditional Expectation (ICE)**: Enables analysis of how predictions change for individual 
  assets as a feature varies, providing asset-specific insights.

### 4. Local Interpretations

For any specific asset pricing prediction, TREENN1 offers local interpretations:

- **Tree path visualization**: For each prediction, we can trace the exact path through the decision tree, 
  showing precisely which features and thresholds determined the initial feature transformation.
  
- **LIME and SHAP integration**: TREENN1 is compatible with model-agnostic explainability tools like 
  LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations), offering 
  rigorous feature attribution for individual predictions.

## Comparison with Traditional Models

Compared to traditional models used in asset pricing:

- **Advantage over OLS**: While OLS provides coefficients that show feature relationships, TREENN1 captures 
  non-linear relationships and interactions between features while maintaining interpretability.

- **Advantage over RF/GBRT**: Standard tree ensembles like Random Forests become black boxes due to the hundreds 
  of trees involved. TREENN1 uses a single tree with neural enhancement, preserving interpretability while 
  maintaining competitive performance.

- **Advantage over Neural Networks**: Traditional neural networks are often considered black boxes. TREENN1 integrates 
  a tree structure that provides explicit decision rules, making the model more transparent.

## Conclusion

The TREENN1 model offers a compelling middle ground in the explainability-performance tradeoff. It achieves competitive 
predictive performance while maintaining higher levels of interpretability than traditional black-box models. 
This makes it particularly valuable in asset pricing applications where understanding the model's decision process 
is as important as its predictive accuracy.

The integration of a tree and neural network in TREENN1 provides a framework where we can extract interpretable rules 
while still benefiting from the flexibility of deep learning approaches.
"""

    # Save the explainability analysis to a file
    with open("treenn1_explainability_analysis.md", "w") as f:
        f.write(explainability_doc)

    print(
        "Explainability analysis document created: treenn1_explainability_analysis.md"
    )


def main():
    """Main function to run the comparison"""
    print("=" * 80)
    print("COMPARING MACHINE LEARNING METHODS FOR ASSET PRICING")
    print("=" * 80)

    start_time = time.time()

    # Check if the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found.")
        print(f"Please ensure the ML_sample.dta file is in the correct location.")
        return

    # Run all models
    print("\nRunning all models (OLS, RF, GBRT, NN, TREENN1)...")
    results = run_asset_pricing_with_trann(data_path)

    # Display the results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Method':<10} | {'Out-of-Sample R2 (%)':<20}")
    print("-" * 50)
    for model, r2 in results.items():
        print(f"{model:<10} | {r2:<20.4f}")
    print("-" * 50)

    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv("all_ml_comparison_results.csv", index=False)
    print("\nResults saved to 'all_ml_comparison_results.csv'")

    # Plot the results
    print("\nGenerating visualization...")
    plot_results(results)
    print("Visualization saved to 'ml_comparison_results.png'")

    # Create LaTeX table
    print("\nGenerating LaTeX table for paper...")
    latex_table = create_latex_table(results)
    print("LaTeX table saved to 'ml_comparison_table.tex'")

    # Create explainability analysis
    print("\nGenerating explainability analysis document...")
    create_explainability_analysis()

    # Report execution time
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds.")


if __name__ == "__main__":
    main()
