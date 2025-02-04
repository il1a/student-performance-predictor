#!/usr/bin/env python3
"""
OLS Activation Inference Script

Loads the trained OLS model (currentOlsSolution.pkl) and
reads a single-entry activation_data.csv to produce a prediction.
Also prints the actual target value for easy comparison.
"""

import os
import pickle
import statsmodels.api as sm
import pandas as pd

# 0. Set the script path
script_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    # 1. Load the OLS Model
    model_path = os.path.join(script_dir, "../knowledgeBase/currentOlsSolution.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        ols_model = pickle.load(f)
    print(f"Loaded OLS model from: {model_path}")

    # 2. Load the Activation Data
    activation_path = os.path.join(script_dir, "../activationBase/activation_data.csv")
    if not os.path.exists(activation_path):
        raise FileNotFoundError(f"Activation data not found: {activation_path}")

    activation_df = pd.read_csv(activation_path)
    print(f"Loaded activation data from: {activation_path}")
    print(f"Data shape: {activation_df.shape}")

    # 3. Check if 'Exam_Score' target column is present
    target_col = "Exam_Score"
    if target_col in activation_df.columns:
        X_activation = activation_df.drop(columns=[target_col])
        y_actual = activation_df[target_col].values  # Extract the actual target values
    else:
        X_activation = activation_df
        y_actual = None

    # 4. Add constant to match OLS training
    X_activation_ols = sm.add_constant(X_activation, prepend=True, has_constant='add')

    # 5. Make Prediction
    y_pred_OLS = ols_model.predict(X_activation_ols)
    print("OLS Predictions for Activation Data:\n", y_pred_OLS)

    # 6. Print Actual Values (if available)
    if y_actual is not None:
        print("Actual Exam_Score values:\n", y_actual)

if __name__ == "__main__":
    main()
