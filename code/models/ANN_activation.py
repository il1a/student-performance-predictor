#!/usr/bin/env python3
"""
ANN Activation Inference Script

Loads the trained TensorFlow ANN model (currentAiSolution.keras) and
reads a single-entry activation_data.csv to produce a prediction.
Also prints the actual target value for easy comparison.
"""

import os
import pandas as pd
from tensorflow.keras.models import load_model

# 0. Set the script path
script_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    # 1. Load the ANN Model
    model_path = os.path.join(script_dir, "../knowledgeBase/currentAiSolution.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    ann_model = load_model(model_path)
    print(f"Loaded ANN model from: {model_path}")

    # 2. Load the Activation Data
    activation_path = os.path.join(script_dir, "../activationBase/activation_data.csv")
    if not os.path.exists(activation_path):
        raise FileNotFoundError(f"Activation data not found: {activation_path}")

    activation_df = pd.read_csv(activation_path)
    print(f"Loaded activation data from: {activation_path}")
    print(f"Data shape: {activation_df.shape}")

    # 3. Check if 'Exam_Score' is present
    target_col = "Exam_Score"
    if target_col in activation_df.columns:
        X_activation = activation_df.drop(columns=[target_col]).values
        y_actual = activation_df[target_col].values  # Extract the actual target values
    else:
        X_activation = activation_df.values
        y_actual = None

    # 4. Make Prediction
    y_pred_ANN = ann_model.predict(X_activation).flatten()
    print("ANN Predictions for Activation Data:\n", y_pred_ANN)

    # 5. Print Actual Values (if available)
    if y_actual is not None:
        print("Actual Exam_Score values:\n", y_actual)

if __name__ == "__main__":
    main()
