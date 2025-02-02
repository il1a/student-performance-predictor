#!/usr/bin/env python3
"""
OLS Pipeline Script

This script:
1. Loads preprocessed training and test data
2. Fits an Ordinary Least Squares (OLS) regression using statsmodels
3. Evaluates the model using MSE, MAE, and R²
4. Visualizes predictions vs actual, and runs regression diagnostics
5. Saves the model, diagnostics, and plots
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Statsmodels for OLS
import statsmodels.api as sm

# Sklearn metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For saving objects
import pickle

# --- Ensure the correct working directory ---
TARGET_FOLDER = 'models'
current_dir = os.path.abspath(os.getcwd())

# Check if we're already in the target folder
if os.path.basename(current_dir) != TARGET_FOLDER:
    target_path = os.path.abspath(os.path.join('code', TARGET_FOLDER))
    if os.path.exists(target_path):
        os.chdir(target_path)
        print(f"Changed working directory to: {target_path}")
    else:
        raise FileNotFoundError(f"Target directory does not exist: {target_path}")
else:
    print("Working directory is already set to the target folder.")

# Set script_dir based on __file__ if available, else current working directory
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print("Script directory:", script_dir)

# Define paths (adjust if needed)
data_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed'))
train_path = os.path.join(data_dir, 'training_data.csv')
test_path = os.path.join(data_dir, 'test_data.csv')

# Directory for saving models
trained_models_path = os.path.abspath(os.path.join(script_dir, '..','..','results','trained_models'))
os.makedirs(trained_models_path, exist_ok=True)

# Directory for learning-related outputs (e.g., Docker images)
learning_base_path = os.path.abspath(os.path.join(script_dir, '..','..','docker','images','learningBase'))
os.makedirs(learning_base_path, exist_ok=True)

# Add code directory to system path
code_path = os.path.abspath(os.path.join(script_dir, '..'))
if code_path not in sys.path:
    sys.path.insert(0, code_path)

# Import custom plot saver function
from utils.plot_saver import save_plot

# Import custom linear regression diagnostic
from utils.LinearRegDiagnostic import LinearRegDiagnostic

# -----------------------------------------------------------------------------
# 1. Load Preprocessed Data
# -----------------------------------------------------------------------------
print("Training data path:", train_path)
print("Test data path:    ", test_path)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)

# Separate features (X) and target (y)
target_col = 'Exam_Score'
X_train = train_df.drop(columns=[target_col]).values
y_train = train_df[target_col].values

X_test = test_df.drop(columns=[target_col]).values
y_test = test_df[target_col].values

# -----------------------------------------------------------------------------
# 2. Ordinary Least Squares (OLS) with Statsmodels
# -----------------------------------------------------------------------------

X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_ols).fit()
print(ols_model.summary())

y_pred_OLS = ols_model.predict(X_test_ols)

mse_OLS = mean_squared_error(y_test, y_pred_OLS)
mae_OLS = mean_absolute_error(y_test, y_pred_OLS)
r2_OLS = r2_score(y_test, y_pred_OLS)

print("\n=== OLS Performance on Test Set ===")
print("MSE :", mse_OLS)
print("MAE :", mae_OLS)
print("R^2 :", r2_OLS)

# 2.1 Scatter Plot of OLS Predictions vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_OLS, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('OLS Predictions vs. Actual')

fig_scatter_ols = plt.gcf()
save_plot(fig_scatter_ols, 'ols_predictions_scatter.png')
fig_scatter_ols.savefig(os.path.join(learning_base_path, 'ols_predictions_scatter.png'), bbox_inches='tight')
plt.show()

# 2.2 Linear Regression Diagnostics Using `LinearRegDiagnostic`
diag = LinearRegDiagnostic(ols_model)
vif_table, fig_diagnostics, ax = diag()

diag_pdf_path = os.path.join(learning_base_path, 'OLS_DiagnosticPlots.pdf')
fig_diagnostics.savefig(diag_pdf_path, format='pdf', bbox_inches='tight')
print(f"Diagnostic plots saved to PDF: {diag_pdf_path}")

plt.show()

# 2.3 Save the OLS Model
ols_model_path = os.path.join(trained_models_path, 'currentOlsSolution.pkl')
with open(ols_model_path, 'wb') as f:
    pickle.dump(ols_model, f)

print(f"OLS model saved at: {ols_model_path}")

# Save the OLS summary to a text file
ols_summary_path = os.path.join(trained_models_path, 'ols_model_summary.txt')
with open(ols_summary_path, 'w') as f:
    f.write(str(ols_model.summary()))

print(f"OLS summary saved at: {ols_summary_path}")