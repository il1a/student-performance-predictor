#!/usr/bin/env python3
"""
ANN Pipeline Script

This script:
1. Loads preprocessed training and test data
2. Defines, compiles, and trains an ANN using TensorFlow/Keras
3. Evaluates the model using MSE, MAE, and R²
4. Visualizes training history, predictions vs actual, and residuals
5. Saves the model and plots
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras for ANN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sklearn metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# (Optional) for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

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

# Add code directory to system path (so we can import from utils if needed)
code_path = os.path.abspath(os.path.join(script_dir, '..'))
if code_path not in sys.path:
    sys.path.insert(0, code_path)

# Import custom plot saver function (if you need it)
from utils.plot_saver import save_plot

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
# 2. Artificial Neural Network (ANN)
# -----------------------------------------------------------------------------

# 2.1 Define and Compile the Model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # single output for regression
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# 2.2 Train the Model
epochs = 50
batch_size = 8

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# 2.3 Visualize Training History
# Plot training & validation loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('ANN Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

fig_loss = plt.gcf()  # get current figure
save_plot(fig_loss, 'ann_loss_curve.png')
fig_loss.savefig(os.path.join(learning_base_path, 'ann_loss_curve.png'), bbox_inches='tight')
plt.show()

# Plot training & validation MAE
plt.figure(figsize=(8,5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Test MAE')
plt.title('ANN Training & Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

fig_mae = plt.gcf()
save_plot(fig_mae, 'ann_mae_curve.png')
fig_mae.savefig(os.path.join(learning_base_path, 'ann_mae_curve.png'), bbox_inches='tight')
plt.show()

# 2.4 Evaluate the ANN on the Test Set
y_pred_ANN = model.predict(X_test).flatten()

mse_ANN = mean_squared_error(y_test, y_pred_ANN)
mae_ANN = mean_absolute_error(y_test, y_pred_ANN)
r2_ANN = r2_score(y_test, y_pred_ANN)

print("=== ANN Performance on Test Set ===")
print("MSE :", mse_ANN)
print("MAE :", mae_ANN)
print("R^2 :", r2_ANN)

# 2.5 Scatter Plot of Predictions vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_ANN, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('ANN Predictions vs. Actual')

fig_scatter_ann = plt.gcf()
save_plot(fig_scatter_ann, 'ann_predictions_scatter.png')
fig_scatter_ann.savefig(os.path.join(learning_base_path, 'ann_predictions_scatter.png'), bbox_inches='tight')
plt.show()

# 2.6 Residual Distribution Plot
residuals_ANN = y_test - y_pred_ANN

plt.figure(figsize=(6,4))
sns.histplot(residuals_ANN, kde=True)
plt.title('ANN Residual Distribution')
plt.xlabel('Residual (Actual - Predicted)')

fig_resid_ann = plt.gcf()
save_plot(fig_resid_ann, 'ann_residual_distribution.png')
fig_resid_ann.savefig(os.path.join(learning_base_path, 'ann_residual_distribution.png'), bbox_inches='tight')
plt.show()

# 2.7 Save the Trained ANN Model
ann_model_path = os.path.join(trained_models_path, 'currentAiSolution.keras')
model.save(ann_model_path)
print(f"ANN model saved at: {ann_model_path}")

# Save training history as CSV
history_df = pd.DataFrame(history.history)
history_csv_path = os.path.join(trained_models_path, 'ann_training_metrics.csv')
history_df.to_csv(history_csv_path, index=False)
print(f"ANN training metrics saved to {history_csv_path}")