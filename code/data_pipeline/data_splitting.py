"""
data_splitting.py

This script loads the final clean dataset created by data_preprocessing.py,
splits it into training and test sets (80% train, 20% test),
and saves:
- training_data.csv
- test_data.csv
- activation_data.csv (a single random sample from the test set)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Ensure the correct working directory ---
TARGET_FOLDER = 'data_pipeline'
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

# Load the final clean dataset
processed_data_dir = os.path.join(script_dir, '..', '..', 'data', 'processed')
joint_dataset_path = os.path.join(processed_data_dir, 'joint_data_collection.csv')
final_clean_df = pd.read_csv(joint_dataset_path)
print("Final clean dataset loaded successfully.")

target_col = 'Exam_Score'
X = final_clean_df.drop(columns=[target_col])
y = final_clean_df[target_col]

# --- 3.1 Split the data (80% training, 20% test), save as CSV under data/processed ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = pd.DataFrame(X_train, columns=X.columns)
train_df[target_col] = y_train.values

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df[target_col] = y_test.values

train_path = os.path.join(processed_data_dir, 'training_data.csv')
test_path = os.path.join(processed_data_dir, 'test_data.csv')

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Training data saved to {train_path}")
print(f"Test data saved to {test_path}")

# --- 3.2 Create activation data (single data entry from test set), save as CSV under data/processed ---
activation_df = test_df.sample(n=1, random_state=42)
activation_path = os.path.join(processed_data_dir, 'activation_data.csv')
activation_df.to_csv(activation_path, index=False)
print(f"Activation data (1 row) saved to: {activation_path}")
