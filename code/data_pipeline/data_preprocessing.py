"""
data_preprocessing.py

This script loads the raw data (saved by data_scraping.py), performs cleaning and preprocessing:
- Missing value imputation (numeric & categorical)
- Conversion of categorical values to numeric through mappings
- Feature selection based on Pearson correlation with the target "Exam_Score"
- Exploratory visualization (correlation heatmap, box plots, histograms)
- Outlier removal using the IQR method
- Feature normalization

The final clean dataset is saved as joint_data_collection.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

# Add the code directory to the Python path
code_path = os.path.abspath(os.path.join('..'))
if code_path not in sys.path:
    sys.path.append(code_path)

# Import the custom save_plot function from utils
from utils.plot_saver import save_plot

# Load raw data produced by data_scraping.py
scraped_data_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'scraped_data.csv')
df = pd.read_csv(scraped_data_path)
print("Scraped data loaded successfully.")

# --- 2.1 Check the number of missing values ---
total_na = df.isna().sum().sum()
print("\nInitial total number of NA values in the dataset:", total_na)

# --- 2.2 Impute missing values ---
#    - Numeric columns: fill with mean
#    - Categorical columns: fill with mode

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute numeric columns with mean
imputer_num = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

# Impute categorical columns with mode
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# Verify that no missing values remain
total_na_after_imputation = df.isna().sum().sum()
print("Total number of NA values after imputation:", total_na_after_imputation)

# --- 2.3 Convert categorical columns to numeric ---
# Create dictionaries with mappings
yes_no_mapping = {
    'No': 0,
    'Yes': 1
}

gender_mapping = {
    'Female': 0,
    'Male': 1
}

ordinal_mappings = {
    'Low': 1,
    'Medium': 2,
    'High': 3
}

pos_neut_neg_mapping = {
    'Negative': 1,
    'Neutral': 2,
    'Positive': 3
}

dist_mapping = {
    'Near': 1,
    'Moderate': 2,
    'Far': 3
}

edu_mapping = {
    'High School': 1,
    'College': 2,
    'Postgraduate': 3
}

# Apply mappings to categorical columns
for col in ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities']:
    if col in df.columns:
        df[col] = df[col].map(yes_no_mapping)

for col in ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality']:
    if col in df.columns:
        df[col] = df[col].map(ordinal_mappings)

if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map(gender_mapping)

if 'Peer_Influence' in df.columns:
    df['Peer_Influence'] = df['Peer_Influence'].map(pos_neut_neg_mapping)

if 'Distance_from_Home' in df.columns:
    df['Distance_from_Home'] = df['Distance_from_Home'].map(dist_mapping)

if 'Parental_Education_Level' in df.columns:
    df['Parental_Education_Level'] = df['Parental_Education_Level'].map(edu_mapping)

# --- 2.4 Choose only relevant variables (X features) based on Pearson correlation with target (Exam_Score) ---

target_col = 'Exam_Score'

# Ensure all columns are numeric now for the correlation matrix
df_for_corr = df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix = df_for_corr.corr()

# Look at correlation with target
corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
print("\nCorrelation with target (absolute values):\n", corr_with_target)

# Keep top 5 correlated features + the target
top_features = corr_with_target.index[0:6]
print("\nFinal top 5 features and target:\n", top_features)

# Only leave the selected top 5 features in the dataset
df_top = df_for_corr[top_features]

# Separate X and y
X = df_top.drop(columns=[target_col])
y = df_top[target_col]

# --- 2.5 Visualise the correlation matrix using a heatmap ---
# Compute the correlation matrix with top 5 features
corr_matrix_top = df_top.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_top, annot=True, cmap='magma', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")

# Save the current figure using the custom function
save_plot(plt.gcf(), 'correlation_matrix.png', os.path.join(script_dir, '..', '..', 'results', 'plots'))

# Display the correlation matrix
plt.show()

# --- 2.6 Outlier detection and removal using IQR on the selected columns + target ---
# Define additional function for IQR outlier removal
def remove_outliers_iqr(dataframe, columns, k=1.5):
    """
    Remove outliers from the specified columns using the IQR method.
    k=1.5 is the default multiplier.
    """
    df_out = dataframe.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        # Filter out rows outside the IQR bounds
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
    return df_out

# Combine X and y temporarily for outlier removal
df_xy = X.copy()
df_xy[target_col] = y

# Identify numeric columns for visualization
numeric_cols = df_xy.select_dtypes(include=[np.number]).columns

# Box plots before outlier removal
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_xy[numeric_cols], orient='v')
plt.title("Box plots Before Removing Outliers")
plt.tight_layout()
save_plot(plt.gcf(), 'boxplots_outliers.png', os.path.join(script_dir, '..', '..', 'results', 'plots'))
plt.show()

# Perform outlier removal using IQR function defined earlier
df_xy_clean = remove_outliers_iqr(df_xy, numeric_cols, k=1.5)

# Box plots after outlier removal
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_xy_clean[numeric_cols], orient='v')
plt.title("Box plots After Removing Outliers")
plt.tight_layout()
save_plot(plt.gcf(), 'boxplots_no_outliers.png', os.path.join(script_dir, '..', '..', 'results', 'plots'))
plt.show()

# Update X and y after outlier removal
X = df_xy_clean.drop(columns=[target_col])
y = df_xy_clean[target_col]

print(f"\nShape before outlier removal: {df_xy.shape}")
print(f"Shape after outlier removal: {df_xy_clean.shape}")

# --- 2.7 Check data distributions visually using histograms before normalization ---
plt.figure(figsize=(12, 12))
df_xy_clean.hist(bins=30, figsize=(12, 12))
plt.tight_layout()
save_plot(plt.gcf(), 'histograms_before_norm.png', os.path.join(script_dir, '..', '..', 'results', 'plots'))
plt.show()

# --- 2.8 Print summary statistics before normalization ---
print("\nSummary statistics before normalization:")
print(df_xy_clean.describe())

# --- 2.9 Perform normalization using the StandardScaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier handling
X = pd.DataFrame(X_scaled, columns=X.columns)

# Combine back X and y into one final dataset
final_clean_df = X.copy()
final_clean_df[target_col] = y.values

# --- 2.10 Check data distributions visually using histograms after normalization ---
plt.figure(figsize=(12, 12))
final_clean_df.hist(bins=30, figsize=(12, 12))
plt.tight_layout()
save_plot(plt.gcf(), 'histograms_after_norm.png', os.path.join(script_dir, '..', '..', 'results', 'plots'))
plt.show()

# --- 2.11 Print summary statistics after normalization ---
print("\nSummary statistics after normalization:")
print(final_clean_df.describe())

# --- 2.12 Save the final clean dataset under data/processed ---
processed_data_dir = os.path.join(script_dir, '..', '..', 'data', 'processed')
os.makedirs(processed_data_dir, exist_ok=True)
joint_dataset_path = os.path.join(processed_data_dir, 'joint_data_collection.csv')
final_clean_df.to_csv(joint_dataset_path, index=False)
print(f"\nFinal clean dataset saved to: {joint_dataset_path}")
