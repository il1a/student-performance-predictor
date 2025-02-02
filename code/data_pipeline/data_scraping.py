#!/usr/bin/env python3
"""
data_scraping.py

This script scrapes the raw data from a remote CSV file, displays some information,
and saves the scraped data to a file so that it can be used in the preprocessing step.
"""

import os
import sys
import requests
import pandas as pd
from io import StringIO

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

# --- 1.1 Scrape the data from the .csv file saved on projects github repo ---
# URL of the raw CSV file
url = 'https://raw.githubusercontent.com/il1a/student-performance-predictor/refs/heads/main/data/raw/original_data.csv'

# Make an HTTP GET request to fetch the CSV content
response = requests.get(url)

if response.status_code == 200:
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)
    print("Data scraped successfully! Here's the DataFrame information:")
    df.info()
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    sys.exit(1)

# --- 1.2 Print some random data samples ---
print("\n10 random data samples:")
print(df.sample(10))

# --- 1.3 Save the scraped raw data to a CSV file for further processing ---
raw_data_dir = os.path.join(script_dir, '..', '..', 'data', 'raw')
os.makedirs(raw_data_dir, exist_ok=True)
scraped_data_path = os.path.join(raw_data_dir, 'scraped_data.csv')
df.to_csv(scraped_data_path, index=False)
print(f"\nRaw data saved to: {scraped_data_path}")
