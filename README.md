# Student Performance Analysis with AI and OLS Models

## Project Overview
This repository is part of the course *Advanced AI-based Application Systems* course taught by M. Grum at the University of Potsdam, Germany.<br>
The project focuses on analyzing factors affecting student performance using both AI-based models (via TensorFlow) and a traditional OLS (Ordinary Least Squares) regression model.<br>
The project adheres to the guidelines set by the course to ensure reproducibility and transparency.

## Key Features
- **Data Scraping and Preparation**: Includes scripts for data cleaning, normalization, and splitting into training, testing, and activation sets.
- **AI Modeling**: Implements a neural network for predicting student performance.
- **OLS Modeling**: Uses a regression model to perform the same task as the AI model for comparison.
- **Dockerization**: Provides Docker images for data, models, and activation components for seamless deployment.
- **Visualization**: Generates diagnostic plots, scatter plots, and performance metrics for insights.
- **Reproducibility**: Designed to ensure the entire workflow can be replicated using provided scripts and data.

## Folder Structure

```
AIBAS-student-performance-predictor/ 

    ├── code/
            ├── data_processing/ # Scripts and notebooks for Exploratory Data Analysis (EDA), data scraping, and cleaning
            ├── models/ # AI and OLS training and testing scripts │ 
            ├── docker/ # Docker files for building and running containers │ 
            ├── visualizations/ # Scripts for creating plots and metrics 
            
    ├── data/
            ├── raw/ # Raw dataset
            ├── processed/ # Scraped, cleaned, and prepared datasets 
            
    ├── docs/ # Reports and other relevant documentation 
    
    ├── images/ # Docker images for deployment 
    
    ├── results/ # Generated plots, trained models, and performance metrics
```

## Dataset
We use the *Student Performance Factors* dataset sourced from Kaggle: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data).<br>
This dataset contains synthetically generated student records with features like study time, family background, health, and other various exam performance relevant characteristics.

## How to Use
1. Clone this repository: `git clone https://github.com/il1a/student-performance-predictor`
2. Navigate to the project directory: `cd student-performance-predictor`
3. Install dependencies: `pip install -r requirements.txt`<br>
   The rest of the steps is **coming soon** ...

## Licensing
This project is licensed under the AGPL-3.0 License.

## Contributors
- Elisa Haxhillazi
- Ilia Sokolovskiy