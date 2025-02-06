# Student Performance Analysis with AI and OLS Models

## Project Overview

This repository is part of the course _Advanced AI-based Application Systems_ course taught by M. Grum at the University of Potsdam, Germany.<br>
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
            ├── data_pipeline/ # Master notebook and singular scripts for data scraping, preprocessing and splitting
            ├── models/ # AI and OLS training and testing scripts
            ├── utils/ # Additional useful utility functions

    ├── data/
            ├── raw/ # Raw and scraoed datasets
            ├── processed/ # Cleaned and prepared datasets

    ├── docker/
            ├── images/ # Docker image folders for deployment
                      ├── activationBase/
                      ├── codeBase/
                      ├── knowledgeBase/
                      ├── learningBase/

    ├── results/ # Generated plots, trained models, and performance metrics
            ├── plots/ # Generated plots
            ├── reports/ # Reports with model performance metrics
            ├── trained_models/ # Trained OLS and AI models

    ├── scenarios/ # Two docker compose files dedicated for testing each model with activation data
            ├── ANN/
            ├── OLS/
```

## Dataset

We use the _Student Performance Factors_ dataset sourced from Kaggle: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data).<br>
This dataset contains synthetically generated student records with features like study time, family background, health, and other various exam performance relevant characteristics.

## How to re-train the models locally

1. Clone this repository: `git clone https://github.com/il1a/student-performance-predictor`
2. Navigate to the project directory: `cd student-performance-predictor`
3. Install dependencies either using pip: `pip install -r requirements.txt` or using conda: `conda env create -f conda.yml`
4. Navigate to the data_pipeline folder and either run all the cells in the notebook or run .py scripts instead in the following order: data_scraping.py -> data_preprocessing.py -> data_splitting.py
5. Finally navigate to the models folder and either run all the cells in the notebook or run separate model scripts ANN.py and OLS.py (order doesn't matter)

## How to test our trained models using Docker

You can test our models using our dedicated docker workflow that includes custom docker images and activation scripts.  
Testing is performed by loading each of the trained models and giving it a single activation data entry (random test data sample) and examining the final predictions of both models.
In order to do that please refer to our README file located in the designated docker folder : [Docker README](https://github.com/il1a/student-performance-predictor/tree/main/docker/README.md)

## Licensing

This project is licensed under the AGPL-3.0 License.

## Contributors

- Elisa Haxhillazi
- Ilia Sokolovskiy
