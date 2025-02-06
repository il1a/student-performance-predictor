# Student Performance Analysis with AI and OLS Models

## Project Overview

This repository is part of the course _Advanced AI-based Application Systems_ course taught by M. Grum at the University of Potsdam, Germany.<br>
The project focuses on analyzing factors affecting student performance using both AI-based models (via TensorFlow) and a traditional OLS (Ordinary Least Squares) regression model.<br>
The project adheres to the guidelines set by the course to ensure reproducibility and transparency.

### Code overview

#### Data Scraping and Preparation

This feature in our project can be found in the code/data_pipeline folder. The results of the prepared data are stored in the data folder.

The Jupyter notebook is designed to provide a more streamlined and user-friendly interface, consolidating the logic into one place for ease of use. The actual implementation is split into 3 separate Python script files in the code/data_pipeline folder, which allows the code to be modular and more easily integrated into a Docker environment for better scalability and deployment.

##### 1. data_scraping.py

Scrapes raw student performance data from a remote CSV file (hosted on GitHub).

##### 2. data_preprocessing.py

Cleans and preprocesses the raw dataset to prepare it for training.

##### 3. data_splitting.py

Splits the cleaned dataset into training, test, and activation data.

---

#### Models

This feature in our project can be found in the code/models folder. The results produced by each model is saved in the results/plots, results/reports, and results/trained folders.

The Jupyter notebook is designed to provide a more streamlined and user-friendly interface, consolidating the logic into one place for ease of use for both models running locally. The actual implementation is split into 2 separate Python script files: ANN_activation.py and OLS_activation.py, which allows the code to be modular and more easily integrated into a Docker environment for better scalability and deployment using Docker paths.

##### 1. ANN_activation.py

Trains an Artificial Neural Network (ANN) to predict a target variable.

##### 2. OLS_activation.py

Trains an Ordinary Least Squares (OLS) regression model to predict exam scores

## Licensing

This project is licensed under the AGPL-3.0 License.

## Contributors

- Elisa Haxhillazi
- Ilia Sokolovskiy
