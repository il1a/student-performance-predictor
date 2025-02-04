# Student Performance Analysis with AI and OLS Models

## GitHub
Project GitHub repository : https://github.com/il1a/student-performance-predictor

## Project Overview
This docker image is part of the course _Advanced AI-based Application Systems_ course taught by M. Grum at the University of Potsdam, Germany.<br>
The project focuses on analyzing factors affecting student performance using both AI-based model (via TensorFlow) and a traditional OLS (Ordinary Least Squares) regression model.<br>
The project adheres to the guidelines set by the course to ensure reproducibility and transparency.

## Docker images overview

---
###### ilia-elisa/learning_base_student-performance-predictor
Contains two .csv files with training and testing data. <br>
You can pull that image with the following command : `docker pull ilia-elisa/learning_base_student-performance-predictor:latest`
---
###### ilia-elisa/activation_base_student-performance-predictor
Contains one .csv file with the activation data (one random data sample). <br>
You can pull that image with the following command : `docker pull ilia-elisa/activation_base_student-performance-predictor:latest`
---
###### ilia-elisa/knowledge_base_student-performance-predictor
Contains two trained model files (.keras and .pkl) and their performance metrics. <br>
You can pull that image with the following command : `docker pull ilia-elisa/knowledge_base_student-performance-predictor:latest`
---
###### ilia-elisa/code_base_student-performance-predictor 
Contains two python scripts for testing both trained models with activation data. <br>
You can pull that image with the following command : `docker pull ilia-elisa/code_base_student-performance-predictor:latest`
---
## How to test the models
You can test both ANN and OLS models by doing the following : <br>
1. Pull **all** the images using the `docker pull` commands described above <br><br>
2. Create an external local volume _ai_system_ by running the following command on your machine : `docker volume create --name ai_system` <br><br>
3. To test the ANN model please save the following docker compose .yml file locally : ... <br> 
And then run it with : `docker compose -f docker-compose-ann.yml up` <br><br>
4. To test the OLS model please save the following docker compose .yml file locally : ... <br>
And then run it with : `docker compose -f docker-compose-ols.yml up` <br><br>

>DISCLAIMER! Please be informed that not all the docker images are required for testing, namely the *learning base* image is not needed as it's simply included for project transparency reasons. If you wish to re-train the models from scratch please refer to our project GitHub repository mentioned above.

## Data
We use the _Student Performance Factors_ dataset sourced from Kaggle: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data).<br>
This dataset contains synthetically generated student records with features like study time, family background, health, and other various exam performance relevant characteristics.

## ANN and OLS models
Both models had been already trained beforehand and saved for the testing inference in the knowledge base folder. 
They can be tested by running each of their dedicated python scripts and reviewing the printed metrics.

## Licensing
This project is licensed under the AGPL-3.0 License.

## Contributors
- Elisa Haxhillazi
- Ilia Sokolovskiy
