## Student Performance Analysis with AI and OLS Models

### GitHub
Project GitHub repository : https://github.com/il1a/student-performance-predictor

### Project Overview
This docker image is part of the course _Advanced AI-based Application Systems_ course taught by M. Grum at the University of Potsdam, Germany.  
The project focuses on analyzing factors affecting student performance using both AI-based model (via TensorFlow) and a traditional OLS (Ordinary Least Squares) regression model.  
The project adheres to the guidelines set by the course to ensure reproducibility and transparency.

### Docker images overview

***
###### learning-base-student-performance-predictor
Contains two .csv files with training and testing data.  
You can pull that image with the following command : `docker pull il1aa/learning-base-student-performance-predictor:v1.0`

>The *learning base* docker image is not required for testing! It's simply included for project transparency reasons.
***

###### activation-base-student-performance-predictor
Contains one .csv file with the activation data (one random data sample).  
You can pull that image with the following command : `docker pull il1aa/activation-base-student-performance-predictor:v1.0`
***

###### knowledge-base-student-performance-predictor
Contains two trained model files (.keras and .pkl) and their performance metrics.  
You can pull that image with the following command : `docker pull il1aa/knowledge-base-student-performance-predictor:v1.0`
***

###### code-base-student-performance-predictor
Contains two python scripts for testing both trained models with activation data.  
You can pull that image with the following command : `docker pull il1aa/code-base-student-performance-predictor:v1.0-amd64`

>**DISCLAIMER!** Due to TensorFlow base image limitations this image only supports amd64 architecture! The arm64 is **NOT** supported!
***
### How to test the models
You can test both ANN and OLS models by doing the following :  
1. Pull **all** the images using the `docker pull` commands described above <br><br>
2. Create an external local volume _ai_system_ for persisting entire container data by running the following command on your machine : `docker volume create --name ai_system` <br><br>
3. To test the ANN model please save the following docker compose .yml file locally : [ANN](https://github.com/il1a/student-performance-predictor/blob/main/scenarios/ANN/docker-compose-ann.yml)   
Open terminal, navigate to directory where the file is saved and run it with : `docker compose -f docker-compose-ann.yml up` <br><br>
4. To test the OLS model please save the following docker compose .yml file locally : [OLS](https://github.com/il1a/student-performance-predictor/blob/main/scenarios/OLS/docker-compose-ols.yml)  
Open another terminal,  navigate to directory where the second file is saved and run it with : `docker compose -f docker-compose-ols.yml up` <br>

After running both docker compose commands, examine the CLI outputs for both models. <br>
You should normally see a result of a single prediction that has been made by each model using the single activation data entry, as well as the actual value of the target (_Exam_Score_).

### Examining the contents of the ai_system volume
After comparing the results you can create and run an interactive ubuntu container with the persisted data from the docker volume _ai_system_ mounted onto it. <br>
This can be done with the following command : `docker run --rm -it -v ai_system:/data ubuntu bash` <br>
Once in the container, navigate to the _/data_ directory and examine the folders and files located inside. <br>
When you exit bash or close the terminal the container is automatically destroyed.

### Data
We use the _Student Performance Factors_ dataset sourced from Kaggle: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data).<br>
This dataset contains synthetically generated student records with features like study time, family background, health, and other various exam performance relevant characteristics.

### ANN and OLS model descriptions

#### ANN Model Description
The ANN was implemented using Keras with TensorFlow as the backend.  
It is a sequential feedforward regression network that accepts five feature inputs, processes them through two hidden layers (each with 64 ReLU-activated neurons) and outputs a single continuous value.

#### OLS Model Description
The model is an ordinary least squares (OLS) linear regression implemented using the statsmodels library.  
It fits a linear relationship between the dependent variable and a set of predictors by minimizing the sum of squared residuals. <br>

>Both models had been already trained beforehand and saved for the testing inference in the knowledge base folder. Their final performance metrics can be found there too. <br>

### Licensing
This project is licensed under the AGPL-3.0 License.

### Contributors
- Elisa Haxhillazi
- Ilia Sokolovskiy