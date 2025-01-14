# Dataset: Student Performance Factors

## Dataset Overview
The dataset explores factors that affect student performance.<br>
It contains synthetically generated student records, including features such as:

- **Hours_Studied**: Number of hours spent studying per week. 
- **Attendance**: Percentage of classes attended. 
- **Parental_Involvement**: Level of parental involvement in the student's education (Low, Medium, High). 
- **Access_to_Resources**: Availability of educational resources (Low, Medium, High). 
- **Extracurricular_Activities**: Participation in extracurricular activities (Yes, No). 
- **Sleep_Hours**: Average number of hours of sleep per night. 
- **Previous_Scores**: Scores from previous exams. 
- **Motivation_Level**: Student's level of motivation (Low, Medium, High). 
- **Internet_Access**: Availability of internet access (Yes, No). 
- **Tutoring_Sessions**: Number of tutoring sessions attended per month. 
- **Family_Income**: Family income level (Low, Medium, High). 
- **Teacher_Quality**: Quality of the teachers (Low, Medium, High). 
- **School_Type**: Type of school attended (Public, Private). 
- **Peer_Influence**: Influence of peers on academic performance (Positive, Neutral, Negative). 
- **Physical_Activity**: Average number of hours of physical activity per week. 
- **Learning_Disabilities**: Presence of learning disabilities (Yes, No). 
- **Parental_Education_Level**: Highest education level of parents (High School, College, Postgraduate). 
- **Distance_from_Home**: Distance from home to school (Near, Moderate, Far). 
- **Gender**: Gender of the student (Male, Female). 
- **Exam_Score**: Final exam score.

## Source
This dataset is sourced from Kaggle: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data).

## Usage in This Project
- **Data Cleaning**: Outliers are removed, and missing data is handled.
- **Feature Engineering**: Relevant features are normalized for model training.
- **Data Splits**:
    - `joint_data_collection.csv`: The cleaned dataset.
    - `training_data.csv`: 80% of the dataset for training.
    - `test_data.csv`: 20% of the dataset for testing.
    - `activation_data.csv`: One entry used for activation testing.

## Licensing
The dataset is used in compliance with its original licensing terms and the AGPL-3.0 license for this project.

## How to Access
The raw dataset is located in the `/data/raw/` directory, while cleaned and processed versions are stored in `/data/processed/`.

## Disclaimer
The dataset is for educational and research purposes only. Any misuse of the data is the responsibility of the user.
