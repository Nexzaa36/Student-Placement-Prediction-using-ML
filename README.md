Student Placement Prediction Using Machine Learning



📌 Overview

This project predicts whether a student will be Placed or Not Placed using Machine Learning. It analyzes various academic and skill-based attributes such as CGPA, IQ, Communication Skills, Aptitude Score, Technical Skills, Internship Experience, Projects, Backlogs, and Attendance to determine a student's placement status.

The project follows a complete Machine Learning workflow, including Data Preprocessing, Exploratory Data Analysis (EDA), Feature Scaling, Model Training, Model Evaluation, and Performance Comparison. Four supervised learning algorithms are implemented and compared to identify the best-performing model.

🚀 Features
📂 Data Loading and Exploration
🧹 Data Cleaning and Preprocessing
📊 Exploratory Data Analysis (EDA)
⚙️ Feature Scaling using StandardScaler
🤖 Logistic Regression
🤖 Support Vector Machine (SVM)
🤖 K-Nearest Neighbors (KNN)
🤖 Decision Tree
📈 Model Performance Comparison
📉 Confusion Matrix
📋 Classification Report
🎯 Accuracy, Precision, Recall & F1-Score Evaluation


📁 Project Structure
Student-Placement-Prediction/
│
├── student_placement_dataset_5000.csv
├── main.py
├── graphs/
│   ├── placement_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── ...
├── README.md
└── requirements.txt
🗂️ Dataset

The dataset contains 5,000 student records with the following features:

Feature	Description
Student_ID	Unique Student ID
CGPA	Academic Performance
IQ	Intelligence Quotient
Communication_Skills	Communication Skill Rating
Aptitude_Score	Aptitude Test Score
Technical_Skills	Technical Skill Rating
Internship	Internship Experience (Yes/No)
Projects	Number of Academic Projects
Backlogs	Number of Active Backlogs
Attendance	Attendance Percentage
Placement	Target Variable (Placed / Not Placed)

🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
🤖 Machine Learning Models

The following supervised learning algorithms are used:

Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Tree
📊 Model Evaluation

Each model is evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report

A comparison of all four models is performed to determine the best-performing algorithm.

📈 Workflow
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Train-Test Split
      │
      ▼
Train ML Models
      │
      ▼
Model Evaluation
      │
      ▼
Performance Comparison
🎯 Objective

The objective of this project is to predict student placement using supervised machine learning algorithms and compare their performance. It demonstrates the complete ML pipeline, from preprocessing and visualization to model training and evaluation, helping understand how academic and technical factors influence placement outcomes.

📷 Sample Output

The project generates:

Dataset Summary
Exploratory Data Analysis Graphs
Correlation Heatmap
Confusion Matrices
Classification Reports
Accuracy Comparison of Models
👨‍💻 Author

Ayush Joshi

BCA Student | Aspiring Software Developer | Machine Learning Enthusiast

⭐ If you found this project useful, consider giving it a star!
