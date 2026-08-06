# 🎓 Student Placement Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Overview

This project predicts whether a student will be **Placed** or **Not Placed** using Machine Learning. It analyzes academic performance and skill-based attributes such as **CGPA, IQ, Communication Skills, Aptitude Score, Technical Skills, Internship Experience, Projects, Backlogs, and Attendance** to determine a student's placement status.

The project follows a complete Machine Learning workflow, including **Data Preprocessing, Exploratory Data Analysis (EDA), Feature Scaling, Model Training, Model Evaluation, and Performance Comparison**.

---

## 🚀 Features

- 📂 Data Loading & Exploration
- 🧹 Data Cleaning & Preprocessing
- 📊 Exploratory Data Analysis (EDA)
- ⚙️ Feature Scaling using StandardScaler
- 🤖 Logistic Regression
- 🤖 Support Vector Machine (SVM)
- 🤖 K-Nearest Neighbors (KNN)
- 🌳 Decision Tree
- 📈 Model Performance Comparison
- 📉 Confusion Matrix
- 📋 Classification Report
- 🎯 Accuracy, Precision, Recall & F1-Score Evaluation

---

## 📁 Project Structure

```text
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
```

---

## 🗂️ Dataset

The dataset contains **5,000 student records** with the following features:

| Feature | Description |
|---------|-------------|
| Student_ID | Unique Student ID |
| CGPA | Academic Performance |
| IQ | Intelligence Quotient |
| Communication_Skills | Communication Skill Rating |
| Aptitude_Score | Aptitude Test Score |
| Technical_Skills | Technical Skill Rating |
| Internship | Internship Experience (Yes/No) |
| Projects | Number of Projects |
| Backlogs | Number of Active Backlogs |
| Attendance | Attendance Percentage |
| Placement | Placed / Not Placed |

---

## 🤖 Machine Learning Algorithms

This project compares the performance of the following supervised learning algorithms:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-placement-prediction.git
```

Navigate to the project folder:

```bash
cd student-placement-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## 📈 Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
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
```

---

## 🎯 Project Objective

The objective of this project is to predict student placement using supervised machine learning algorithms and compare their performance. It demonstrates the complete machine learning pipeline, from preprocessing and visualization to model training and evaluation, providing insights into how academic and technical factors influence placement outcomes.

---

## 📌 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Importance Analysis
- Model Deployment using Flask or Streamlit
- Interactive Dashboard

---

## 👨‍💻 Author

**Ayush Joshi**

BCA Student | Machine Learning Enthusiast | Aspiring Software Developer

---

## ⭐ Support

If you found this project helpful, consider giving this repository a ⭐ on GitHub.
