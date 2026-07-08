<img width="1536" height="1024" alt="a246ee74-c4bb-435d-92d1-b240e9164720" src="https://github.com/user-attachments/assets/14f93944-fcd7-4d95-9e6a-3c242a04822f" />




# 🌲 ML Random Forest

> **An end-to-end implementation of Random Forest Classification using the Crop Recommendation dataset, covering exploratory data analysis, feature engineering, ensemble learning, feature importance, and model evaluation.**

---

## 📖 Overview

Random Forest is an ensemble machine learning algorithm that combines multiple Decision Trees to improve predictive performance, reduce overfitting, and increase model robustness.

This project demonstrates the complete machine learning workflow using the **Crop Recommendation Dataset**, including data exploration, preprocessing, model training, prediction, evaluation, feature importance analysis, and model serialization.

This repository is part of my **ML Fundamentals** series, where each project focuses on understanding a core machine learning algorithm through theory, implementation, and practical experimentation.

---

## 🚀 Project Status

> 🚧 Currently under development.

The repository will include:

- Dataset exploration and preprocessing
- Exploratory Data Analysis (EDA)
- Random Forest model training
- Feature importance analysis
- Prediction on unseen data
- Confusion Matrix
- Accuracy, Precision, Recall and F1-Score
- ROC Curve and AUC
- Model serialization using Joblib

---

## 📊 Dataset

This project uses the **Crop Recommendation Dataset**.

The dataset contains soil nutrient values and environmental conditions used to predict the most suitable crop for cultivation.

### Features

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

### Target

The target variable is the **recommended crop**, making this a **multi-class classification** problem.

---

## 📂 Repository Structure

```text
ml-random-forest
│
├── data/
├── models/
├── outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
## 📈 Results

The Random Forest Classifier achieved excellent performance on the Crop Recommendation dataset, accurately identifying the most suitable crop based on soil nutrient composition and environmental conditions. The ensemble learning approach demonstrated strong generalization across all crop categories.

### Model Performance

| Metric | Score |
|--------|------:|
| Accuracy | **99.32%** |
| Weighted Precision | **99.35%** |
| Weighted Recall | **99.32%** |
| Weighted F1-Score | **99.32%** |
| Out-of-Bag (OOB) Score | **99.55%** |

### Feature Importance

The Random Forest model identified the following features as the most influential for crop recommendation:

| Rank | Feature | Importance |
|-----:|---------|-----------:|
| 1 | Rainfall | **0.2239** |
| 2 | Humidity | **0.2151** |
| 3 | Potassium (K) | **0.1791** |
| 4 | Phosphorus (P) | **0.1528** |
| 5 | Nitrogen (N) | **0.1065** |
| 6 | Temperature | **0.0729** |
| 7 | pH | **0.0498** |

### Sample Prediction

For a sample agricultural observation with:

- Nitrogen (N): **90**
- Phosphorus (P): **42**
- Potassium (K): **43**
- Temperature: **20.88°C**
- Humidity: **82.00%**
- pH: **6.50**
- Rainfall: **202.94 mm**

The Random Forest model recommended:

> 🌾 **Rice**

Top prediction probabilities:

| Crop | Probability |
|------|------------:|
| Rice | **95.51%** |
| Jute | **4.31%** |
| Maize | **0.17%** |
| Papaya | **0.01%** |
| Apple | **0.00%** |

### Key Observations

- Achieved **99.32% classification accuracy** across **22 crop categories**.
- Obtained an **Out-of-Bag Score of 99.55%**, indicating excellent generalization performance without requiring a separate validation dataset.
- Rainfall and humidity emerged as the most influential environmental factors for crop recommendation.
- Soil nutrient levels (Nitrogen, Phosphorus, and Potassium) also played a significant role in determining the recommended crop.
- Successfully generated feature importance rankings, confusion matrix, classification report, and prediction outputs for comprehensive model evaluation.
- Exported the trained Random Forest model and Label Encoder using **Joblib** for future inference.



---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 📚 ML Fundamentals Series

This repository is part of my **ML Fundamentals** series, where each project explores a fundamental machine learning algorithm through practical implementation, visualization, evaluation, and interpretation.
