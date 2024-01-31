# Heart Disease Prediction Machine Learning Model

This repository contains a machine learning model for predicting the likelihood of heart disease in individuals based on key health metrics. The model, trained on a dataset with 303 entries and 14 columns, achieves an accuracy of 81.967%.

## Overview

Heart disease is a critical health concern, and early prediction can play a pivotal role in timely intervention and treatment. This machine learning model utilizes a dataset featuring essential health parameters such as age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol levels (chol), fasting blood sugar (fbs), rest electrocardiographic results (restecg), maximum heart rate achieved (thalach), exercise-induced angina (exang), oldpeak, slope, number of major vessels colored by fluoroscopy (ca), thalassemia (thal), and the target variable indicating the presence or absence of heart disease.

## Features

- **Machine Learning Algorithms**: The model employs logistic regression and support vector machine (SVM) algorithms for heart disease prediction.

- **Accuracy**: Achieving an accuracy of 81.967%, this model provides reliable predictions for identifying individuals at risk of heart disease.

- **Input Features**: The model considers a comprehensive set of health metrics, ensuring a robust analysis for accurate predictions.

## Data

### Columns

1. **age**: Age of the individual
2. **sex**: Gender (0 for female, 1 for male)
3. **cp**: Chest pain type
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 for true, 0 for false)
7. **restecg**: Resting electrocardiographic results
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise-induced angina (1 for true, 0 for false)
10. **oldpeak**: Depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment
12. **ca**: Number of major vessels colored by fluoroscopy
13. **thal**: Thalassemia
14. **target**: Presence of heart disease (1 for healthy heart, 0 for defective heart)

### Usage

1. **Download the Dataset**: Access the dataset from the above uploaded data file

2. **File Format**: The dataset is provided in a CSV format, facilitating seamless integration for training and evaluation.

3. **Data Exploration**: Explore the dataset to understand the distribution of features and labels before utilizing the dataset for model training.

### Model Evaluation

- **Accuracy on Training Data**: 85.12%
- **Accuracy on Test Data**: 81.97%

## How to Use

1. **Training the Model**: Utilize the provided Jupyter notebook or script for training the model on the above provided dataset.

2. **Prediction**: Leverage the trained model for heart disease prediction by providing relevant input features.

## How to Contribute

Contributions are welcome! Whether you're enhancing the model, adding features, or improving documentation, follow the standard GitHub workflow. Fork the repository, create a branch, make changes, and submit a pull request.

