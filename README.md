# Startup Success Prediction

## Introduction

This Python script implements a Random Forest classifier using the scikit-learn library to predict the success of startups based on various features. The code reads a cleaned dataset from a CSV file, preprocesses the data, splits it into training and testing sets, trains a Random Forest classifier, and evaluates its performance.

## Requirements

- pandas
- scikit-learn

## Installation

1. Clone the repository:


$ git clone https://github.com/Manas5353/startup-success-prediction.git
$ cd startup-success-prediction


2. Install dependencies:

$ pip install -r requirements.txt

## Usage ##

Enter 'cleaned_merged_data.csv' in the script with the actual path to your cleaned dataset CSV file.

## Run the script: ##

$ python startup_success_prediction.py


## Understanding the Code ##

1. Data Preparation
The script reads the CSV dataset, dropping rows with missing values.
Features (X) and the target variable (y) are defined.

2. Feature Engineering
Categorical variables are converted to numerical using one-hot encoding.

3. Train-Test Split
The data is split into training and testing sets (80% training, 20% testing).

4. Model Training
A Random Forest classifier is initialized and trained on the training data.

5. Prediction and Evaluation
Predictions are made on the test set.
The script prints accuracy, confusion matrix, and classification report.

## Evaluation Metrics ##

The script prints the following evaluation metrics:
Accuracy: The overall accuracy of the model.
Confusion Matrix: A table showing true positive, true negative, false positive, and false negative values.
Classification Report: Precision, recall, F1-score, and support for each class.

## Contributing ##
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License ##
This project is licensed under the MIT License.
