# ML-Regression_models_evaluation

This repository contains implementations of various regression models in Python.

## Overview
The main purpose of this project is to compare the performance of different regression algorithms on a given dataset and choose the best one among all. The models included in this project are:

- Decision Tree Regression-
- Multiple Linear Regression
- Polynomial Regression
- Random Forest Regression
- Support Vector Regression (SVR)

##  Usage

To use these models, follow these steps:

1. Install the required dependencies by running:
    pip install -r requirements.txt
2. Place the .csv file in the main folder. Then, go to `main.py` and on line 21, replace `'Position_Salaries.csv'` with the name of your dataset file inside the brackets. For example:
   `dataset = pd.read_csv('Name_of_your_dataset.csv')`
   ```Position_Salaries.csv is left for testing purposes if you want to see how application perform just run `main.py` ```
   2.1 NOTE: Current implementation does NOT take care of any missing data.
   2.2 NOTE: Grid_search is shadowed because of the size of the dataset, if your dataset is bigger it may be good option to include grid search
   
3. Run the main.py script to train and evaluate the models on the given dataset.
 The script will output the best performing model along with its R2 score, mean squared error, and cross-validation score.

## Dataset
The models are trained and evaluated on the provided dataset, which contains various features and corresponding target values.

## Implementation Details
Each model is implemented as a separate Python class, with a train_model method that takes training and test data as input and returns the mean squared error, R2 score, and cross-validation score.

## Example
```python
from decision_tree_regression import DecisionTreeRegression
from multiple_linear_regression import MultipleLinearRegression
from polynomial_regression import PolynomialRegression
from random_forest_regression import RandomForestRegression
from support_vector_regression import SupportVectorRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize regression models
dtc = DecisionTreeRegression()
mlr = MultipleLinearRegression()
pr = PolynomialRegression()
rfr = RandomForestRegression()
svr = SupportVectorRegression()

regression_models = (dtc, mlr, pr, rfr, svr)

# Load and preprocess dataset
def get_data():
    # Import dataset
    dataset = pd.read_csv('Data.csv')
    
    # Separate features (X) and labels (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Return the preprocessed data
    return X_train, X_test, y_train, y_test

# Train each model on the dataset and get best based on cross-validation score
def get_best_model():
    X_train, X_test, y_train, y_test = get_data()
    trained_models = []
    for model in regression_models:
        mse, r2, cross_val_score = model.train_model(X_train, X_test, y_train, y_test)
        trained_models.append((f'Model: {model.model_name}\nR2 score: {r2:.2f}\nMean squared error: '
                               f'{mse}\nCross validation score: {cross_val_score:.2f}'))
        trained_models.sort(key=lambda x: x[3], reverse=True)
    return trained_models[0]

print(get_best_model())
This script trains and evaluates each regression model on the provided dataset and outputs the best performing model based on the cross-validation score. Adjust the dataset and configurations as needed for your specific regression tasks.
