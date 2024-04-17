from decision_tree_classification import DecisionTreeRegression
from multiple_linear_regression import MultipleLinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from polynomial_regression import PolynomialRegression
from random_forest_regression import RandomForestRegression
from support_vector_regression import SupportVectorRegression

dtc, mlr, pr, rfr, svm = (DecisionTreeRegression(), MultipleLinearRegression(), PolynomialRegression(),
                     RandomForestRegression(), SupportVectorRegression())
classification_models = (dtc, mlr, pr, rfr, svm)


def get_data():
    # Import dataset
    dataset = pd.read_csv('Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test


def get_best_model():
    X_train, X_test, y_train, y_test = get_data()
    trained_models = []
    for model in classification_models:
        acc_score, conf_score, cross_val_score = model.train_model(X_train, X_test, y_train, y_test)
        trained_models.append((f'Model: {model.model_name}\nR2 score: {acc_score:.2f}\nMean squared error: '
                               f'{conf_score}\nCross validation score: {cross_val_score:.2f}'))
        trained_models.sort(key=lambda x: x[3], reverse=True)
    return f'Best model: {trained_models[0]}\nSecond model by performance: {trained_models[1]}'


print(get_best_model())
