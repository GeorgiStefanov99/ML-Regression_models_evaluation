
# Importing libraries
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR


class SupportVectorRegression:
    def __init__(self):
        self.model_name = 'SupportVectorRegression'

    def train_model(self, X_train, X_test, y_train, y_test):
        # svm_regressor = SVR()
        # parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #               'C': [0.1, 1, 10, 100],
        #               'epsilon': [0.1, 0.2, 0.3, 0.5]}
        # grid_search = GridSearchCV(estimator=svm_regressor,
        #                            param_grid=parameters,
        #                            scoring='neg_mean_squared_error',
        #                            cv=10,
        #                            n_jobs=-1)
        # grid_search.fit(X_train, y_train)
        # best_params = grid_search.best_params_
        svm_regressor = SVR() # **best_params
        svm_regressor.fit(X_train, y_train)
        y_pred = svm_regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cross_validation_score = cross_val_score(estimator=svm_regressor, X=X_train, y=y_train, cv=2, n_jobs=-1,
                                                 scoring='neg_mean_squared_error')
        return mse, r2, -cross_validation_score.mean()