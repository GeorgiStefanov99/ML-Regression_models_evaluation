# Importing libraries
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression:
    def __init__(self):
        self.model_name = 'MultipleLinearRegression'

    def train_model(self, X_train, X_test, y_train, y_test):
        # regressor = LinearRegression()
        # parameters = {}
        # grid_search = GridSearchCV(estimator=regressor,
        #                            param_grid=parameters,
        #                            scoring='neg_mean_squared_error',
        #                            cv=10,
        #                            n_jobs=-1)
        # grid_search.fit(X_train, y_train)
        # best_params = grid_search.best_params_
        regressor = LinearRegression() # **best_params
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cross_validation_score = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10, n_jobs=-1,
                                                 scoring='neg_mean_squared_error')
        return mse, r2, cross_validation_score.mean()


