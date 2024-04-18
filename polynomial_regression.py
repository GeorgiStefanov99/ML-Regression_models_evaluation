
# Importing libraries
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score


class PolynomialRegression:
    def __init__(self):
        self.model_name = 'MultipleLinearRegression'

    def train_model(self, X_train, X_test, y_train, y_test):
        poly_reg = PolynomialFeatures(degree=4)
        X_poly_train = poly_reg.fit_transform(X_train)
        X_poly_test = poly_reg.transform(X_test)
        parameters = {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'n_jobs': [-1],
        }
        grid_search = GridSearchCV(estimator=LinearRegression(),
                                   param_grid=parameters,
                                   scoring='neg_mean_squared_error',
                                   cv=2,
                                   n_jobs=-1)
        grid_search.fit(X_poly_train, y_train)
        best_params = grid_search.best_params_
        regressor = LinearRegression()
        regressor.fit(X_poly_train, y_train)
        y_pred = regressor.predict(X_poly_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cross_validation_score = cross_val_score(estimator=regressor, X=X_poly_train, y=y_train, cv=2, n_jobs=-1,
                                                 scoring='neg_mean_squared_error')
        return mse, r2, cross_validation_score.mean()


