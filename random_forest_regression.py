
# Importing libraries
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegression:
    def __init__(self):
        self.model_name = 'RandomForestRegression'

    def train_model(self, X_train, X_test, y_train, y_test):
        # tree = RandomForestRegressor()
        # parameters = {}
        # grid_search = GridSearchCV(estimator=tree,
        #                            param_grid=parameters,
        #                            scoring='neg_mean_squared_error',  # Метрика за оценка на грид сърч
        #                            cv=10,
        #                            n_jobs=-1)
        # grid_search.fit(X_train, y_train)
        # best_params = grid_search.best_params_
        tree = RandomForestRegressor()  # **best_params
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        cross_validation_score = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=10, n_jobs=-1,
                                                 scoring='neg_mean_squared_error')
        return mse, r2, cross_validation_score.mean()