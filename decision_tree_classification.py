
# Importing libraries
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegression:
    def __init__(self):
        self.model_name = 'DecisionTreeRegression'

    def train_model(self, X_train, X_test, y_train, y_test):
        # classifier = DecisionTreeRegressor()
        # parameters = {'max_depth': [3, 5, 7],
        #               'min_samples_split': [2, 5, 10]}
        # grid_search = GridSearchCV(estimator=classifier,
        #                            param_grid=parameters,
        #                            scoring='neg_mean_squared_error',
        #                            cv=10,
        #                            n_jobs=-1)
        # grid_search.fit(X_train, y_train)
        # best_params = grid_search.best_params_
        classifier = DecisionTreeRegressor() #**best_params
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cross_validation_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=2, n_jobs=-1,
                                                 scoring='r2')
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2, cross_validation_score.mean()



