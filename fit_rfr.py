import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# sklearn Analysis Suite
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('data_for_fit.csv')
df.drop('student_id', axis = 1, inplace = True)
X = df.drop('score', axis = 1)
y = df['score']

test_size = 0.20
seed = 7

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size=test_size,
                                                                    random_state=seed)
# max_depth options in linear space from None - 100
max_depth = [int(x) for x in np.linspace(1, 20, num = 11)]
max_depth.append(None)

tree_params = {
               'max_depth': max_depth,
               'n_estimators': np.logspace(1, 5, num=10, dtype='int')
               }

adaboost_params = {
                'base_estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=4), DecisionTreeRegressor(max_depth=5)],
                'learning_rate' : [0.01,0.05,0.1,0.3,1,2],
                'loss' : ['linear', 'square'],
                'n_estimators': np.logspace(1, 4, num=10, dtype='int')
                }

gboost_params = {
                'learning_rate' : [0.01,0.05,0.1,0.3,1,2],
                'n_estimators': np.logspace(1, 4, num=10, dtype='int'),
                'loss' : ['ls', 'lad']
                }

models = []
# models.append(('ABR', AdaBoostRegressor(), adaboost_params))
models.append(('RFR', RandomForestRegressor(), tree_params))
# models.append(('GBR', GradientBoostingRegressor(), gboost_params))
# models.append(('ETR', ExtraTreesRegressor(), tree_params))

# Evaluate each model and pick most optimized model with associated hyper parameters
results = []
names = []
best_estimators = []
all_best_params = []
all_cv_results = []
all_model_results = []

for name, model, params in models:
    print(name)
    rs_cv = model_selection.RandomizedSearchCV(estimator=model,
                                               param_distributions=params,
                                               cv=5,
                                               random_state=seed)

    model_result = rs_cv.fit(X_train, y_train)
    best_estimator = model_result.best_estimator_
    score = model_result.best_score_
    best_params = model_result.best_params_
    cv_results = model_result.cv_results_

    all_model_results.append(model_result)
    all_cv_results.append(cv_results)
    all_best_params.append(best_params)
    best_estimators.append(best_estimator)
    results.append(score)
    names.append(name)
    msg = "%s: %f" % (name, score)
    print(msg)

import pickle
to_pick = {
    "results": results,
    "names": names,
    "best_estimator": best_estimator,
    'all_best_params' :all_best_params ,
    'all_cv_results' :all_cv_results,
    'all_model_results':all_model_results
}

def pickle_file(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

file_path = 'rfr_fitted_stuff.pickle'

pickle_file(file_path, to_pick)
