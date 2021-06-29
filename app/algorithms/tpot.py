from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from utils.usefull_functions import return_X_y, fill_and_to_category
import copy

def get_stat(model):
  return model.export()

#devo fare datacleaning: pulizia nel senso nan -> fill_nan
def TPOT(df, task, timelife):
  df_new = copy.copy(df)
  pd.options.mode.chained_assignment = None
  #if isinstance(df_new, pd.DataFrame):
  df_new = fill_and_to_category(df_new)
  X, y, _ = return_X_y(df_new)
  #if not isinstance(df_new, pd.DataFrame):
    #X = fill_and_to_category(X)


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  le = LabelEncoder()
  if task == 'classification':
    model =  TPOTClassifier(generations=timelife, cv=5, max_time_mins=1, random_state=1, verbosity=2, n_jobs=1)
    model.fit(X_train, y_train)

    y_test = le.fit_transform(y_test)
    y_pred = le.fit_transform(model.predict(X_test))

    pipelines = get_stat(model)

    if len(np.unique(y)) > 2:
      return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), pipelines
    else:
      return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), pipelines
  else:
    model =  TPOTRegressor(generations=timelife, cv=5, max_time_mins=1, random_state=1, verbosity=2, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pipelines = get_stat(model)
    return np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred), pipelines
