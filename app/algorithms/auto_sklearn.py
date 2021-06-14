from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import autosklearn.classification
import autosklearn.regression
import pandas as pd
import numpy as np
from utils.usefull_functions import return_X_y, fill_and_to_category

def get_stat(automl):
  automl.sprint_statistics()
  automl.show_models()

def auto_sklearn(df, task):
  pd.options.mode.chained_assignment = None
  #categorical, binary, nuymerical features

  if isinstance(df, pd.DataFrame):
    df = fill_and_to_category(df)
  X, y, _ = return_X_y(df)
  if not isinstance(df, pd.DataFrame):
    X = fill_and_to_category(X)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  
  if(task == 'classification'):
    automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=1*60,
          per_run_time_limit=30,
          n_jobs=-1
    )
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)

    get_stat(automl) # necessario il salvataggio

    if len(np.unique(y)) > 2:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'))
    else:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))
  else:
    automl = autosklearn.regression.AutoSklearnRegressor(
          time_left_for_this_task=3*60,
          per_run_time_limit=30,
          n_jobs=-1
    )
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    
    get_stat(automl) # necessario il salvataggio

    return (np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred))

  
#print(automl.sprint_statistics())
#print(automl.show_models())