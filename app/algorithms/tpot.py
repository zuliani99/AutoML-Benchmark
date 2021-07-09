from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from utils.usefull_functions import return_X_y, fill_and_to_category
import copy
import dash_html_components as html

le = LabelEncoder()

def make_classification(X_train, X_test, y_train, y_test, timelife, y):
  model =  TPOTClassifier(generations=3, cv=5, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=1, max_eval_time_mins=0.05) #, subsample=0.5
  #print(type(X_train), type(y_train))
  #model.fit(np.array(X_train), np.array(y_train).ravel())
  model.fit(X_train, y_train)

  #y_test = le.fit_transform(y_test)
  #y_pred = le.fit_transform(model.predict(X_test))

  y_pred = model.predict(X_test)

  pipelines = model.export() #get_stat(model)

  if len(np.unique(y)) > 2:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, average='weighted'), 3), pipelines, timelife
  else:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), pipelines, timelife

def make_regression(X_train, X_test, y_train, y_test, timelife):
  model =  TPOTRegressor(generations=3, cv=5, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=1, max_eval_time_mins=0.05) #, subsample=0.5
  print(type(X_train), type(y_train))
  #model.fit(np.array(X_train), np.array(y_train).ravel())
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  pipelines = model.export() #get_stat(model)
  return round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines, timelife

#devo fare datacleaning: pulizia nel senso nan -> fill_nan
def TPOT(df, task, timelife):
  try:
    df_new = copy.copy(df)
    pd.options.mode.chained_assignment = None
    #if isinstance(df_new, pd.DataFrame):
    df_new = fill_and_to_category(df_new)
    X, y = return_X_y(df_new)
    #if not isinstance(df_new, pd.DataFrame):
      #X = fill_and_to_category(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    if task == 'classification':
      return make_classification(X_train, X_test, y_train, y_test, timelife, y)
    else:
      return make_regression(X_train, X_test, y_train, y_test, timelife)
  except Exception as e:
    if str(e) == 'There was an error in the TPOT optimization process. This could be because the data was not formatted properly, or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly. If you enabled PyTorch estimators, please check the data requirements in the online documentation: https://epistasislab.github.io/tpot/using/':
      return TPOT(df, task, timelife+5)
    else:
      raise(e)

