# Import needed
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from utils.usefull_functions import return_X_y, fill_and_to_category
import copy
from termcolor import colored

from dask.distributed import Client
import joblib
import shutil

def make_classification(X_train, X_test, y_train, y_test, timelife, y, client):

  # Model Classification
  model =  TPOTClassifier(generations=100, cv=10, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=-1, max_eval_time_mins=5)
  #model.fit(X_train, y_train)

  with joblib.parallel_backend("dask"):
    model.fit(np.array(X_train), np.array(y_train).ravel())

  y_pred = model.predict(X_test)
  pipelines = model.export() # Get the pipeline

  print("-----------------------------------TPOT------------------------------------\n\n")
  client.close()
  shutil.rmtree('./dask-worker-space')

  # Check if it is a binary or multilables case
  if len(np.unique(y)) > 2:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, average='weighted'), 3), pipelines, timelife
  else:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), pipelines, timelife


def make_regression(X_train, X_test, y_train, y_test, timelife, client):

  # Regression model
  model =  TPOTRegressor(generations=100, cv=10, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=-1, max_eval_time_mins=5)
  #model.fit(X_train, y_train)

  with joblib.parallel_backend("dask"):
    model.fit(np.array(X_train), np.array(y_train).ravel())

  y_pred = model.predict(X_test)
  pipelines = model.export() # Get the pipeline

  print("-----------------------------------TPOT------------------------------------\n\n")
  client.close()
  shutil.rmtree('./dask-worker-space')

  return round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines, timelife


def TPOT(df, task, options):
  print("-----------------------------------TPOT------------------------------------")
  try:
    client = Client(processes=False)
    df_new = copy.copy(df) # Deep copy of the DataFrame passed to parameter
    pd.options.mode.chained_assignment = None

    df_new = fill_and_to_category(df_new) # Initial cleaning of the DataFrame
    X, y = return_X_y(df_new) # Obtain the two DataFrame X and y needed to execute the train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    if task == 'classification':
      return make_classification(X_train, X_test, y_train, y_test, options['time'], y, client)
    else:
      return make_regression(X_train, X_test, y_train, y_test, options['time'], client)

  except Exception as e:
    # In case of exception
    print(colored('Error: ' + str(e), 'red'))
    if str(e) == 'There was an error in the TPOT optimization process. This could be because the data was not formatted properly, or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly. If you enabled PyTorch estimators, please check the data requirements in the online documentation: https://epistasislab.github.io/tpot/using/':
      if options['rerun'] == True:
        # If the exception is caused by the short time made available by the user but it has ticked the checkbox for the re-execution of the algorithm, it is re-executed with a longer time
        return TPOT(df, task, {'time': options['time']+5, 'rerun': options['rerun']})
      print("-----------------------------------TPOT------------------------------------\n\n")
      return (None, None, 'Error duo to short algorithm timelife: ' + str(e), None)

    # Otherwise, None are returned with the exception placed on the pipeline
    print("-----------------------------------TPOT------------------------------------\n\n")
    return (None, None, 'Error: ' + str(e), None)

