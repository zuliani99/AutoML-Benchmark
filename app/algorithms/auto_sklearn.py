# Import needed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import autosklearn.classification
import autosklearn.regression
import pandas as pd
import numpy as np
from utils.usefull_functions import return_X_y, fill_and_to_category
import copy
from termcolor import colored
import psutil


def make_classification(X_train, X_test, y_train, y_test, timelife, y):
  # Classification model
  automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=timelife*60,
          per_run_time_limit=30,
          memory_limit=psutil.virtual_memory().available,
          n_jobs=-1,
          resampling_strategy_arguments = {'cv': 10}
    )
  automl.fit(X_train, y_train)
  y_pred = automl.predict(X_test)
  pipelines = str(pd.DataFrame(pd.Series(automl.show_models())).iloc[0].squeeze()) # Pipeline
  print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
  # Check if it is a binary or multilables case
  if len(np.unique(y)) > 2:
    return (round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, average='weighted'), 3), pipelines, timelife)
  else:
    return (round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), pipelines, timelife)


def make_regression(X_train, X_test, y_train, y_test, timelife):
  # Regression model
  automl = autosklearn.regression.AutoSklearnRegressor(
          time_left_for_this_task=timelife*60,
          per_run_time_limit=30,
          memory_limit=psutil.virtual_memory().available,
          n_jobs=-1,
          resampling_strategy_arguments = {'cv': 10}
    )
  automl.fit(X_train, y_train)
  y_pred = automl.predict(X_test)
  pipelines = str(pd.DataFrame(pd.Series(automl.show_models())).iloc[0].squeeze().split('\n')) # Pipeline
  print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
  return (round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines, timelife)


def auto_sklearn(df, task, options):
  print("--------------------------------AUTOSKLEARN--------------------------------")
  try:
    df_new = copy.copy(df) # Deep copy of the DataFrame passed to parameter
    pd.options.mode.chained_assignment = None

    df_new = fill_and_to_category(df_new) # Initial cleaning of the DataFrame
    X, y = return_X_y(df_new) # Obtain the two DataFrame X and y needed to execute the train_test_split


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    if(task == 'classification'):
      return make_classification(X_train, X_test, y_train, y_test, options['time'], y)
    else:
      return make_regression(X_train, X_test, y_train, y_test, options['time'])

  except Exception as e:
    # In case of exception
    print(colored('Error: ' + str(e), 'red'))
    if str(e) == 'No valid model found in run history. This means smac was not able to fit a valid model. Please check the log file for errors.':
      if options['rerun'] == True:
        # If the exception is caused by the short time made available to the user but it has ticked the checkbox for the re-execution by the algorithm, it will be re-executed with a longer time
        return auto_sklearn(df, task, {'time': options['time']+1, 'rerun': options['rerun']})
      print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
      return (None, None, 'Error duo to short algorithm timelife: ' + str(e), None)
    # Otherwise, None are returned with the exception placed on the pipeline
    print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
    return (None, None, 'Error: ' + str(e), None)
    
    