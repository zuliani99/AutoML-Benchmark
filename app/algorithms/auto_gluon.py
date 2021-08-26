# Import needed
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from utils.usefull_functions import return_X_y, get_list_single_df
import pandas as pd
import shutil
import copy
from sklearn.metrics import f1_score
import numpy as np
from termcolor import colored
import time


# Definition of the algorithms that autogluon will test
hyperparameters = {
  'GBM': {'num_boost_round': 10000},
  'CAT': {'iterations': 10000},
  'RF': {'n_estimators': 300},
  'XT': {'n_estimators': 300},
  'KNN': {},
}


def autogluon(df, task, options, time_start):
  print(colored("---------------------------------- AUTOGLUON --------------------------------", "blue"))
  try:
    pd.options.mode.chained_assignment = None
    df_new = copy.copy(df) # Deep copy of the DataFrame passed to parameter
    df_new = get_list_single_df(df_new) # Initial cleaning of the DataFrame

    X, y = return_X_y(df_new) # Obtain the two DataFrame X and y needed to execute the train_test_split
    
    if isinstance(y, pd.Series): y = y.to_frame()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    if isinstance(y_train, pd.Series): y_train = y_train.to_frame()
    target = y_train.columns[0]
    if isinstance(y_test, pd.Series): y_test = y_test.to_frame()
    X_train[target] = y_train

    pt, f1 = get_options(task, y)

   # Definition of the predictor with all its hyperparameters
    predictor = TabularPredictor(label=target, problem_type=pt).fit(
      train_data=X_train,
      time_limit=options['time']*60,
      presets=['best_quality'],
      #auto_stack=True
      #hyperparameters=hyperparameters    # -> Questo aggiunge le NN 
    )
    
    predictor.fit_summary()
    y_pred = predictor.predict(X_test)
    pipelines = (predictor.leaderboard(X_train, silent=True)).to_markdown() # Pipeline
    res = predictor.evaluate_predictions(y_true=y_test.squeeze(), y_pred=y_pred, auxiliary_metrics=True)

    shutil.rmtree('./AutogluonModels') # Deleting the folder created for saving the models tested by AutoGluon

    print(colored("---------------------------------- AUTOGLUON --------------------------------", "blue"))

    time_elapsed = round((time.time() - time_start)/60, 3) # Time consumed for computation

    if task != 'classification':
      return (abs(round(res['root_mean_squared_error'], 3)), round(res['r2'], 3), pipelines, time_elapsed)
    # If the parameter 'f1' is not present in the result variable it means that we are in a case of multilables classification and therefore it is necessary to calculate the f1_score manually
    try: return (round(res['accuracy'], 3),  round(res['f1'], 3), pipelines, time_elapsed)
    except: return (round(res['accuracy'], 3),  round(f1(y_test, y_pred), 3), pipelines, time_elapsed)

  except Exception as e:
    # In case of exception
    print(colored('Error: ' + str(e), 'red'))
    if str(e) == 'AutoGluon did not successfully train any models':
      if options['rerun'] == True:
        # If the exception is caused by the short time made available by the user but it has ticked the checkbox for the re-execution of the algorithm, it is re-executed with a longer time
        return autogluon(df, task, {'time': options['time']+1, 'rerun': options['rerun']}, time_start)
      print(colored("---------------------------------- AUTOGLUON --------------------------------", "blue"))
      return (None, None, 'Error duo to short algorithm timelife: ' + str(e), None)
    # Otherwise, None are returned with the exception placed on the pipeline
    print(colored("---------------------------------- AUTOGLUON --------------------------------", "blue"))
    return (None, None, 'Error: ' + str(e), None)


def get_options(task, y):
  f1 = None
  if task == 'classification':
    # Check if it is a case of binary or multilables classification
    if len(y[y.columns[0]].unique()) > 2:
      pt = 'multiclass'
      f1 = lambda y_test, y_pred : f1_score(y_test, y_pred, average='weighted')
    else:
      pt = 'binary'
      f1 = lambda y_test, y_pred : f1_score(y_test, y_pred, pos_label=np.unique(y)[0])
  else:
    pt = 'regression'
  return pt, f1

