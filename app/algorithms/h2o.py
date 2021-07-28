# Import needed
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.usefull_functions import return_X_y, fill_and_to_category, get_list_single_df
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import copy
from termcolor import colored

# Function for getting the pipeline
def get_summary(model):
  summary = []
  model.summary(print_fn=lambda x: summary.append(x))
  return '\n'.join(summary)

def prepare_and_test(train, test, task, timelife):
  x = train.columns
  y = train.columns[train.shape[1]-1]
  x.remove(y)

  target = train[y]

  if task == 'classification':
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

  aml = H2OAutoML(max_runtime_secs=timelife*60, nfolds=10, max_models=50, seed=1)
  aml.train(x, y, training_frame=train)
  lb = aml.leaderboard
  lb = h2o.as_list(lb)

  pred = aml.predict(test)

  pred = h2o.as_list(pred)['predict']
  target = h2o.as_list(test[y])

  pipelines = str((h2o.as_list(h2o.automl.get_leaderboard(aml, extra_columns = 'ALL'))).to_markdown()) # Pipeline
  h2o.shutdown() # Termination of the H2O cluster
  
  print("------------------------------------H2O------------------------------------\n\n")
  
  if task != 'classification':
    return (round(np.sqrt(mean_squared_error(target, pred)), 3), round(r2_score(target, pred), 3), pipelines, timelife)

  # Check if it is a binary or multilables case
  if len(np.unique(target)) > 2:
    return (round(accuracy_score(target, pred), 3), round(f1_score(target, pred, average='weighted'), 3), pipelines, timelife)
  return (round(accuracy_score(target, pred), 3), round(f1_score(target, pred, pos_label=np.unique(target)[0]), 3), pipelines, timelife)


def H2O(df, task, options):
  print("------------------------------------H2O------------------------------------")
  try:
    return do_h20(df, task, options['time'])

  except Exception as e:
    # In case of exception
    print(colored('Error: ' + str(e), 'red'))
    if str(e) == 'Argument `data` should be an H2OFrame, got NoneType None':
      if options['rerun'] == True:
        # If the exception is caused by the short time made available by the user but it has ticked the checkbox for the re-execution of the algorithm, it is re-executed with a longer time
        return H2O(df, task, {'time': options['time'] + 1, 'rerun': options['rerun']})
      print('------------------------------------H2O------------------------------------\n\n')
      return None, None, 'Error duo to short algorithm timelife: ' + str(e), None
    # Otherwise, None are returned with the exception placed on the pipeline
    print('------------------------------------H2O------------------------------------\n\n')
    return None, None, 'Error: ' + str(e), None

def do_h20(df, task, timelife):
  pd.options.mode.chained_assignment = None
  h2o.init()  # Starting the H2O cluster
  df_new = copy.copy(df) # Deep copy of the DataFrame passed to parameter

  df_new = get_list_single_df(df_new)
  X, y = return_X_y(df_new) # Obtain the two DataFrame X and y needed to execute the train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

  if isinstance(y_test, pd.Series):
    y_train = y_train.to_frame()
  X_train[y_train.columns[0]] = y_train
  train = X_train

  if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame()
  X_test[y_test.columns[0]] = y_test
  test = X_test

  train = h2o.H2OFrame(train)
  test = h2o.H2OFrame(test)

  return(prepare_and_test(train, test, task, timelife))
    