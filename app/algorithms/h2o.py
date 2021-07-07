import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.usefull_functions import return_X_y, fill_and_to_category
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import copy


def get_summary(model):
  summary = []
  model.summary(print_fn=lambda x: summary.append(x))
  return '\n'.join(summary)

def prepare_and_test(train, test, task, timelife):
  print(type(train), type(test))

  x = train.columns
  y = train.columns[train.shape[1]-1]
  x.remove(y)

  target = train[y]

  if task == 'classification':
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

  aml = H2OAutoML(max_runtime_secs=timelife*60, nfolds=15, max_models=20, seed=1) #secondi
  aml.train(x, y, training_frame=train)
  lb = aml.leaderboard
  lb = h2o.as_list(lb)

  pred = aml.predict(test)

  pred = h2o.as_list(pred)['predict']
  target = h2o.as_list(test[y])

  pipelines = str((h2o.as_list(h2o.automl.get_leaderboard(aml, extra_columns = 'ALL'))).to_markdown())

  if task != 'classification':
    return (np.sqrt(mean_squared_error(target, pred)), r2_score(target, pred), pipelines, timelife)

  if len(np.unique(target)) > 2:
    return (accuracy_score(target, pred), f1_score(target, pred, average='weighted'), pipelines, timelife)
  return (accuracy_score(target, pred), f1_score(target, pred, pos_label=np.unique(target)[0]), pipelines, timelife)


def H2O(df, task, timelife):
  try:
    return do_h20(df, task, timelife)
  except Exception as e:
    if(str(e) == 'Argument `data` should be an H2OFrame, got NoneType None'):
      return H2O(df, task, timelife+1)
    else:
      raise(e)

def do_h20(df, task, timelife):
  pd.options.mode.chained_assignment = None
  h2o.init()
  df_new = copy.copy(df)

  #if isinstance(df_new, pd.DataFrame):
  df_new = fill_and_to_category(df_new)
  X, y, _ = return_X_y(df_new)
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
    