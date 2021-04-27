import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.usefull_functions import get_target
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score

def prepare_and_test(train, test, task):
  x = train.columns
  y = train.columns[train.shape[1]-1]
  x.remove(y)

  target = train[y]

  if task == 'classification':
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

  aml = H2OAutoML(max_runtime_secs=2*60, nfolds=15, max_models=20, seed=1)
  aml.train(x, y, training_frame=train)
  lb = aml.leaderboard
  lb = h2o.as_list(lb)

  pred = aml.predict(test)

  pred = h2o.as_list(pred)['predict']
  target = h2o.as_list(test[y])

  le = LabelEncoder()

  if task[0] == 'classification':
    target = le.fit_transform(target)
    pred = le.fit_transform(pred)
    return (accuracy_score(target, pred), f1_score(target, pred))
  else:
    return (np.sqrt(mean_squared_error(target, pred)), r2_score(target, pred))


def H2O(df, task):
  h2o.init()

  if isinstance(df, pd.DataFrame):
    n_target = df['n_target'][0]
    df = df.drop('n_target', axis = 1)

    y = df.iloc[:, -n_target].to_frame()
    X = df.iloc[:, :-n_target]
  else:
    train = df[0]
    test = df[1]
    target = get_target(train, test)
    y = train[target]
    X = train.drop(target, axis=1)

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

  return(prepare_and_test(train, test, task))