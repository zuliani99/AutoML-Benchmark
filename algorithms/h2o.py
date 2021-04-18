import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.usefull_functions import get_target

def prepare_and_test(train, test, task):
  x = train.columns
  y = train.columns[train.shape[1]-1]
  x.remove(y)

  target = train[y]

  if task == 'classification':
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

  aml = H2OAutoML(max_runtime_secs=1*60, nfolds=15, max_models=20, seed=1)
  aml.train(x, y, training_frame=train)
  lb = aml.leaderboard
  lb.head(rows=lb.nrows)

  pred = aml.predict(test)

  pred = h2o.as_list(pred)
  target = h2o.as_list(test[y])


  if task == 'classification':
    pred = pred.drop(pred.columns[-2:].astype(str), 1)

    temp = pred
    temp['true'] = target

    temp['predict'] = temp['predict'].astype(str)
    temp['true'] = temp['true'].astype(str)
    temp['ver'] = np.where(temp['predict'] == temp['true'], 1, 0)
    
    return (temp['ver'].sum()/pred.shape[0])
  else:
    return np.sqrt(np.mean((target.to_numpy() - pred.to_numpy())**2))


def H2O(df, task):
  h2o.init()

  if isinstance(df, pd.DataFrame):
    y = df.iloc[:, -1].to_frame()
    X = df.iloc[:, :-1]
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