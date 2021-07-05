from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from utils.usefull_functions import return_X_y, get_list_single_df
import pandas as pd
import shutil
import copy
from sklearn.metrics import f1_score
import numpy as np

def get_options(task, y):
  f1 = None
  if task == 'classification':
    if len(y[y.columns[0]].unique()) > 2:
      pt = 'multiclass'
      f1 = lambda y_test, y_pred : f1_score(y_test, y_pred, average='weighted')
    else:
      pt = 'binary'
      f1 = lambda y_test, y_pred : f1_score(y_test, y_pred, pos_label=np.unique(y)[0])
  else:
    pt = 'regression'
  return pt, f1

def autogluon(df, task, timelife):
  pd.options.mode.chained_assignment = None
  df_new = copy.copy(df)
  df_new = get_list_single_df(df_new)

  X, y, _ = return_X_y(df_new)
  
  if isinstance(y, pd.Series): y = y.to_frame()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if isinstance(y_train, pd.Series): y_train = y_train.to_frame()
  target = y_train.columns[0]
  if isinstance(y_test, pd.Series): y_test = y_test.to_frame()
  X_train[target] = y_train

  #print(type(X_train), type(X_test), type(y_train), type(y_test))

  pt, f1 = get_options(task, y)

  predictor = TabularPredictor(label=target , problem_type=pt).fit(train_data=X_train, time_limit=timelife*60, presets='best_quality')
  results = predictor.fit_summary()
  y_pred = predictor.predict(X_test)
  #if isinstance(y_pred, pd.Series): y_pred = y_pred.to_frame()
  #print(type(y_pred))
  pipelines = (predictor.leaderboard(X_train, silent=True)).to_markdown()
  #y_test = y_test.squeeze()
  #if isinstance(y_test, pd.Series): y_test = y_test.to_frame()
  res = predictor.evaluate_predictions(y_true=y_test.squeeze(), y_pred=y_pred, auxiliary_metrics=True)

  shutil.rmtree('./AutogluonModels')

  if task != 'classification':
    return (res['root_mean_squared_error'], res['r2'], pipelines)
  return (res['accuracy'],  f1(y_test, y_pred), pipelines)

