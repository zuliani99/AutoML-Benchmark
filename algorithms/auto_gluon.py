from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from utils.usefull_functions import get_target
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder 

def autogluon(df, task):
  if type(df) == tuple:
    train = df[0]
    test = df[1]
    #df = TabularDataset(train)
    target = get_target(train, test)
    y = train[target]
    X = train.drop([target], axis=1)
  else:
    #df = TabularDataset(df)
    n_target = df['n_target'][0]
    df = df.drop('n_target', axis = 1)

    y = df.iloc[:, -n_target].to_frame()
    X = df.iloc[:, :-n_target]
  


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if isinstance(y_train, pd.Series):
    y_train = y_train.to_frame()
  target = y_train.columns[0]

  if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame()

  X_train[target] = y_train
  train = X_train

  test = X_test
  
  predictor = TabularPredictor(label=target, path='/home/riccardo/.local/share/Trash').fit(train, time_limit=60, presets=['optimize_for_deployment'])   # TEMPORANEO -> attenzione salvo sul cestino
  results = predictor.fit_summary()
  
  y_pred = predictor.predict(test)

  res = predictor.evaluate_predictions(y_true=y_test.squeeze(), y_pred=y_pred, auxiliary_metrics=True)
  le = LabelEncoder()

  if(task == 'classification'):
    '''y_test = le.fit_transform(y_test)
    y_pred = le.fit_transform(y_pred)
    if len(np.unique(y_pred)) > 2:
      f1 = f1_score(y_test, y_pred, average='weighted')
    else:
      f1 = f1_score(y_test, y_pred)
    return (res['accuracy'], f1)'''
    print(res)
    return (res['accuracy'], res['classification_report']['weighted avg']['f1-score'])
  else:
    print(res)
    return (res['root_mean_squared_error'], res['r2_score'])

