from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from utils.usefull_functions import get_target
import pandas as pd

def autogluon(df, task):
  if type(df) == tuple:
    train = df[0]
    test = df[1]
    #df = TabularDataset(train)
    target = get_target(train, test)
    y = train[target]
    X = train.drop([target], axis=1)
  else:
    df = TabularDataset(df)
    y = df.iloc[:, -1].to_frame()
    X = df.iloc[:, :-1]
  


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if isinstance(y_train, pd.Series):
    y_train = y_train.to_frame()
  target = y_train.columns[0]

  if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame()
  X_train[target] = y_train
  train = X_train

  test = X_test

  predictor = TabularPredictor(label=target, path='/media/riccardo/8A13-C277/model_saved').fit(train, time_limit=60, presets=['optimize_for_deployment'])   # TEMPORANEO
  results = predictor.fit_summary()
  
  y_pred = predictor.predict(test)

  if(task == 'classification'):
    res = predictor.evaluate_predictions(y_true=y_test.squeeze(), y_pred=y_pred, auxiliary_metrics=True)
    return (res['accuracy'], res['f1_score'])
  else:
    res = predictor.evaluate_predictions(y_true=y_test.squeeze(), y_pred=y_pred, auxiliary_metrics=True)
    return (res['root_mean_squared_error'], res['r2_score'])

