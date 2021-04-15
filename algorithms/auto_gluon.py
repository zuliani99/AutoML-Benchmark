from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

def autogluon(df, task):
  df = TabularDataset(df)

  y = df.iloc[:, -1].to_frame()
  X = df.iloc[:, :-1]


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  target = y_train.columns[0]
  X_train[target] = y_train
  train = X_train

  test = X_test

  predictor = TabularPredictor(label=target).fit(train, time_limit=60, presets=['optimize_for_deployment'])
  results = predictor.fit_summary()
  
  y_pred = predictor.predict(test)

  if(task == 'classification'):
    return (predictor.evaluate_predictions(y_true=y_test.squeeze(), y_pred=y_pred, auxiliary_metrics=True))['accuracy']
  else:
    return (predictor.evaluate_predictions(y_true=y_test.squeeze(), y_pred=y_pred, auxiliary_metrics=True))['root_mean_squared_error']