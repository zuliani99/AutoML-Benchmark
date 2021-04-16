import pandas as pd
import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split

def prepare_and_test(x_train, x_test, target, task):

  y_train = x_train.pop(target)
  y_train = pd.DataFrame(y_train)
  x_train = x_train.to_numpy()
  y_train = y_train.to_numpy()
  y_test = x_test.pop(target)


  if(task == 'classification'):
    clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3, directory='/media/riccardo/8A13-C277/model_saved')
  else:
    clf = ak.StructuredDataRegressor(overwrite=True, max_trials=3, directory='/media/riccardo/8A13-C277/model_saved')
    
  clf.fit(x_train, y_train, validation_split=0.15, epochs=5)
  predicted_y = clf.predict(x_test)

  return clf.evaluate(x_test, y_test) if task == 'classification' else np.sqrt(clf.evaluate(x_test, y_test)[0])


def autokeras(df, task):
  y = df.iloc[:, -1]
  X = df.iloc[:, :-1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  y_train = y_train.to_frame() 
  target = y_train.columns[0]
  X_train[target] = y_train
  train = X_train

  y_test = y_test.to_frame() 
  X_test[target] = y_test
  test = X_test

  return (prepare_and_test(train, test, target, task))