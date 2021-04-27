import pandas as pd
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
from utils.usefull_functions import get_target
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder 

def prepare_and_test(x_train, x_test, target, task):

  y_train = x_train.pop(target)
  y_train = pd.DataFrame(y_train)
  x_train = x_train.to_numpy()
  y_train = y_train.to_numpy()
  y_test = x_test.pop(target)


  if(task == 'classification'):
    clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3, directory='/home/riccardo/.local/share/Trash')#attenzione salvo sul cestino
  else:
    clf = ak.StructuredDataRegressor(overwrite=True, max_trials=3, directory='/home/riccardo/.local/share/Trash')#attenzione salvo sul cestino
    
  clf.fit(x_train, y_train, validation_split=0.15, epochs=5)
  predicted_y = clf.predict(x_test)

  if task == 'classification':
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    y_pred = le.fit_transform(predicted_y)
    if len(np.unique(predicted_y)) > 2:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'))
    else:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))
  else:
    return (np.sqrt(mean_squared_error(y_test, predicted_y)), r2_score(y_test, predicted_y))


def autokeras(df, task):
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
    X = train.drop([target], axis=1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if isinstance(y_train, pd.Series):
    y_train = y_train.to_frame()
  target = y_train.columns[0]
  X_train[target] = y_train
  train = X_train

  if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame() 
  X_test[target] = y_test
  test = X_test

  return (prepare_and_test(train, test, target, task))