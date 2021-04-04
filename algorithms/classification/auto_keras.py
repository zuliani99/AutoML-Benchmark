#pip3 install autokeras

import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split

def prepare_and_test(train_file_path, test_file_path, target):
  # x_train as pandas.DataFrame, y_train as pandas.Series
  x_train = pd.read_csv(train_file_path)

  print(type(x_train))  # pandas.DataFrame
  y_train = x_train.pop(target)
  print(type(y_train))  # pandas.Series

  y_train = pd.DataFrame(y_train)
  print(type(y_train))  # pandas.DataFrame

  x_train = x_train.to_numpy()
  y_train = y_train.to_numpy()
  print(type(x_train))  # numpy.ndarray
  print(type(y_train))  # numpy.ndarray

  # Preparing testing data.
  x_test = pd.read_csv(test_file_path)
  y_test = x_test.pop(target)

  # It tries 10 different models.
  clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
  # Feed the structured data classifier with training data.
  clf.fit(x_train, y_train, validation_split=0.15, epochs=10)
  # Predict with the best model.
  predicted_y = clf.predict(x_test)
  # Evaluate the best model with testing data.
  return (clf.evaluate(x_test, y_test))


def autokeras_class(df):
  y = df.iloc[:, -1]
  X = df.iloc[:, :-1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  #y_train = y_train.to_frame() 
  target = y_train.columns[0]
  X_train[target] = y_train
  train = X_train

  #y_test = y_test.to_frame() 
  X_test[target] = y_test
  test = X_test

  train.to_csv("../train.csv", index=False, header=True)
  test.to_csv("../test.csv", index=False, header=True)

  return (prepare_and_test("../train.csv", "../test.csv", target))