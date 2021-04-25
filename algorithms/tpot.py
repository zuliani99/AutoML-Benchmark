from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from utils.usefull_functions import get_target


def prepare_and_test(X, y, task):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  le = LabelEncoder()
  if task == 'classification':
    model =  TPOTClassifier(generations=5, cv=5, max_time_mins=1, random_state=1, verbosity=2, n_jobs=-1)
    model.fit(X_train, y_train)

    y_test = le.fit_transform(y_test)
    y_pred = le.fit_transform(model.predict(X_test))

    if len(np.unique(y_test)) > 2:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'))
    else:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))
  else:
    model =  TPOTRegressor(generations=5, cv=5, max_time_mins=1, random_state=1, verbosity=2, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred))


#devo fare datacleaning: pulizia nel senso nan -> fill_nan
def TPOT(df, task):
  if isinstance(df, pd.DataFrame):
    for col in df.columns:
      t = pd.api.types.infer_dtype(df[col])
      if t == "string" or t == 'object':
        df[col] = df[col].astype('category').cat.codes
      if t == 'categorical' :
        df[col] = df[col].cat.codes

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

  else:
    train = df[0]
    test = df[1]
    target = get_target(train, test)
    y = train[target]
    train = train.drop([target], axis=1)
    X = train.apply(LabelEncoder().fit_transform)

  return prepare_and_test(X, y, task)
