from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold


def prepare_and_test(X, y, task):
  if task == 'classification':
    model =  TPOTClassifier(generations=5, cv=5, max_time_mins=1, random_state=1, verbosity=2)
    score = lambda t, p: accuracy_score(t, p)
  else:
    model =  TPOTRegressor(generations=5, cv=5, max_time_mins=1, random_state=1, verbosity=2)
    score = lambda t, p: np.sqrt(mean_squared_error(y_true=t, y_pred=p))

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  return (score(y_test, preds))


#devo fare datacleaning: pulizia nel senso nan -> fill_nan
def TPOT(df, task):

  for col in df.columns:
    t = pd.api.types.infer_dtype(df[col])
    if t == "string" or t == 'object':
      df[col] = df[col].astype('category').cat.codes
    if t == 'categorical' :
      df[col] = df[col].cat.codes

    
  y = df.iloc[:, -1]
  X = df.iloc[:, :-1]

  return prepare_and_test(X, y, task)