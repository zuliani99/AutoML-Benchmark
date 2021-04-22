from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from utils.usefull_functions import get_target


def prepare_and_test(X, y, task):
  if task == 'classification':
    model =  TPOTClassifier(generations=5, cv=5, max_time_mins=1, random_state=1, verbosity=2, n_jobs=-1)
    le = LabelEncoder()
    score = lambda t, p: (accuracy_score(t, p), f1_score(le.fit_transform(t), le.fit_transform(p)))
  else:
    model =  TPOTRegressor(generations=5, cv=5, max_time_mins=1, random_state=1, verbosity=2, n_jobs=-1)
    score = lambda t, p: (np.sqrt(mean_squared_error(t, p)), r2_score(t, p))

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  return (score(y_test, preds))


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
