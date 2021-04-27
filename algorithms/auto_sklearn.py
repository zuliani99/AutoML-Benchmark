from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import autosklearn.classification
import autosklearn.regression
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
import numpy as np
from utils.usefull_functions import get_target

def auto_sklearn(df, task):
  #categorical, binary, nuymerical features
  le = LabelEncoder()

  if isinstance(df, pd.DataFrame):
    #df = df.apply(LabelEncoder().fit_transform)

    for col in df.columns:
      t = pd.api.types.infer_dtype(df[col]) # test -> 
      if t == "string" or t == 'object':
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.add_categories('Unknown')
        df[col].fillna('Unknown', inplace =True)
      if t == "integer" or t == "floating":
        df[col] = df[col].fillna(df[col].mean())

    n_target = df['n_target'][0]
    df = df.drop('n_target', axis = 1)

    y = df.iloc[:, -n_target].to_frame()
    X = df.iloc[:, :-n_target]
  else:
    train = df[0]
    test = df[1]
    target = get_target(train, test)
    y = train[target]
    train = train.drop([target], axis=1)
    X = train.apply(LabelEncoder().fit_transform)

  '''for col in df.columns:
    t = pd.api.types.infer_dtype(df[col]) # test -> 
    if t == "string" or t == 'object':
      df[col] = df[col].astype('category')'''

  

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  y_test = le.fit_transform(y_test)

  if(task == 'classification'):
    automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=1*60,
          per_run_time_limit=30,
          n_jobs=-1
    )
    automl.fit(X_train, y_train)
    y_pred = le.fit_transform(automl.predict(X_test))
    if len(np.unique(y_test)) > 2:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'))
    else:
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))
  else:
    automl = autosklearn.regression.AutoSklearnRegressor(
          time_left_for_this_task=1*60,
          per_run_time_limit=30,
          n_jobs=-1
    )
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    return (np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred))

  #print(automl.sprint_statistics())
  #print(automl.show_models())




'''for col in train.columns:
    t = pd.api.types.infer_dtype(train[col])
    if t == "string" or t == 'object':
      train[col] = train[col].astype('category')
      train[col] = train[col].cat.add_categories('Unknown')
      train[col].fillna('Unknown', inplace =True) 
    if t == "integer" or t == "floating":
      train[col] = train[col].fillna(train[col].mean())'''

  