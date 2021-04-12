from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import autosklearn.classification
import autosklearn.regression
import pandas as pd
import numpy as np

def auto_sklearn(df, task):
  #categorical, binary, nuymerical features
  for col in df.columns:
    t = pd.api.types.infer_dtype(df[col])
    if t == "string" or t == 'object':
      df[col] = df[col].astype('category')

  y = df.iloc[:, -1].to_frame()
  X = df.iloc[:, :-1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if(task == 'classification'):
    automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=1*60,
          per_run_time_limit=30,
          n_jobs=-1
    )
    score = lambda t, p: accuracy_score(t, p)
  else:
    automl = autosklearn.regression.AutoSklearnRegressor(
      time_left_for_this_task=1*60,
      per_run_time_limit=30,
      n_jobs=-1
    )
    score = lambda t, p: np.sqrt(mean_squared_error(t, p))
  automl.fit(X_train, y_train)
  #print(automl.sprint_statistics())
  #print(automl.show_models())
  y_pred = automl.predict(X_test)
  return (score(y_test, y_pred))

