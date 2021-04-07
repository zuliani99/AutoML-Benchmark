#pip3 install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
#sudo apt install default-jre

import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
h2o.init()

def prepare_and_test(X, y):
  x = X.columns
  target_str = y.columns[0]
  X[target_str] = y[target_str]
  aml = H2OAutoML(max_runtime_secs=1*60, nfolds=15, max_models=20, seed=1)
  aml.train(x, target_str, training_frame=X)
  lb = aml.leaderboard
  lb.head(rows=lb.nrows)
  print(h2o.automl.get_leaderboard(aml, extra_columns = 'ALL'))
  return (aml.predict(y))

def h2o_class(df):
  y = df.iloc[:, -1:]
  X = df.iloc[:, 0:df.shape[1]-1]
  X = h2o.H2OFrame(X)
  y = h2o.H2OFrame(y)
  return(prepare_and_test(X, y))