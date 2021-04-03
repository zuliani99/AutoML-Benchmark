#pip3 install deap update_checker tqdm stopit xgboost
#pip3 install dask[delayed] dask[dataframe] dask-ml fsspec>=0.3.3 distributed>=2.10.0
#pip3 install scikit-mdr skrebate
#pip3 install tpot


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

def prepare_and_test(X, y):
  model =  TPOTClassifier(generations=5, cv=5, max_time_mins=10, random_state=1, verbosity=2)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  model.fit(X_train, y_train) 
  print(model.score(X_test, y_test))
  model.export('tpot_exported_pipeline.py')



def tpot_class(df):
  y = df.iloc[:, -1:]
  X = df.iloc[:, 0:df.shape[1]-1]
  y[y.columns[0]] = y[y.columns[0]].cat.codes # 0 -> UP, 1 -> DOWN
  prepare_and_test(X, y)