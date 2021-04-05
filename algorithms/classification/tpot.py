#pip3 install deap update_checker tqdm stopit xgboost
#pip3 install dask[delayed] dask[dataframe] dask-ml fsspec>=0.3.3 distributed>=2.10.0
#pip3 install scikit-mdr skrebate
#pip3 install tpot


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

def prepare_and_test(X, y):
  model =  TPOTClassifier(generations=5, cv=5, max_time_mins=1, random_state=1, verbosity=2)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  model.fit(X_train, y_train) 
  return (model.score(X_test, y_test))


#devo fare datacleaning: pulizia nel senso nan -> fill_nan
def tpot_class(df):

  for col in df.columns:
    t = pd.api.types.infer_dtype(df[col])
    if t == "string" or t == 'object':
      df[col] = df[col].astype('category').cat.codes

  print(df.info())
  print(df.head())
  
  y = df.iloc[:, -1]
  X = df.iloc[:, :-1]

  #y = y.to_frame()

  #y[y.columns[0]] = y[y.columns[0]].cat.codes

  #print(X.info())
  #print(y.info())

  return prepare_and_test(X, y)