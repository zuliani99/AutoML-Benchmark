import pandas as pd
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
from utils.usefull_functions import return_X_y, fill_and_to_category
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder 
import shutil


def prepare_and_test(X, y, task, timelife):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if isinstance(y_train, pd.Series):
    y_train = y_train.to_frame()

  if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame() 

  if(task == 'classification'):
    clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)#attenzione salvo sul cestino
  else:
    clf = ak.StructuredDataRegressor(overwrite=True, max_trials=3)#attenzione salvo sul cestino
    
  clf.fit(X_train, y_train, validation_split=0.15, epochs=timelife)

  model = clf.export_model()

  model.summary()

  #shutil.rmtree('./structured_data_classifier/')

  summary = []
  model.summary(print_fn=lambda x: summary.append(x))
  model_summary = '\n'.join(summary)

  y_pred = clf.predict(X_test)


  if task == 'classification':
    shutil.rmtree('./structured_data_classifier')
    le = LabelEncoder() # forse è meglio che tolga il tutto relativo al label encoder
    y_test = le.fit_transform(y_test)
    y_pred = le.fit_transform(y_pred)
    if len(np.unique(y)) > 2:
      print('multiclass')
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), model_summary)
    else:
      print('binary')
      return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), model_summary)
  else:
    shutil.rmtree('./structured_data_regressor')
    return (np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred), model_summary)


def autokeras(df, task, timelife):
  pd.options.mode.chained_assignment = None
  X, y, _ = return_X_y(df)
  if not isinstance(df, pd.DataFrame):
    #X = X.apply(LabelEncoder().fit_transform)
    X = fill_and_to_category(X)
  return (prepare_and_test(X, y, task, timelife))