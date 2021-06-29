import pandas as pd
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
from utils.usefull_functions import return_X_y, fill_and_to_category
from sklearn.preprocessing import LabelEncoder 
import sklearn
import shutil
from tensorflow.keras import backend as K
import keras_tuner as kt
import copy



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def prepare_and_test(X, y, task, timelife):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if isinstance(y_train, pd.Series):
    y_train = y_train.to_frame()

  if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame() 

  print(y)

  #print(X_train)
  #print(y_train, 'vediamoooooo: y_train ' + str(len(np.unique(y_train))))
  #print(y_test, 'vediamoooooo: y_test ' + str(len(np.unique(y_test))))

  if(task == 'classification'):
    clf = ak.StructuredDataClassifier(
      overwrite=True,
      max_trials=3,
      metrics=['accuracy', f1_score]
    )
    custom_obj = { 'f1_score': f1_score }
  else:
    clf = ak.StructuredDataRegressor(
      overwrite=True,
      max_trials=3,
      metrics=['mean_squared_error', r2_score]
    )
    custom_obj = { 'r2_score': r2_score }

  clf.fit(X_train, y_train, validation_split=0.15, epochs=timelife)


  model = clf.export_model(custom_objects=custom_obj)
  model.summary()


  summary = []
  model.summary(print_fn=lambda x: summary.append(x))
  model_summary = '\n'.join(summary)

  y_pred = clf.predict(X_test,custom_objects=custom_obj)



  #le = LabelEncoder() # forse è meglio che tolga il tutto relativo al label encoder
  #print('non so cosa sto facendo ', accuracy_score(y_test, y_pred))
  #y_test = le.fit_transform(y_test).to_numpy()
  #y_pred = le.fit_transform(y_pred).to_numpy()

  y_test = y_test.to_numpy()

  print((y_test))
  print((y_pred))

  #print(clf.evaluate(X_test, y_test))

  if task == 'classification':
    shutil.rmtree('./structured_data_classifier')
    if len(np.unique(y)) > 2:
      print('multiclass')
      print(sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
      #print(clf.evaluate(X_test, y_test, custom_objects=custom_obj))
      #return (clf.evaluate(X_test, y_test)[0], f1_score(y_test, y_pred, average='weighted'), model_summary)
      return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.f1_score(y_test, y_pred, average='weighted'), model_summary
    else:
      print('binary')
      #print(clf.evaluate(X_test, y_test))
      #return (clf.evaluate(X_test, y_test)[0], f1_score(y_test, y_pred), model_summary)
      return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.f1_score(y_test, y_pred), model_summary
  else:
    shutil.rmtree('./structured_data_regressor')
    #print(clf.evaluate(X_test, y_test))
    #print(clf.evaluate(X_test, y_test)[0], r2_score(y_test, y_pred))
    #return (clf.evaluate(X_test, y_test)[0], r2_score(y_test, y_pred), model_summary)
    return np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)), sklearn.metrics.r2_score(y_test, y_pred), model_summary


def autokeras(df, task, timelife):
  df_new = copy.copy(df)
  #df_new = fill_and_to_category(df_new)
  pd.options.mode.chained_assignment = None
  X, y, _ = return_X_y(df_new)
  
  if not isinstance(df_new, pd.DataFrame):
    #X = X.apply(LabelEncoder().fit_transform)
    print('                                                                         sono quaaaaaaa') #kaggle
    X = fill_and_to_category(X)
  else:
    print('                                                                         invece quaaaaaaa') # openml test
  return (prepare_and_test(X, y, task, timelife))