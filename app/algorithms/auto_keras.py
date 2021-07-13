# Import necessari
import pandas as pd
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
from utils.usefull_functions import return_X_y, fill_and_to_category
import sklearn
import shutil
from tensorflow.keras import backend as K
import copy
from termcolor import colored

# Funzioni d'appoggio per il calcolo del f1_score e del r2_score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Funzione per la definizone del modello automl da effettuare a seconda del task
def get_automl(task):
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
  return clf, custom_obj

# Funzion per l'ottenimento della pipeline
def get_summary(model):
  table=pd.DataFrame(columns=["Name","Type","Shape"])
  for layer in model.layers:
    print(layer)
    table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)
  return (table.to_markdown())

# Caso di classificazione
def get_classification(y_test, y_pred, model_summary, y, timelife):
  shutil.rmtree('./structured_data_classifier') # Eliminazione della cartella creare per il salvataggio dei modelli testati da AutoKeras
  print("---------------------------------AUTOKERAS---------------------------------\n\n")
  # Controllo se si tratta di un caso binario o multilables
  if len(np.unique(y)) > 2:
    return round(sklearn.metrics.accuracy_score(y_test, y_pred), 3), round(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'), 3), model_summary, timelife
  else:
    return round(sklearn.metrics.accuracy_score(y_test, y_pred), 3), round(sklearn.metrics.f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), model_summary, timelife

# Caso di regressione
def get_regression(y_test, y_pred, model_summary, timelife):
  shutil.rmtree('./structured_data_regressor') # Eliminazione della cartella creare per il salvataggio dei modelli testati da AutoKeras
  print("---------------------------------AUTOKERAS---------------------------------\n\n")
  return round(np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)), 3), round(sklearn.metrics.r2_score(y_test, y_pred), 3), model_summary, timelife


def prepare_and_test(X, y, task, timelife):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  if isinstance(y_train, pd.Series):
    y_train = y_train.to_frame()

  if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame() 


  clf, custom_obj = get_automl(task) # Definizone del modello a seconda del task dato con il custom_obj che è uguale al secondo score che si vuole tenere in considerazione 
  clf.fit(x=X_train, y=y_train, validation_split=0.15, epochs=timelife)
  model = clf.export_model(custom_objects=custom_obj)
  model.summary()
  model_summary = get_summary(model) # Pipeline
  y_pred = clf.predict(X_test,custom_objects=custom_obj)
  
  y_test = np.array(y_test, dtype = np.int32)
  y_pred = np.array(y_pred, dtype = np.int32)

  if task == 'classification':
    return get_classification(y_test, y_pred, model_summary, y, timelife)
  else:
    return get_regression(y_test, y_pred, model_summary, timelife)


def autokeras(df, task, options):
  print("---------------------------------AUTOKERAS---------------------------------")
  try:
    df_new = copy.copy(df) # Copia profonda del DataFrame passato a paramentro 
    df_new = fill_and_to_category(df_new) # Pulizia iniziale del DataFrame
    pd.options.mode.chained_assignment = None
    X, y = return_X_y(df_new) # Ottenimento dei due DataFrame X ed y pnecessari per eseguire il train_test_split
    return (prepare_and_test(X, y, task, options['time']))

  except Exception as e:
    # In caso di eccezione
    print(colored('Error: ' + str(e), 'red'))
    print("---------------------------------AUTOKERAS---------------------------------\n\n")
    # Ritorno dei None con l'eccezione posto sulla pipeline 
    return (None, None, 'Error: ' + str(e), None)