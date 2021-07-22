# Import necessari
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from utils.usefull_functions import return_X_y, fill_and_to_category
import copy
from termcolor import colored

from dask.distributed import Client
import joblib
import shutil

def make_classification(X_train, X_test, y_train, y_test, timelife, y, client):
  le = LabelEncoder()

  # Modello Calssificazione
  model =  TPOTClassifier(generations=100, cv=10, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=-1, max_eval_time_mins=5)
  #model.fit(X_train, y_train)

  with joblib.parallel_backend("dask"):
    model.fit(np.array(X_train), np.array(y_train).ravel())

  y_pred = model.predict(X_test)
  pipelines = model.export() # Ottengo la pipeline

  print("-----------------------------------TPOT------------------------------------\n\n")
  client.close()
  shutil.rmtree('./dask-worker-space')

  # Controllo se si tratta di un caso binario o multilables
  if len(np.unique(y)) > 2:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, average='weighted'), 3), pipelines, timelife
  else:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), pipelines, timelife


def make_regression(X_train, X_test, y_train, y_test, timelife, client):
  le = LabelEncoder()

  # Modello Regressione
  model =  TPOTRegressor(generations=100, cv=10, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=-1, max_eval_time_mins=5)
  #model.fit(X_train, y_train)

  with joblib.parallel_backend("dask"):
    model.fit(np.array(X_train), np.array(y_train).ravel())

  y_pred = model.predict(X_test)
  pipelines = model.export() # Ottengo la pipeline

  print("-----------------------------------TPOT------------------------------------\n\n")
  client.close()
  shutil.rmtree('./dask-worker-space')

  return round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines, timelife


def TPOT(df, task, options):
  print("-----------------------------------TPOT------------------------------------")
  try:
    client = Client(processes=False)
    df_new = copy.copy(df) # Copia profonda del DataFrame passato a paramentro 
    pd.options.mode.chained_assignment = None

    df_new = fill_and_to_category(df_new) # Pulizia iniziale del DataFrame
    X, y = return_X_y(df_new) # Ottenimento dei due DataFrame X ed y pnecessari per eseguire il train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    if task == 'classification':
      return make_classification(X_train, X_test, y_train, y_test, options['time'], y, client)
    else:
      return make_regression(X_train, X_test, y_train, y_test, options['time'], client)

  except Exception as e:
    # In caso di eccezione
    print(colored('Error: ' + str(e), 'red'))
    if str(e) == 'There was an error in the TPOT optimization process. This could be because the data was not formatted properly, or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly. If you enabled PyTorch estimators, please check the data requirements in the online documentation: https://epistasislab.github.io/tpot/using/':
      if options['rerun'] == True:
        # Se l'eccezione è provocata del poco tempo messo a disposizione dall'utente ma esso ha spuntato la checkbox per la riesecuzione dell'algoritmo si va a rieseguirlo con un tempo maggiore
        return TPOT(df, task, {'time': options['time']+5, 'rerun': options['rerun']})
      print("-----------------------------------TPOT------------------------------------\n\n")
      return (None, None, 'Expected Error duo to short algorithm timelife: ' + str(e), None)

    # Altrimenti si ritornano dei None con l'eccezione posto sulla pipeline 
    print("-----------------------------------TPOT------------------------------------\n\n")
    return (None, None, 'Unexpected Error: ' + str(e), None)

