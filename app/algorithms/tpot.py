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

le = LabelEncoder()

def make_classification(X_train, X_test, y_train, y_test, timelife, y):
  # Modello Calssificazione
  model =  TPOTClassifier(generations=3, cv=5, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=1, max_eval_time_mins=0.05) #, subsample=0.5
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  pipelines = model.export() # Ottengo la pipeline

  print("-----------------------------------TPOT------------------------------------\n\n")
  # Controllo se si tratta di un caso binario o multilables
  if len(np.unique(y)) > 2:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, average='weighted'), 3), pipelines, timelife
  else:
    return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), pipelines, timelife

def make_regression(X_train, X_test, y_train, y_test, timelife):
  # Modello Regressione
  model =  TPOTRegressor(generations=3, cv=5, max_time_mins=timelife, random_state=1, verbosity=2, n_jobs=1, max_eval_time_mins=0.05) #, subsample=0.5
  print(type(X_train), type(y_train))
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  pipelines = model.export() # Ottengo la pipeline
  print("-----------------------------------TPOT------------------------------------\n\n")
  return round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines, timelife


def TPOT(df, task, options):
  print("-----------------------------------TPOT------------------------------------")
  try:
    df_new = copy.copy(df) # Copia profonda del DataFrame passato a paramentro 
    pd.options.mode.chained_assignment = None

    df_new = fill_and_to_category(df_new) # Pulizia iniziale del DataFrame
    X, y = return_X_y(df_new) # Ottenimento dei due DataFrame X ed y pnecessari per eseguire il train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    if task == 'classification':
      return make_classification(X_train, X_test, y_train, y_test, options['time'], y)
    else:
      return make_regression(X_train, X_test, y_train, y_test, options['time'])

  except Exception as e:
    # In caso di eccezione
    print(colored('Error: ' + str(e), 'red'))
    if (str(e) == 'There was an error in the TPOT optimization process. This could be because the data was not formatted properly, or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly. If you enabled PyTorch estimators, please check the data requirements in the online documentation: https://epistasislab.github.io/tpot/using/' 
      and options['rerun'] == True):
      # Se l'eccezione è provocata del poco tempo messo a disposizione dall'utente ma esso ha spuntato la checkbox per la riesecuzione dell'algoritmo si va a rieseguirlo con un tempo maggiore
      return TPOT(df, task, {'time': options['time']+5, 'rerun': options['rerun']})
    # Altrimenti si ritornano dei None con l'eccezione posto sulla pipeline 
    print("-----------------------------------TPOT------------------------------------\n\n")
    return (None, None, 'Error: ' + str(e), None)

