# Import necessari
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import autosklearn.classification
import autosklearn.regression
import pandas as pd
import numpy as np
from utils.usefull_functions import return_X_y, fill_and_to_category
import copy
from termcolor import colored


def make_classification(X_train, X_test, y_train, y_test, timelife, y):
  # Modello Classificazione
  automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=timelife*60,
          per_run_time_limit=30,
          memory_limit=8192,
          n_jobs=-1
    )
  automl.fit(X_train, y_train)
  y_pred = automl.predict(X_test)

  pipelines = ((pd.DataFrame(pd.Series(automl.show_models()))).to_markdown()) # Pipeline

  print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
  # Controllo se si tratta di un caso binario o multilables
  if len(np.unique(y)) > 2:
    return (round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, average='weighted'), 3), pipelines, timelife)
  else:
    return (round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), pipelines, timelife)


def make_regression(X_train, X_test, y_train, y_test, timelife):
  # Modello Regressione
  automl = autosklearn.regression.AutoSklearnRegressor(
          time_left_for_this_task=timelife*60,
          per_run_time_limit=30,
          memory_limit=8192,
          n_jobs=-1
    )
  automl.fit(X_train, y_train)
  y_pred = automl.predict(X_test)
  pipelines = ((pd.DataFrame(pd.Series(automl.show_models()))).to_markdown())
  print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
  return (round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines, timelife)


def auto_sklearn(df, task, options):
  print("--------------------------------AUTOSKLEARN--------------------------------")
  try:
    df_new = copy.copy(df) # Copia profonda del DataFrame passato a paramentro 
    pd.options.mode.chained_assignment = None

    df_new = fill_and_to_category(df_new) # Pulizia iniziale del DataFrame
    X, y = return_X_y(df_new) # Ottenimento dei due DataFrame X ed y pnecessari per eseguire il train_test_split


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    if(task == 'classification'):
      return make_classification(X_train, X_test, y_train, y_test, options['time'], y)
    else:
      return make_regression(X_train, X_test, y_train, y_test, options['time'])

  except Exception as e:
    # In caso di eccezione
    print(colored('Error: ' + str(e), 'red'))
    if (str(e) ==
        'No valid model found in run history. This means smac was not able to fit a valid model. Please check the log file for errors.'
        ) and options['rerun'] == True:
      # Se l'eccezione è provocata del poco tempo messo a disposizione dell'utente ma esso ha spuntato la checkbox per la riesecuzione dall'algoritmo si va a rieseguirlo con un tempo maggiore
      return auto_sklearn(df, task, {'time': options['time']+1, 'rerun': options['rerun']})
    # Altrimenti si ritornano dei None con l'eccezione posto sulla pipeline 
    print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
    return (None, None, 'Error: ' + str(e), None)
    
    