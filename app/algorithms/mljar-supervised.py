from datetime import time
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from utils.usefull_functions import return_X_y, fill_and_to_category
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import shutil
import copy
from termcolor import colored



def mljar(df, task, options):
  print("---------------------------------AUTOKERAS---------------------------------")
  try:
    df_new = copy.copy(df) # Copia profonda del DataFrame passato a paramentro 
    df_new = fill_and_to_category(df_new) # Pulizia iniziale del DataFrame
    pd.options.mode.chained_assignment = None
    X, y = return_X_y(df_new) # Ottenimento dei due DataFrame X ed y pnecessari per eseguire il train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    automl = AutoML(total_time_limit=60*options['time'])
    automl.fit(X_train, y_train)
    y_pred = automl.predict_all(X_test)

    pipelines = pd.read_csv('AutoML_1/leaderboard.csv')
    shutil.rmtree('./AutoML_1') 

    if task != 'classification':
      return round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines.to_string(index = False), options['time']

    if len(np.unique(y)) > 2:
      return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, average='weighted'), 3), pipelines.to_string(index = False), options['time']
    else:
      return round(accuracy_score(y_test, y_pred), 3), round(f1_score(y_test, y_pred, pos_label=np.unique(y)[0]), 3), pipelines.to_string(index = False), options['time']

  except Exception as e:
    # In caso di eccezione
    print(colored('Error: ' + str(e), 'red'))
    print("---------------------------------AUTOKERAS---------------------------------\n\n")
    # Ritorno dei None con l'eccezione posto sulla pipeline 
    return (None, None, 'Error: ' + str(e), None)