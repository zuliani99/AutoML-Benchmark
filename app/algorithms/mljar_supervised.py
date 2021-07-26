import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from utils.usefull_functions import return_X_y, fill_and_to_category
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import shutil
import copy
from termcolor import colored
import os
import re

def read_md(path):
  with open(path, 'r') as file:
    data = file.read().replace('\n', '')
  return data

def mljar(df, task, options):
  print("---------------------------------MLJAR---------------------------------" + task)
  try:
    df_new = copy.copy(df) # Deep copy of the DataFrame passed to parameter
    df_new = fill_and_to_category(df_new) # Initial cleaning of the DataFrame
    pd.options.mode.chained_assignment = None
    X, y = return_X_y(df_new) # Obtain the two DataFrame X and y needed to execute the train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    automl = AutoML(mode="Compete", total_time_limit=60*options['time'])
    automl.fit(X_train, y_train)
    y_pred = automl.predict_all(X_test)


    path = './AutoML_1'
    directory_contents = os.listdir(path)
    dirs = {
        item: read_md(path + '/' + item + '/README.md')
        for item in directory_contents if os.path.isdir(path + '/' + item)
    }
    
    pipelines = pd.read_csv('AutoML_1/leaderboard.csv')

    for index, row in pipelines.iterrows():
      md = re.split(row['name']+'|## Validation -|## Optimized metric',dirs.get(row['name']))

      model_parameters = re.sub('[*]', '', md[1].split('##')[1])
      model_parameters = re.sub('[-]', ',', model_parameters).replace('n_jobs: ,1', 'n_jobs: -1')

      validation_parameters = re.sub('[*]', '', md[2])
      validation_parameters = re.sub('[-]', ',', validation_parameters)

      pipelines.at[index, 'model_parameters'] = model_parameters
      pipelines.at[index, 'validation_parameters'] = validation_parameters

    shutil.rmtree('./AutoML_1') 

    if task != 'classification':
      return (round(np.sqrt(mean_squared_error(y_test, y_pred['label'])), 3), round(r2_score(y_test, y_pred['label']), 3), pipelines.to_string(index = False), options['time'])
    # Check if it is a binary or multilables case
    if len(np.unique(y)) > 2:
      return (round(accuracy_score(y_test, y_pred['label']), 3), round(f1_score(y_test, y_pred['label'], average='weighted'), 3), pipelines.to_string(index = False), options['time'])
    return (round(accuracy_score(y_test, y_pred['label']), 3), round(f1_score(y_test, y_pred['label'], pos_label=np.unique(y)[0]), 3), pipelines.to_string(index = False), options['time'])


  except Exception as e:
    print(colored('Error: ' + str(e), 'red'))
    print(
        '---------------------------------MLJAR---------------------------------\n\n'
    )
    return None, None, 'Error: ' + str(e), None