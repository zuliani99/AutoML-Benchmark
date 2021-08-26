# Import needed
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
import time


def mljar(df, task, options, time_start):
  print(colored("--------------------------------- MLJAR ---------------------------------", "cyan"))
  try:
    return do_mljar(df, options, task, time_start)
  except Exception as e:
    # In case of exception
    print(colored('Error: ' + str(e), 'red'))
    print(colored("--------------------------------- MLJAR ---------------------------------\n\n", "cyan"))
    # Return of None with the exception placed on the pipeline
    return None, None, 'Error: ' + str(e), None


def do_mljar(df, options, task, time_start):
  df_new = copy.copy(df) # Deep copy of the DataFrame passed to parameter
  df_new = fill_and_to_category(df_new) # Initial cleaning of the DataFrame
  #df_new = get_list_single_df(df_new) # Initial cleaning of the DataFrame

  pd.options.mode.chained_assignment = None

  X, y = return_X_y(df_new) # Obtain the two DataFrame X and y needed to execute the train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  # Specifying the ml task
  if task == 'classification':
    if len(np.unique(y)) > 2: ml_task = 'multiclass_classification'
    else: ml_task = 'binary_classification'
  else: ml_task=task

  automl = AutoML(mode="Compete", total_time_limit=60*options['time'], ml_task=ml_task)
  
  automl.fit(X_train, y_train)
  y_pred = automl.predict_all(X_test)
 
  dirs = get_dirs()
  pipelines = make_pipeline_mljar(pd.read_csv('AutoML_1/leaderboard.csv'), dirs) # Pipelines
  shutil.rmtree('./AutoML_1') # Deleting the folder created for saving the models tested by MLJAR

  print(colored("--------------------------------- MLJAR ---------------------------------\n\n", "cyan"))

  time_elapsed = round((time.time() - time_start)/60, 3) # Time consumed for computation

  if task != 'classification':
    return (round(np.sqrt(mean_squared_error(y_test, y_pred)), 3), round(r2_score(y_test, y_pred), 3), pipelines, time_elapsed)

  # Check if it is a binary or multilables case
  if len(np.unique(y)) > 2:
    return (round(accuracy_score(y_test, y_pred['label']), 3), round(f1_score(y_test, y_pred['label'], average='weighted'), 3), pipelines, time_elapsed)
  return (round(accuracy_score(y_test, y_pred['label']), 3), round(f1_score(y_test, y_pred['label'], pos_label=np.unique(y)[0]), 3), pipelines, time_elapsed)


# Function for reading the README.md file related to one of the created pipelines
def read_md(path):
  with open(path, 'r') as file:
    data = file.read()
  return data

# Function for creating the pipeline string
def make_pipeline_mljar(pipelines, dirs):
  for index, row in pipelines.iterrows():
    #if(row['name'] not in ['Ensemble', 'Ensamble_Staked', 'folds']):
    if(row['name'].split('_')[-1] != 'Staked' and row['name'] != 'folds' and row['name'] != 'Ensemble'):
      md = re.split('## ',dirs.get(row['name']))

      model_parameters = re.sub('[*-]', '', md[1])
      model_parameters = re.sub('[\n]', ',', model_parameters).replace('n_jobs: ,1', 'n_jobs: -1')

      validation_parameters = re.sub('[*\n]', '', md[2])
      validation_parameters = re.sub('[-]', ',', validation_parameters).replace('Validation , ', '')

      print(validation_parameters)

      pipelines.at[index, 'model_parameters'] = model_parameters
      pipelines.at[index, 'validation_parameters'] = validation_parameters

  ens = ''
  for dict in ['Ensemble', 'Ensamble_Staked']:
    if dirs.get(dict) is not None:
      md = re.split('## Ensemble structure|## Confusion matrix|## Learning curves',dirs.get(dict))
      ens += '\n\n## ' + dict + ' Structure\n' + md[1]

  return str(pipelines.to_markdown() + ens)

# Function to get the list of folders each of which represents a tested opipeline
def get_dirs():
  path = './AutoML_1'
  directory_contents = os.listdir(path)
  return {
    item: read_md(path + '/' + item + '/README.md')
    for item in directory_contents if os.path.isfile(path + '/' + item + '/README.md')
  }
