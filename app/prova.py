# Import needed
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH
import dash_bootstrap_components as dbc
import re
import pandas as pd
import os

def read_md(path):
  with open(path, 'r') as file:
    data = file.read()
  return data

def get_dirs():
  path = './AutoML_1'
  directory_contents = os.listdir(path)
  return {
      item: read_md(path + '/' + item + '/README.md')
      for item in directory_contents if os.path.isdir(path + '/' + item)
  }

# Function for creating the pipeline string
def make_pipeline_mljar(pipelines, dirs):
  for index, row in pipelines.iterrows():
    if(row['name'] != 'Ensemble'):
      md = re.split('## ',dirs.get(row['name']))

      model_parameters = re.sub('[*-]', '', md[1])
      model_parameters = re.sub('[\n]', ',', model_parameters).replace('n_jobs: ,1', 'n_jobs: -1')

      validation_parameters = re.sub('[*\n]', '', md[2])
      validation_parameters = re.sub('[-]', ',', validation_parameters).replace('Validation , ', '')

      print(validation_parameters)

      pipelines.at[index, 'model_parameters'] = model_parameters
      pipelines.at[index, 'validation_parameters'] = validation_parameters
          
  md = re.split('## Ensemble structure|### Metric details|## Confusion matrix',dirs.get('Ensemble'))

  return str(pipelines.to_markdown()), (md[1]), (md[2])

def start():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

    contest = make_pipeline_mljar(pd.read_csv('AutoML_1/leaderboard.csv'), get_dirs())

    app.layout = html.Div([
      dbc.Table(dcc.Markdown(contest[0])),
      dbc.Table(dcc.Markdown(contest[1])),
      dbc.Table(dcc.Markdown(contest[2]))
    ])

    app.run_server(debug=True)
    

if __name__ == '__main__':
    start()