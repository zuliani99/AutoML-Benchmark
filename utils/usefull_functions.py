import plotly.graph_objs as go
import plotly.offline as py
#import matplotlib.pyplot as plt

def get_target(train, test):
    for c in train.columns:
        if c not in test.columns:
            return c


'''
data = pd.read_csv('./results/openml/2021-04-18 15:37:55.986520/regression.csv')
plt.plot('dataset', 'autosklearn', data=data, marker='', color='red', linewidth=2)
plt.plot('dataset', 'tpot', data=data, marker='', color='green', linewidth=2)
plt.plot('dataset', 'h2o', data=data, marker='', color='yellow', linewidth=2)
plt.plot('dataset', 'autokeras', data=data, marker='', color='blue', linewidth=2)
plt.plot('dataset', 'autogluon', data=data, marker='', color='black', linewidth=2)
plt.legend()
plt.show()
'''


'''
from utils.usefull_functions import scatter
import pandas as pd

data = pd.read_csv('./results/openml/2021-04-18 15:37:55.986520/regression.csv')
scatter(data, 'Openml - Regressione')
'''

def scatter(data, task):
    data = [go.Scatter(x=data['dataset'], y=data['autosklearn'], name='AutoSklearn', mode='lines+markers'), 
            go.Scatter(x=data['dataset'], y=data['tpot'], name='TPOT', mode='lines+markers'),
            go.Scatter(x=data['dataset'], y=data['h2o'], name='H2O', mode='lines+markers'),
            go.Scatter(x=data['dataset'], y=data['autokeras'], name='AutoKeras', mode='lines+markers'),
            go.Scatter(x=data['dataset'], y=data['autogluon'], name='AutoGluon', mode='lines+markers')]
    if task.split(' ')[2] == 'Classificazione':
        y = 'Accuracy'
    else:
        y = 'RMSE'
    layout = go.Layout(dict(title = 'Risultati ' + task,
                    xaxis = dict(title = 'Datasets'),
                    yaxis = dict(title = y),
                    ),legend=dict(
                    orientation="v"))
    py.iplot(dict(data=data, layout=layout))