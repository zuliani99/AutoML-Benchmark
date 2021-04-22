import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.pyplot as plt
import numpy as np

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



def hist(data, task):
    #data.plot.bar()

    labels = data['dataset'].to_numpy()
    x = np.arange(len(labels))  # the label locations
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(20, 8))

    if(task.split(' ')[2] == 'Classificazione'):
        score = '-acc'
        ylabel = 'Accuracy'
    else:
        score = '-rmse'
        ylabel = 'RMSE'

    rects1 = ax.bar(x, data['autosklearn' + score].to_numpy(), width=bar_width, label='AutoSklearn')
    rects2 = ax.bar(x + bar_width, data['tpot' + score].to_numpy(), width=bar_width, label='TPOT')
    rects3 = ax.bar(x + bar_width*2, data['h2o' + score].to_numpy(), width=bar_width, label='H2O')
    rects4 = ax.bar(x + bar_width*3, data['autokeras' + score].to_numpy(), width=bar_width, label='AutoKeras')
    rects5 = ax.bar(x + bar_width*4, data['autogluon' + score].to_numpy(), width=bar_width, label='AutoGluon')

    # Add some text for labels, title and custom x-axis tick labels, etc.

   
    ax.set_title('Risultati per ' + task)
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    fig.tight_layout()
    plt.show()


    if score == '-acc':
        score = '-f1'
        ylabel = 'f1_score'
    else:
        score = '-r2'
        ylabel = 'r2_score'

    fig2, ax2 = plt.subplots(figsize=(20, 8))
    rects1 = ax2.bar(x, data['autosklearn' + score].to_numpy(), width=bar_width, label='AutoSklearn')
    rects2 = ax2.bar(x + bar_width, data['tpot' + score].to_numpy(), width=bar_width, label='TPOT')
    rects3 = ax2.bar(x + bar_width*2, data['h2o' + score].to_numpy(), width=bar_width, label='H2O')
    rects4 = ax2.bar(x + bar_width*3, data['autokeras' + score].to_numpy(), width=bar_width, label='AutoKeras')
    rects5 = ax2.bar(x + bar_width*4, data['autogluon' + score].to_numpy(), width=bar_width, label='AutoGluon')

    # Add some text for labels, title and custom x-axis tick labels, etc.


    ax2.set_title('Risultati per ' + task)
    ax2.set_xticks(x)
    ax2.set_ylabel(ylabel)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.bar_label(rects1, padding=3)
    ax2.bar_label(rects2, padding=3)
    ax2.bar_label(rects3, padding=3)
    ax2.bar_label(rects4, padding=3)
    ax2.bar_label(rects5, padding=3)
    fig2.tight_layout()
    plt.show()