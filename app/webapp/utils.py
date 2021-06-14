import os
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import pandas as pd


def get_lisd_dir(test):
    lis = (os.listdir('./results/'+test))
    dropdown = []
    for l in lis:
        if l != '.gitignore':
            dropdown.append({'label': l, 'value': l})
    return dropdown



def retrun_graph_table(dfs, title, t):
    scatters = []
    histos = []
    table = [html.H3(title)]
    for df in dfs:
        table.append(dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True))
        for col in df.columns[1:]:
            scatters.append(go.Scatter(x=df['dataset'], y=df[col], name=col.split('-')[0], mode='lines+markers'))
            histos.append(go.Bar(x=df['dataset'], y=df[col], name=col.split('-')[0]))
        

    table.append(
        dbc.Tabs(
            [
                dbc.Tab(label="Scatter", tab_id="scatter"),
                dbc.Tab(label="Histograms", tab_id="histogram"),
            ],
            id="tabs-"+t,
            active_tab="scatter",
        )
    )
    #table.append(html.Div(id='result-past-bench-openml-graph-'+type))

    # acc f1 acc f1
    return scatters[:5], scatters[5:], histos[:5], histos[5:], table

def get_store_and_tables(dfs):
    store_dict_class = {'scatter_class_acc': None, 'histo_class_acc': None, 'scatter_class_f1': None, 'histo_class_f1': None}
    store_dict_reg = {'scatter_reg_rmse': None, 'histo_reg_rmse': None, 'scatter_reg_r2': None, 'histo_reg_r2': None}
    tables = [[None], [None]]
    if dfs[0] is not None:
        #tables_graphs.append(retrun_graph_table(dfs[:2], 'Risultati Classificazione', ['Accuracy', 'F1-score']))
        res = retrun_graph_table(dfs[:2], 'Risultati Classificazione', 'class')
        store_dict_class['scatter_class_acc'] = res[0]
        store_dict_class['histo_class_acc'] = res[2]
        store_dict_class['scatter_class_f1'] = res[1]
        store_dict_class['histo_class_f1'] = res[3]
        tables[0] = res[4]
    else:
        tables[0].append(
            dbc.Tabs(
                [],
                id="tabs-class",
                active_tab="",
                style={'hidden':'true'}
            )
        )
    if dfs[2] is not None:
        #tables_graphs.append(retrun_graph_table(dfs[2:], 'Risultati Regressione', ['RMSE', 'R2-score']))
        res = retrun_graph_table(dfs[2:], 'Risultati Regressione', 'reg')
        store_dict_reg['scatter_reg_rmse'] = res[0]
        store_dict_reg['histo_reg_rmse'] = res[2]
        store_dict_reg['scatter_reg_r2'] = res[1]
        store_dict_reg['histo_reg_r2'] = res[3]
        tables[1] = res[4]
    else:
        tables[1].append(
            dbc.Tabs(
                [],
                id="tabs-reg",
                active_tab="",
                style={'hidden':'true'}
            )
        )
    
    #print(store_dict_class)

    return store_dict_class, store_dict_reg, tables[0], tables[1]


def get_store_past_bech_function(timestamp, type):
    if timestamp is not None:
        dfs = []
        scores = [('classification','acc'), ('classification','f1_score'), ('regression','rmse'), ('regression','r2_score')]
        for score in scores:
            print('./results/'+ type +'/'+timestamp+'/'+ score[0] +'/'+ score[1] +'.csv')
            if os.path.exists('./results/'+ type +'/'+timestamp+'/'+ score[0] +'/'+ score[1] +'.csv'):
                dfs.append(pd.read_csv('./results/'+ type +'/'+timestamp+'/'+ score[0] +'/'+ score[1] +'.csv'))
            else:
                dfs.append(None)
        #print(dfs)
        return get_store_and_tables(dfs)
    else:
        raise PreventUpdate



def render_tab_content(active_tab, data, type): #pathname
    #render = {'openml': None, 'kaggle': None, 'results-openml': None, 'results-kaggle': None}
    if active_tab and data is not None:
        if active_tab == "scatter":
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[0]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[0].split('_')[1]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[1]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[1].split('_')[1]))))),
                                ], align="center"
                            )
                        )]
        else: #active_tab == "histogram":
            #print(data['histo_'+type[0]])
            #print(data['histo_'+type[1]])
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[0]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[0].split('_')[1]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[1]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[1].split('_')[1]))))),
                                ], align="center"
                            )
                        )]
        #render[pathname] = ret
        #return render
    return "No tab selected"