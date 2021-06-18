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



def retrun_graph_table(dfs, title, task, t, opts):
    scatters = []
    histos = []
    table = [html.H3(title)]
    for df in dfs:
        table.append(dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True))
        for col in df.columns[1:]:
            print(col)
            scatters.append(go.Scatter(x=df['dataset'], y=df[col], name=col.split('-')[0], mode='lines+markers'))
            histos.append(go.Bar(x=df['dataset'], y=df[col], name=col.split('-')[0]))
        

    table.append(
        dbc.Tabs(
            [
                dbc.Tab(label="Histograms", tab_id="histogram"),
                dbc.Tab(label="Scatter", tab_id="scatter"),
                dbc.Tab(label="Algorithm Options", tab_id="algo-options"),
            ],
            id="tabs-"+task,
            active_tab="histogram",
        )
    )

    opts = opts.to_dict()
    options = [
        html.Div([
            html.P(["Running time for Autosklearn: " + str(opts['autosklearn'][0]) + " minute/s"]),
            html.P(["Running time for TPOT: " + str(opts['tpot'][0]) + " generation/s"]),
            html.P(["Running time for H2O: " + str(opts['h2o'][0]) + " minute/s"]),
            html.P(["Running time for AutoKeras: " + str(opts['autokeras'][0]) + " epoch/s"]),
            html.P(["TRunning time for AutoGluon: " + str(opts['autogluon'][0]) + " minute/s"]),
        ])
    ]

    #table.append(html.Div(id='result-past-bench-openml-graph-'+type))

    # acc f1 acc f1 / rmse r2 rmse r2
    #return scatters[:5], scatters[5:], histos[:5], histos[5:], table if t == 'OpenML' else scatters[:6], scatters[5:], histos[:6], histos[5:], table
    #print(scatters)
    #print(histos)
    if(t == 'Kaggle'): 
        return scatters[:6], scatters[6:], histos[:6], histos[6:], table, options
    else:
        return scatters[:5], scatters[5:], histos[:5], histos[5:], table, options 

def get_store_and_tables(dfs, type):
    print(dfs)
    store_dict_class = {'scatter_class_acc': None, 'histo_class_acc': None, 'scatter_class_f1': None, 'histo_class_f1': None, 'options_class': None}
    store_dict_reg = {'scatter_reg_rmse': None, 'histo_reg_rmse': None, 'scatter_reg_r2': None, 'histo_reg_r2': None, 'options_reg': None}
    tables = [[None], [None]]
    if dfs[0] is not None:
        #tables_graphs.append(retrun_graph_table(dfs[:2], 'Risultati Classificazione', ['Accuracy', 'F1-score']))
        res = retrun_graph_table(dfs[:2], 'Classification Results', 'class', type, dfs[4])
        store_dict_class['scatter_class_acc'] = res[0]
        store_dict_class['histo_class_acc'] = res[2]
        store_dict_class['scatter_class_f1'] = res[1]
        store_dict_class['histo_class_f1'] = res[3]
        store_dict_class['options_class'] = res[5]
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
        res = retrun_graph_table(dfs[2:4], 'Regression Results', 'reg', type, dfs[4])
        store_dict_reg['scatter_reg_rmse'] = res[0]
        store_dict_reg['histo_reg_rmse'] = res[2]
        store_dict_reg['scatter_reg_r2'] = res[1]
        store_dict_reg['histo_reg_r2'] = res[3]
        store_dict_reg['options_reg'] = res[5]
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
        dfs.append(pd.read_csv('./results/'+ type +'/'+timestamp+'/options.csv'))
        return get_store_and_tables(dfs, type)
    else:
        raise PreventUpdate



def render_tab_content(active_tab, data, type): #pathname
    #render = {'openml': None, 'kaggle': None, 'results-openml': None, 'results-kaggle': None}
    if active_tab and data is not None:
        if active_tab == "scatter":
            print(data['scatter_'+type[0]])
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[0]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[0].split('_')[1]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[1]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[1].split('_')[1]))))),
                                ], align="center"
                            )
                        )]
        elif active_tab == "histogram":
            print(data['histo_'+type[0]])
            #print(data['histo_'+type[1]])
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[0]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[0].split('_')[1]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[1]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[1].split('_')[1]))))),
                                ], align="center"
                            )
                        )]
        else:
            return [
                data['options_'+type[0].split('_')[0]]
            ]
        #render[pathname] = ret
        #return render
    return "No tab selected"


def create_collapse(algo, measure, min, disabled):
    return dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H2(
                                            dbc.Button(
                                                algo + " Options",
                                                color="link",
                                                id=algo.lower()+"-options",
                                                disabled=disabled
                                            )
                                        )
                                    ),
                                    dbc.Collapse(
                                        dbc.CardBody([
                                            dbc.FormGroup([
                                                dbc.Label("Runnung time in "+measure,  width=5),
                                                dbc.Col([
                                                    dbc.InputGroup([
                                                        dbc.Input( id=algo.lower()+"-timelife", type="number", value=min, placeholder=measure, min=min, max=100000),
                                                        dbc.InputGroupAddon("at least " + str(min), addon_type="prepend")]
                                                    ),
                                                ], width=5),
                                            ],row=True),
                                        ]), id="collapse-"+algo.lower()
                                    ),
                                ]
                            )
def render_collapse_options(choice):
    return {
        'autosklearn': [False, True, True, True, True],
        'h2o': [True, False, True, True, True],
        'tpot': [True, True, False, True, True],
        'autokeras': [True, True, True, False, True],
        'autogluon':[True, True, True, True, False],
        'all': [False, False, False, False, False],
    }.get(choice)