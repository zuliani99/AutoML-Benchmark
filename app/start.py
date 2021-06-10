#!/usr/bin/env python3

from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import pandas as pd
import os


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


def retrun_graph_table(dfs, title, type):
    scatters = []
    histos = []
    table = [html.H3(title)]
    for df in dfs:
        table.append(dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True))
        for col in df.columns[1:]:
            #print(type(df['dataset']))
            scatters.append(go.Scatter(x=df['dataset'], y=df[col], name=col.split('-')[0], mode='lines+markers'))
            histos.append(go.Histogram(x=df['dataset'], y=df[col], name=col.split('-')[0]))

    table.append(
        dbc.Tabs(
            [
                dbc.Tab(label="Scatter", tab_id="scatter"),
                dbc.Tab(label="Histograms", tab_id="histogram"),
            ],
            id="tabs",
            active_tab="scatter",
        )
    )
    #table.append(html.Div(id='result-past-bench-openml-graph-'+type))

    # acc f1 acc f1
    return scatters[:5], scatters[5:], histos[:5], histos[5:], table

def get_store_and_tables(dfs):
    store_dict_class = {'scatter_class_acc': None, 'histo_class_acc': None, 'scatter_class_f1': None, 'histo_class_f1': None}
    store_dict_reg = {'scatter_reg_rmse': None, 'histo_reg_rmse': None, 'scatter_reg_r2': None, 'histo_reg_r2': None}
    tables = [None, None]
    if dfs[0] is not None:
        #tables_graphs.append(retrun_graph_table(dfs[:2], 'Risultati Classificazione', ['Accuracy', 'F1-score']))
        res = retrun_graph_table(dfs[:2], 'Risultati Classificazione', 'class')
        store_dict_class['scatter_class_acc'] = res[0]
        store_dict_class['histo_class_acc'] = res[2]
        store_dict_class['scatter_class_f1'] = res[1]
        store_dict_class['histo_class_f1'] = res[3]
        tables[0] = res[4]
    if dfs[2] is not None:
        #tables_graphs.append(retrun_graph_table(dfs[2:], 'Risultati Regressione', ['RMSE', 'R2-score']))
        res = retrun_graph_table(dfs[2:], 'Risultati Regressione', 'reg')
        store_dict_reg['scatter_reg_rmse'] = res[0]
        store_dict_reg['histo_reg_rmse'] = res[2]
        store_dict_reg['scatter_reg_r2'] = res[1]
        store_dict_reg['histo_reg_r2'] = res[3]
        tables[1] = res[4]
    return store_dict_class, store_dict_reg, tables[0], tables[1]


def get_store_past_bech_openml_function(timestamp, type):
    if timestamp is not None:
        dfs = []
        scores = [('classification','acc'), ('classification','f1_score'), ('regression','rmse'), ('regression','r2_score')]
        for score in scores:
            print('./results/'+ type +'/'+timestamp+'/'+ score[0] +'/'+ score[1] +'.csv')
            if os.path.exists('./results/'+ type +'/'+timestamp+'/'+ score[0] +'/'+ score[1] +'.csv'):
                dfs.append(pd.read_csv('./results/'+ type +'/'+timestamp+'/'+ score[0] +'/'+ score[1] +'.csv'))
            else:
                dfs.append(None)
        print(dfs)
        return get_store_and_tables(dfs)
    else:
        raise PreventUpdate



def render_tab_content_past_bench_openml(active_tab, data, type):
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
        elif active_tab == "histogram":
            print(data['histo_'+type[0]])
            print(data['histo_'+type[1]])
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[1]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[0].split('_')[1]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[0]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[1].split('_')[1]))))),
                                ], align="center"
                            )
                        )]
    return "No tab selected"



def get_lisd_dir(test):
    lis = (os.listdir('./results/'+test))
    dropdown = []
    for l in lis:
        dropdown.append({'label': l, 'value': l})
    return dropdown



def start():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    server = app.server

    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "22rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

    sidebar = html.Div(
        [
            html.H2("AutoML BenchMark", className="display-4"),
            html.Hr(),
            html.P(
                "Scegli il Benchmark da effettuare", className="lead"
            ),
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/", active="exact"),
                    dbc.NavLink("OpenML Benchmark", href="/openml", active="exact"),
                    dbc.NavLink("Kaggle BenchMark", href="/kaggle", active="exact"),
                    dbc.NavLink("Test BenchMark", href="/test", active="exact"),
                    dbc.NavLink("Risultati Precedenti OpenML", href="/results-openml", active="exact"),
                    dbc.NavLink("Risultati Precedenti Kaggle", href="/results-kaggle", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    openmlbenchmark = html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("OpenMl BenchMark", className="card-title"),
                            #html.P("This is some card text", className="card-text"),
                            dbc.FormGroup([
                                dbc.Label("Numero di DataFrame da testare", width=5),
                                dbc.Col(
                                    dbc.Input(
                                        id="ndf", type="number", placeholder="Numero di DF", min=1
                                    ),
                                    width=5,
                                )
                            ],row=True),
                            dbc.FormGroup([
                                dbc.Label("Numero minimo di istanze per ogni DataFrame",  width=5),
                                dbc.Col(
                                    dbc.Input(
                                        id="nmore", type="number", placeholder="N minimo di istanze", min=1, max=100000
                                    ),
                                    width=5,
                                ),
                            ],row=True),
                            dbc.Button("Avvia BenchMark", id='submit-openml', color="primary", className="mr-1")
                        ])
                    ], style={"width": "auto"},
                ),
                html.Hr(),
                dbc.Spinner(children=[html.Div(id='res-bench-openml')],size="lg", color="primary", type="border", fullscreen=True)
            ])

    kagglebenchmark = html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Kaggle BenchMark", className="card-title"),
                        #html.P("This is some card text", className="card-text"),
                        dbc.FormGroup([
                            dbc.Label("Numero di DataFrame da testare", width=5),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='kaggledataset',
                                    options=[
                                        {'label': 'Titanic', 'value': 'titanic'},
                                        {'label': 'altro', 'value': 'altro'}
                                    ],
                                    placeholder="Seleziona Dataframe",
                                    value=None,
                                    multi=True
                                ),
                                width=5,
                            )
                        ],row=True),
                        dbc.Button("Avvia BenchMark", id='submit-kaggle', color="primary", className="mr-1")
                    ])
                ], style={"width": "auto"}
            ),
            html.Hr(),
            dbc.Spinner(children=[html.Div(id='res-bench-kaggle')],size="lg", color="primary", type="border", fullscreen=True)
    ])

    testbenchmark = html.Div([
        dbc.Card([
                    dbc.CardBody([
                        html.H4("Test BenchMark", className="card-title"),
                        #html.P("This is some card text", className="card-text"),
                        dbc.FormGroup([
                            dbc.Label("ID DataFrame da testare", width=5),
                            dbc.Col(
                                dbc.Input(
                                    id="dfid", type="number", placeholder="DataFrame ID", min=1
                                ),
                                width=5,
                            )
                        ],row=True),
                        dbc.FormGroup([
                            dbc.Label("Algoritmo da utilizzare",  width=5),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='algorithms',
                                    options=[
                                        {'label': 'Autosklearn', 'value': 'autosklearn'},
                                        {'label': 'H2O', 'value': 'h2o'},
                                        {'label': 'TPOT', 'value': 'tpot'},
                                        {'label': 'AutoKears', 'value': 'autokeras'},
                                        {'label': 'AutoGluon', 'value': 'autogluon'},
                                        {'label': 'Tutti', 'value': 'all'}
                                    ],
                                    value='autosklearn'
                                ),
                                width=5,
                            ),
                        ],row=True),
                        dbc.Button("Avvia BenchMark", id='submit-test', color="primary", className="mr-1")
                    ])
                ], style={"width": "auto"},
            ),
        html.Hr(),
        dbc.Spinner(children=[html.Div(id='res-bench-test')],size="lg", color="primary", type="border", fullscreen=True)
    ])

    pastresultopenml = html.Div([
        dbc.Select(id='pastresultopenml', options=get_lisd_dir('OpenML'),
            placeholder='Filtra un BenchMark per Data',
        ),
        html.Hr(),
        
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-openml-table-class'),
            html.Div(id='result-past-bench-openml-graph-class'),
            html.Div(id='result-past-bench-openml-table-reg'),
            html.Div(id='result-past-bench-openml-graph-reg')
        ],size="lg", color="primary", type="border", fullscreen=False) 
    ])
    
    '''html.Div(id='result-past-bench-openml-table-class'),
        html.Div(id='result-past-bench-openml-graph-class'),
        html.Div(id='result-past-bench-openml-table-reg'),
        html.Div(id='result-past-bench-openml-graph-reg')'''

    pastresultkaggle = html.Div([
        dbc.Select(id='pastresultkaggle',options=get_lisd_dir('Kaggle'),
            placeholder='Filtra un BenchMark per Data',
        ),
        html.Hr(),
        dbc.Spinner(children=[html.Div(id='result-past-bench-kaggle')],size="lg", color="primary", type="border", fullscreen=False) 
    ])



    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div([dcc.Location(id="url"), sidebar, dcc.Store(id="store_class"), dcc.Store(id="store_reg"), content])
    app.validation_layout=html.Div([openmlbenchmark, kagglebenchmark, testbenchmark, pastresultopenml, pastresultkaggle])


   


    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return html.P('HOME')
        elif pathname == "/openml":
            return openmlbenchmark
        elif pathname == "/kaggle":
            return kagglebenchmark
        elif pathname == "/test":
            return testbenchmark
        elif pathname == '/results-openml':
            return pastresultopenml
        elif pathname == '/results-kaggle':
            return pastresultkaggle
        
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

    
    @app.callback(
        [Output('res-bench-openml', 'children')],
        [Input('submit-openml', 'n_clicks'), Input('res-bench-openml', 'disabled')],
        State('nmore', 'value'), State('ndf', 'value'))
    def start_openml(n_clicks, state_button, nmore, ndf):
        #print(state_button)
        if nmore is not None and ndf is not None:
            res = openml_benchmark(ndf, nmore)
            return [print_table_graphs(res)]
        else:
            raise PreventUpdate

    @app.callback(
        [Output('res-bench-kaggle', 'children')],
        [Input('submit-kaggle', 'n_clicks')],
        [State('kaggledataset', 'value')])
    def start_kaggle(n_clicks, kaggledataset):
        if kaggledataset is not None:
            res = kaggle_benchmark(kaggledataset)
            return [html.P(res, style={'color':'red'})] if isinstance(res, str) else [print_table_graphs(res)]
        else:
            raise PreventUpdate


    @app.callback(
        [Output('res-bench-test', 'children')],
        [Input('submit-test', 'n_clicks')],
        [State('dfid', 'value'), State('algorithms', 'value')])
    def start_test(n_clicks, dfid, algorithms):
        if dfid is not None and algorithms is not None:
            res = test(dfid, algorithms)
            #print(res)
            if isinstance(res[1], pd.DataFrame):
                return [dbc.Table.from_dataframe(res[1], striped=True, bordered=True, hover=True)]
            else:
                if res[0] is None:
                    return [html.P(res[1], style={'color':'red'})]
                else:
                    if(res[0] == 'classification'):
                        text = 'Accuracy: ' + str(res[1][0]) + '     f1_score: ' + str(res[1][1])
                    else:
                        text = 'RMSE: ' + str(res[1][0]) + '     r2_score: ' + str(res[1][1])
                    return [html.Div([
                        html.P('Risultati del Dataset: ' + str(dfid) + " utilizzando l'algoritmo: " + str(algorithms)),
                        html.P(text)
                    ])]
        else:
            raise PreventUpdate



#qui aggiorno i store di class e reg e stampo inizialmente le tabelle con i relativi risultati
    @app.callback([
        Output('store_class', 'data'),
        Output('store_reg', 'data'),
        Output('result-past-bench-openml-table-class', 'children'),
        Output('result-past-bench-openml-table-reg', 'children')
    ], [Input('pastresultopenml', 'value')])
    def get_store_past_bech_openml(timestamp):
        return get_store_past_bech_openml_function(timestamp, 'OpenML')

#modfico stra scatter e histogram i risultati di classificazione
    @app.callback([
        Output('result-past-bench-openml-graph-class', 'children'),
    ], [Input("tabs", "active_tab"), Input('store_class', 'data')])
    def render_tab_content_past_bench_openml_class(active_tab, data):
        if(data['scatter_class_acc'] is not None):
            return render_tab_content_past_bench_openml(active_tab, data, ('class_acc', 'class_f1'))
        else:
            return [None]
        #print(data)
        #return [None]

#modfico stra scatter e histogram i risultati di regressione
    @app.callback([
        Output('result-past-bench-openml-graph-reg', 'children')
    ], [Input("tabs", "active_tab"), Input('store_reg', 'data')])
    def render_tab_content_past_bench_openml_reg(active_tab, data):
        if(data['scatter_reg_rmse'] is not None):
            return render_tab_content_past_bench_openml(active_tab, data, ('reg_rmse', 'reg_r2'))
        else:
            return [None]
        #return [None]



    @app.callback([Output('result-past-bench-kaggle', 'children')], [Input('pastresultkaggle', 'value')])
    def retpastbenchopenml(timestamp):
        return get_csv_from_timestamp(timestamp, 'Kaggle')
    '''host='0.0.0.0', port=8050, '''

    app.run_server(debug=True)

if __name__ == '__main__':
    start()