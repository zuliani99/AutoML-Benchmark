#!/usr/bin/env python3

from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import sys
import pandas as pd
import argparse
import os
from termcolor import colored


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


def print_table(res):
    #cla = res[0]
    #reg = res[1]
    print(res)
    if not isinstance(res, list):
        res = [res]
    ris=[]
    for r in res:
        ris.append(html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in r.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(r.iloc[i][col]) for col in r.columns
                ]) for i in range(min(len(r), r.shape[0]))
            ])
        ]))
        if len(ris) == 1:
            return html.Div([
                ris[0]
            ])
        else:
            return html.Div([
                ris[0], ris[1]
            ])


def get_lisd_dir(test):
    lis = (os.listdir('./results/'+test))
    dropdown = []
    for l in lis:
        dropdown.append({'label': l, 'value': l})
    return dropdown



def start():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

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
                            html.H4("OpenMlBenchMark", className="card-title"),
                            #html.P("This is some card text", className="card-text"),
                            dbc.FormGroup([
                                dbc.Label("Numero di DataFrame da testare", width=5),
                                dbc.Col(
                                    dbc.Input(
                                        id="ndf", type="number", placeholder="Numero di DF", min=0
                                    ),
                                    width=5,
                                )
                            ],row=True),
                            dbc.FormGroup([
                                dbc.Label("Numero minimo di istanze per ogni DataFrame",  width=5),
                                dbc.Col(
                                    dbc.Input(
                                        id="nmore", type="number", placeholder="N minimo di istanze", min=0, max=100000
                                    ),
                                    width=5,
                                ),
                            ],row=True),
                            dbc.Button("Avvia BenchMark", id='submit-openml', color="primary", className="mr-1")
                        ])
                    ], style={"width": "auto"},
                ),
                dbc.Card(id='res-becnh-openml')
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
            dbc.Card(id='res-bench-kaggle')
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
                                    id="dfid", type="number", placeholder="DataFrame ID", min=0
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
        dbc.Card(id='res-becnh-test')
    ])

    pastresultopenml = html.Div([
        dbc.Select(id='pastresultopenml', options=get_lisd_dir('OpenML'),
            placeholder='Filtra un BenchMark per Data',
        ),
        dbc.Card(id='result-past-becnh-openml')
    ])

    pastresultkaggle = html.Div([
        dbc.Select(id='pastresultkaggle',options=get_lisd_dir('Kaggle'),
            placeholder='Filtra un BenchMark per Data',
        ),
        dbc.Card(id='result-past-becnh-kaggle')
    ])



    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
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
        Output('res-bench-openml', 'children'),
        Input('submit-openml', 'n_clicks'),
        State('nmore', 'value'), State('ndf', 'value'))
    def start_openml(n_clicks, nmore, ndf):
        res = openml_benchmark(ndf, nmore)
        return  print_table(res)

    @app.callback(
        Output('res-bench-kaggle', 'children'),
        Input('submit-kaggle', 'n_clicks'),
        State('kaggledataset', 'value'))
    def start_kaggle(n_clicks, kaggledataset):
        #res = kaggle_benchmark(kaggledataset)
        res = pd.read_csv('./results/OpenML/2021-04-28 19:52:53.106617/classification.csv')
        return  print_table(res)
        


    @app.callback(
        Output('res-bench-test', 'children'),
        Input('submit-test', 'n_clicks'),
        State('dfid', 'value'), State('algorithms', 'value'))
    def start_test(n_clicks,dfid, algorithms):
        res = test(dfid, algorithms)
        print('start')
        print(res)
        if isinstance(res[1], pd.DataFrame):
            return html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in dataframe.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                    ])
                ])
            ])
        else:
            if(res[0] == 'classification'):
                text = 'Accuracy: ' + str(res[1][0]) + '     f1_score: ' + str(res[1][1])
            else:
                text = 'RMSE: ' + str(res[1][0]) + '     r2_score: ' + str(res[1][1])
            return html.Div([
                html.Br(),
                html.P('Risultati del Dataset: ' + str(dfid) + " utilizzando l'algoritmo: " + str(algorithms)),
                html.P(text)
            ])

    @app.callback(Output('result-past-becnh-openml', 'children'), Input('pastresultopenml', 'value'))
    def retpastbenchopenml(timestamp):
        return print_table([pd.read_csv('./results/OpenML/'+timestamp+'/classification.csv'),pd.read_csv('./results/OpenML/'+timestamp+'/regression.csv')])

    app.run_server(debug=True)

if __name__ == '__main__':
    start()
