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


def printres(res):
    #cla = res[0]
    #reg = res[1]
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


def start():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = html.Div([
        html.H1('AutoML BenchMark'),
        html.H3('Seleziona una delle opzioni'),
        html.Div([
            dcc.Dropdown(
                id='commands',
                options=[
                    {'label': 'OpenML BenchMark', 'value': 'openml'},
                    {'label': 'Kaggle BenchMark', 'value': 'kaggle'},
                    {'label': 'Test BenchMark', 'value': 'test'}
                ],
                value='openml'
            )
        ]),

        html.Div(id='option', children=[]),

        html.Div(id='result-openml', children=[]),
        html.Div(id='result-kaggle', children=[]),
        html.Div(id='result-test', children=[])

    ])

    @app.callback(
        Output('option', 'children'),
        Input('commands', 'value'))
    def update_output(value):
        if(value == 'openml'):
            return html.Div(id='openml', children=[
                html.H4("Quanti DataFrame da testare?"),
                dcc.Input(id="ndf", type="number", placeholder="Numero di DF", min=0),
                html.Br(),
                html.H4("Numero minimo di istanze per ogni dataframe"),
                dcc.Input(id="nmore", type="number", placeholder="N minimo di istanze", min=0),
                html.Br(),
                dbc.Button("Avvia BenchMark", id='submit-openml', color="primary", className="mr-1")
            ])
        elif (value == 'kaggle'):
            return html.Div(id='kaggle', children=[
                html.H4("Che dataset testare?"),
                dcc.Dropdown(
                    id='kaggledataset',
                    options=[
                        {'label': 'Titanic', 'value': 'titanic'},
                        {'label': 'altro', 'value': 'altro'}
                    ],
                    value=None,
                    multi=True
                ),
                html.Br(),
                dbc.Button("Avvia BenchMark", id='submit-kaggle', color="primary", className="mr-1")
            ])
        else:
            return html.Div(id='test', children=[
                html.H4("Che dataset testare?"),
                dcc.Input(id="dfid", type="number", placeholder="DF ID", min=0),
                html.Br(),
                html.H4("Quale Algoritmo utilizzare?"),
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
                html.Br(),
                dbc.Button("Avvia BenchMark", id='submit-test', color="primary", className="mr-1")
            ])


    @app.callback(
        Output('restul-openml', 'children'),
        Input('submit-openml', 'n_clicks'),
        State('nmore', 'value'), State('ndf', 'value'))
    def start_openml(n_clicks, nmore, ndf):
        res = openml_benchmark(ndf, nmore)
        return  printres(res)


    @app.callback(
        Output('result-kaggle', 'children'),
        Input('submit-kaggle', 'n_clicks'),
        State('kaggledataset', 'value'))
    def start_kaggle(n_clicks, kaggledataset):
        #res = kaggle_benchmark(kaggledataset)
        res = pd.read_csv('./results/OpenML/2021-04-28 19:52:53.106617/classification.csv')
        return  printres(res)
        


    @app.callback(
        Output('result-test', 'children'),
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


    app.run_server(debug=True)



def start2():
    app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return html.P('HOME')
        elif pathname == "/openml":
            return dbc.Card([
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
            )
        elif pathname == "/kaggle":
            return dbc.Card([
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
                ], style={"width": "auto"},
            )
        elif pathname == "/test":
            return dbc.Card([
                    dbc.CardBody([
                        html.H4("OpenMlBenchMark", className="card-title"),
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
            )
        # If the user tries to reach a different page, return a 404 message
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )
    app.run_server(debug=True)

if __name__ == '__main__':
    #main()
    start2()