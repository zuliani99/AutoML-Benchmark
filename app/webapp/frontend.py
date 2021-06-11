import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from .utils import get_lisd_dir


SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
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
                #dbc.Spinner(children=[
                    #html.Div(id='res-bench-openml')
                html.Div(id='res-bench-openml-table-class'),
                html.Div(id='res-bench-openml-graph-class'),
                html.Div(id='res-bench-openml-table-reg'),
                html.Div(id='res-bench-openml-graph-reg')
                #],size="lg", color="primary", type="border", fullscreen=True)
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
            dbc.Spinner(children=[
                #html.Div(id='res-bench-kaggle')
                html.Div(id='res-bench-kaggle-table-class'),
                html.Div(id='res-bench-kaggle-graph-class'),
                html.Div(id='res-bench-kaggle-table-reg'),
                html.Div(id='res-bench-kaggle-graph-reg')
            ],size="lg", color="primary", type="border", fullscreen=True)
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


pastresultkaggle = html.Div([
        dbc.Select(id='pastresultkaggle',options=get_lisd_dir('Kaggle'),
            placeholder='Filtra un BenchMark per Data',
        ),
        html.Hr(),
        dbc.Spinner(dbc.Spinner(children=[
            html.Div(id='result-past-bench-kaggle-table-class'),
            html.Div(id='result-past-bench-kaggle-graph-class'),
            html.Div(id='result-past-bench-kaggle-table-reg'),
            html.Div(id='result-past-bench-kaggle-graph-reg')
        ],size="lg", color="primary", type="border", fullscreen=False))
    ]) 
    