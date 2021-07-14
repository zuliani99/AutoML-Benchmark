# Import necessari
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from .utils import get_lisd_dir, create_collapses, read_markdown


# Personalizzazione dell stile dell'applicazione web
SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "backgroundColor": "#f8f9fa",
    }

# Definizione del sidebar
sidebar = html.Div(
        [
            html.H2("AutoML BenchMark", className="display-4"),
            html.Hr(),
            html.P(
                "Choose an Options", className="lead"
            ),
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/", active="exact"),
                    dbc.NavLink("OpenML Benchmark", href="/openml", active="exact"),
                    dbc.NavLink("Kaggle BenchMark", href="/kaggle", active="exact"),
                    dbc.NavLink("Test BenchMark", href="/test", active="exact"),
                    dbc.NavLink("Past Results OpenML", href="/results-openml", active="exact"),
                    dbc.NavLink("Past Results Kaggle", href="/results-kaggle", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

# Ddefinizone della pagina home
home = dcc.Markdown(read_markdown())

# definizione della pagina openmlbenchmark
openmlbenchmark = html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("OpenML BenchMark", className="card-title"),
                            dbc.FormGroup([
                                dbc.Label("Number of DataFrame to test each for classification tasks and regression tasks", width=5),
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="ndf", type="number", placeholder="Number of DF", min=1
                                        ),
                                        dbc.InputGroupAddon("at least 1", addon_type="prepend"),
                                    ])
                                ],width=5)
                            ],row=True),
                            dbc.FormGroup([
                                dbc.Label("Minimum number of instances for each DataFrame",  width=5),
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="nmore", type="number", placeholder="Number of instances", min=50, max=100000
                                        ),
                                        dbc.InputGroupAddon("at least 50 and at most 100000", addon_type="prepend"),
                                    ])
                                ],width=5),
                            ],row=True),
                            html.Div(create_collapses()),

                            dbc.Button("Start BenchMark", id='submit-openml', color="primary", className="mr-1")
                        ])
                    ], style={"width": "auto"},
                ),
                html.Hr(),
                dbc.Spinner(children=[
                    html.Div(id='res-bench-openml-table-class'),
                    dbc.Spinner(children=[
                        html.Div(id='res-bench-openml-graph-class'),
                    ],size="lg", color="primary", type="border", fullscreen=False),
                    html.Div(id='res-bench-openml-table-reg'),
                    dbc.Spinner(children=[
                        html.Div(id='res-bench-openml-graph-reg'),
                    ],size="lg", color="primary", type="border", fullscreen=False),
                ],size="lg", color="primary", type="border", fullscreen=True)
            ])

# Definizone della pagina kagglebenchmark
kagglebenchmark = html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Kaggle BenchMark", className="card-title"),
                        dbc.FormGroup([
                            dbc.Label("Competition's Dataframe to test", width=5),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='kaggledataframe',
                                    options=[
                                        {'label': 'Titanic - Classification', 'value': 'titanic'},
                                        {'label': 'CommonLit Readability Prize - Classification', 'value': 'contradictory-my-dear-watson'},
                                        {'label': "Forest Cover Type Prediction - Classification", 'value': 'forest-cover-type-prediction'},
                                        {'label': "Ghouls Goblins and Ghosts Boo - Classification", 'value': 'ghouls-goblins-and-ghosts-boo'},
                                        
                                        {'label': 'BigQuery-Geotab Intersection Congestion - Regression', 'value': 'bigquery-geotab-intersection-congestion'},
                                        {'label': 'Restaurant Revenue Prediction - Regression', 'value': 'restaurant-revenue-prediction'},
                                        {'label': 'CommonLit Readability Prize - Regression', 'value': 'commonlitreadabilityprize'},
                                        {'label': 'Tabular Playground Series - Feb 2021 - Regression', 'value': 'tabular-playground-series-feb-2021'},
                                        {'label': 'Global Energy Forecasting Competition 2012 - Regression', 'value': 'GEF2012-wind-forecasting'},

                                    ],
                                    placeholder="Select a Dataframe",
                                    value=None,
                                    multi=True
                                ),
                                width=5,
                            )
                        ],row=True),
                        html.Div(create_collapses()),

                        dbc.Button("Start BenchMark", id='submit-kaggle', color="primary", className="mr-1")
                    ])
                ], style={"width": "auto"}
            ),
            html.Hr(),
            dbc.Spinner(children=[
                html.Div(id='res-bench-kaggle-table-class'),
                dbc.Spinner(children=[
                    html.Div(id='res-bench-kaggle-graph-class'),
                ],size="lg", color="primary", type="border", fullscreen=False),
                html.Div(id='res-bench-kaggle-table-reg'),
                dbc.Spinner(children=[
                    html.Div(id='res-bench-kaggle-graph-reg'),
                ],size="lg", color="primary", type="border", fullscreen=False),
            ],size="lg", color="primary", type="border", fullscreen=True)
    ])

# Definizone della pagina testbenchmark
testbenchmark = html.Div([
        dbc.Card([
                    dbc.CardBody([
                        html.H4("Test BenchMark", className="card-title"),
                        dbc.FormGroup([
                            dbc.Label("DataFrame Id to test", width=5),
                            dbc.Col(
                                dbc.Input(
                                    id="dfid", type="number", placeholder="DataFrame ID", min=1
                                ),
                                width=5,
                            )
                        ],row=True),
                        dbc.FormGroup([
                            dbc.Label("Algorithms to use",  width=5),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='algorithms',
                                    options=[
                                        {'label': 'Autosklearn', 'value': 'autosklearn'},
                                        {'label': 'H2O', 'value': 'h2o'},
                                        {'label': 'TPOT', 'value': 'tpot'},
                                        {'label': 'AutoKears', 'value': 'autokeras'},
                                        {'label': 'AutoGluon', 'value': 'autogluon'},
                                        {'label': 'All', 'value': 'all'}
                                    ],
                                    value='autosklearn',
                                    searchable=False
                                ),
                                width=5,
                            ),
                        ],row=True),
                        html.Div(create_collapses()),
                        
                        dbc.Button("Start BenchMark", id='submit-test', color="primary", className="mr-1"),
                    ])
                ], style={"width": "auto"},
            ),
        html.Hr(),
        dbc.Spinner(children=[html.Div(id='res-bench-test')],size="lg", color="primary", type="border", fullscreen=True)
    ])

# Definizone della pagina get_pastresultopenml
def get_pastresultopenml():
    return html.Div([
        dbc.Select(id='pastresultopenml', options=get_lisd_dir('OpenML'),
            placeholder='Filter a BenchMark by Date',
        ),
        html.Hr(),
        html.Div(id='result-past-bench-openml-table-class'),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-openml-graph-class'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        html.Div(id='result-past-bench-openml-table-reg'),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-openml-graph-reg'),
        ],size="lg", color="primary", type="border", fullscreen=False),
    ])

# Definizone della pagina get_pastresultkaggle
def get_pastresultkaggle():
    return html.Div([
        dbc.Select(id='pastresultkaggle', options=get_lisd_dir('Kaggle'),
            placeholder='Filter a BenchMark by Date',
        ),
        html.Hr(),
        html.Div(id='result-past-bench-kaggle-table-class'),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-kaggle-graph-class'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        html.Div(id='result-past-bench-kaggle-table-reg'),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-kaggle-graph-reg'),
        ],size="lg", color="primary", type="border", fullscreen=False),
    ]) 
    