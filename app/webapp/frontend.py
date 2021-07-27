# Import needed
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from .utils import get_lisd_dir, create_collapses, read_markdown


# Customize the style of the web application
SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "backgroundColor": "#f8f9fa",
    }

# Definition of the sidebar
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

# Definition of the home page
home = dcc.Markdown(read_markdown())

# Definition of the OpenML Benchmark Tab for a specific DataFrame sequence
knownID = dbc.CardBody([
                        html.H4("OpenML BenchMark", className="card-title"),
                        dbc.FormGroup([
                                    dbc.Label("Specify the sequence of DatFrame IDs that you whant to test, each ID must be followed by a comma like so: 10,52,111", width=5),
                                    dbc.Col([
                                        dbc.InputGroup([
                                            dbc.Input(
                                                id="dfs-sequence", type="text", placeholder="Sequence of DF"
                                            ),
                                        ])
                                    ],width=5)
                                ],row=True),
                    ])

            
# Definition of the OpenML Benchmark Tab for a sequence of n classification and regression Dataframes with a greater number of x instances
unknownID = dbc.CardBody([
                                html.H4("OpenML BenchMark", className="card-title"),
                                dbc.FormGroup([
                                    dbc.Label("Number of DataFrame to test each for classification tasks and regression tasks or ", width=5),
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
                                            dbc.InputGroupAddon("at least 50", addon_type="prepend"),
                                        ])
                                    ],width=5),
                                ],row=True),
                    ])
            

# Definition of the openmlbenchmark page
openmlbenchmark = html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Tabs(
                                [
                                    dbc.Tab(knownID, label="Benchmark for known DataFrame IDs", tab_id='knownID'),
                                    dbc.Tab(unknownID, label="Benchmark for unknown DataFrame IDs", tab_id='unknownID'),
                                ],
                                id='openmlbenchmark-tabs',
                                card=True,
                                active_tab="knownID"
                            ),
                            html.Div(create_collapses()),
                            dbc.Button("Start BenchMark", id='submit-openml', color="primary", className="mr-1"),
                        ]),
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

# Definition of the kagglebenchmark page
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

# Definition of the testbenchmark page
testbenchmark = html.Div([
        dbc.Card([
                    dbc.CardBody([
                        html.H4("Test BenchMark", className="card-title"),
                        dbc.FormGroup([
                            dbc.Label("OpenML DataFrame ID to test", width=5),
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

# Definition of the get_pastresultopenml page
def get_pastresultopenml():
    return html.Div([
        dbc.Select(id='pastresultopenml', options=get_lisd_dir('OpenML'),
            placeholder='Filter a BenchMark by Date',
        ),
        dcc.Dropdown(
            id='pastresultopenml-comapre', placeholder="Compare with", multi=True, searchable=False
        ),
        dbc.Button("Search", id='submit-search-openml', color="primary", className="mr-1"),
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

# Definition of the get_pastresultkaggle page
def get_pastresultkaggle():
    return html.Div([
        dbc.Select(id='pastresultkaggle', options=get_lisd_dir('Kaggle'),
            placeholder='Filter a BenchMark by Date',
        ),
        dcc.Dropdown(
            id='pastresultkaggle-comapre', placeholder="Compare with", multi=True, searchable=False
        ),
        dbc.Button("Search", id='submit-search-kaggle', color="primary", className="mr-1"),
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
    