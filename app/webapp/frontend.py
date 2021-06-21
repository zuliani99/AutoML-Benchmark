import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from .utils import get_lisd_dir, create_collapse


SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "backgroundColor": "#f8f9fa",
    }

sidebar = html.Div(
        [
            html.H2("AutoML BenchMark", className="display-4"),
            html.Hr(),
            html.P(
                "Choose the Benchmark to be made", className="lead"
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


home = dcc.Markdown('''
# AutoML-Benchmark
Benchmark for some usual automated machine learning, such as: [auto-sklearn](https://automl.github.io/auto-sklearn/master/), [auto-keras](https://autokeras.com/), [h20](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [tpot](http://epistasislab.github.io/tpot/) and [autogluon](http://epistasislab.github.io/tpot/). All visualized via a responsive Dash Web Application.


## Installation
First of all download the full package or clone it where ever you want. Then all you have to do is to run thid line of code in your bash window: 
```bash
sudo apt install install python3-pip
```

To install all dependencies run 
```bash
make install
```

## Usage
To run the app execute the following line of code:
```bash
python3 start.py
```
Open your favourite browser and go to: [http://127.0.0.1:8050/](http://127.0.0.1:8050/). Here you will be albe to interact with the application

There are five types of operations:

1. **OpenML Benchmark:** Here you can choose the number of datasets each for classification task and for regression task and the number of instances that a dataset at least has. This command will start a benchmark using openml datasets.

2. **Kaggle Benchmark:** Here you can choose multiple kaggle's datasets for running a benchmark on them.

3. **Test Benchmark:** Here you can run a benchmark on a specific dataset by insering the *dataset id* and using a single *algorithm* ot all of them by selecting a options

4. **Past Results OpenML:** Here you can navigate between past *OpenML* benchmark by selecting a specific date

5. **Past Results Kaggle:** Here you can navigate between past *Kaggle* benchmark by selecting a specific date

''')

openmlbenchmark = html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("OpenMl BenchMark", className="card-title"),
                            #html.P("This is some card text", className="card-text"),
                            dbc.FormGroup([
                                dbc.Label("Number of dataframe to test", width=5),
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
                            create_collapse('AutoSklearn', 'Minutes', 1, False), create_collapse('H2O', 'Minutes', 1, False), create_collapse('TPOT', 'Generations', 5, False), create_collapse('AutoKeras', 'Epochs', 10, False), create_collapse('AutoGluon', 'Minutes', 1, False),

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

kagglebenchmark = html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Kaggle BenchMark", className="card-title"),
                        #html.P("This is some card text", className="card-text"),
                        dbc.FormGroup([
                            dbc.Label("Number of dataframe to test", width=5),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='kaggledataset',
                                    options=[
                                        {'label': 'Titanic', 'value': 'titanic'},
                                        {'label': 'Tabular Playground', 'value': 'tabular-playground-series-mar-2021'},
                                        {'label': 'Mercedes Greener Manufactoring', 'value': 'mercedes-benz-greener-manufacturing'},
                                        {'label': 'Restaurant Revenue Prediction', 'value': 'restaurant-revenue-prediction'}
                                    ],
                                    placeholder="Select a Dataframe",
                                    value=None,
                                    multi=True
                                ),
                                width=5,
                            )
                        ],row=True),
                        create_collapse('AutoSklearn', 'Minutes', 1, False), create_collapse('H2O', 'Minutes', 1, False), create_collapse('TPOT', 'Generations', 5, False), create_collapse('AutoKeras', 'Epochs', 10, False), create_collapse('AutoGluon', 'Minutes', 1, False),

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

testbenchmark = html.Div([
        dbc.Card([
                    dbc.CardBody([
                        html.H4("Test BenchMark", className="card-title"),
                        #html.P("This is some card text", className="card-text"),
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
                                    value='autosklearn'
                                ),
                                width=5,
                            ),
                        ],row=True),
                        dbc.Button("Start BenchMark", id='submit-test', color="primary", className="mr-1"),
                        #html.Div(id='collapse-test')
                        create_collapse('AutoSklearn', 'Minutes', 1, False), create_collapse('H2O', 'Minutes', 1, True), create_collapse('TPOT', 'Generations', 5, True), create_collapse('AutoKeras', 'Epochs', 10, True), create_collapse('AutoGluon', 'Minutes', 1, True),
                    ])
                ], style={"width": "auto"},
            ),
        html.Hr(),
        dbc.Spinner(children=[html.Div(id='res-bench-test')],size="lg", color="primary", type="border", fullscreen=True)
    ])

pastresultopenml = html.Div([
        dbc.Modal(
            [
                dbc.ModalHeader("Header"),
                dbc.ModalBody(id='modalbody-pipelines'),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close-lg", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="modal-pipelines",
            size="lg",
            is_open=False,
        ),
        dbc.Select(id='pastresultopenml', options=get_lisd_dir('OpenML'),
            placeholder='Filter a BenchMark by Date',
        ),
        html.Hr(),
        
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-openml-table-class'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-openml-graph-class'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        dbc.Spinner(children=[
        html.Div(id='result-past-bench-openml-table-reg'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-openml-graph-reg'),
        ],size="lg", color="primary", type="border", fullscreen=False),
    ])


pastresultkaggle = html.Div([
        dbc.Select(id='pastresultkaggle', options=get_lisd_dir('Kaggle'),
            placeholder='Filter a BenchMark by Date',
        ),
        html.Hr(),

        dbc.Spinner(children=[
            html.Div(id='result-past-bench-kaggle-table-class'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-kaggle-graph-class'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        dbc.Spinner(children=[
        html.Div(id='result-past-bench-kaggle-table-reg'),
        ],size="lg", color="primary", type="border", fullscreen=False),
        dbc.Spinner(children=[
            html.Div(id='result-past-bench-kaggle-graph-reg'),
        ],size="lg", color="primary", type="border", fullscreen=False),
    ]) 
    