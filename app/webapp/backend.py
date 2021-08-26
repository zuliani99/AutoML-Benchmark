# Import needed
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from .frontend import openmlbenchmark, kagglebenchmark, testbenchmark, get_pastresultopenml, get_pastresultkaggle, home
from .utils import render_tab_content, get_store_past_bech_function, set_body, create_table, get_body_from_pipelines, checkoptions, displaying_error,check_dfs_sequence
from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import pandas as pd
import plotly.graph_objects as go

# Function for rendering web application pages
def render_page_content_function(pathname):
    return {
        '/': home,
        '/openml': openmlbenchmark,
        '/kaggle': kagglebenchmark,
        '/test': testbenchmark,
        '/results-openml': get_pastresultopenml(),
        '/results-kaggle': get_pastresultkaggle()
    }.get(pathname, # If the page you don't know is present, the 404 error will be shown
        dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )
    )

# Function for running the OpenML Benchmark
def start_openml_function(active_tab, dfs_squence, ndf, nmore, options):
    if active_tab == 'knownID':
        # If the benchmark is wanted on a specified ID list
        return start_openml_function_knownID(dfs_squence, options)
    # Otherwise it means that you have chosen the option to perform a benchmark on DataFrame filtered by parameters entered by the user
    if ndf is None or nmore is None or ndf < 1 or nmore < 50 or nmore > 100000:
        return displaying_error('')
    if not checkoptions(options):  # Check algorithm options entered
        return displaying_error('Please check the algorithms options inserted') # Error display
    return get_store_past_bech_function(openml_benchmark((ndf, nmore), options), 'start-OpenML', None) # Start benchmark with user entered options

# Function for running the OpenML Benchmark on a specific DataFrame sequence
def start_openml_function_knownID(dfs_squence, options):
    # Check the parameters entered by the user
    if dfs_squence is None or dfs_squence == '': 
        return displaying_error('')
    if not check_dfs_sequence(dfs_squence):
        return displaying_error('Please make sure each ID is followed by a comma')
    res = openml_benchmark(dfs_squence, options) # Running the benchmark
    if isinstance(res, str) and res[0:5] == 'Error': # In case of error, print the returned exception
        return displaying_error(res)
    return get_store_past_bech_function(res, 'start-OpenML', None) # Visualization of the results obtained
    
    

# Function for running the Kagle Benchmark
def start_kaggle_function(kaggledataframe, options):
    if kaggledataframe is None:
        return displaying_error('')
    if not checkoptions(options): # Check algorithm options entered
        return displaying_error('Please check the algorithms options inserted') # Error display
    return get_store_past_bech_function(kaggle_benchmark(kaggledataframe, options), 'start-Kaggle', None)


# Function for performing the Test Benchmark
def start_test_function(dfid, algorithms, options):
    if dfid is None or algorithms is None or dfid < 1:
        raise PreventUpdate
    if not checkoptions(options): # Check algorithm options entered
        if algorithms == 'all': return [html.P('Please check the algorithms options inserted', style={'color':'red'})] # Benchmark test on all algorithms
        else: return [html.P('Please check the ' + algorithms +' options inserted', style={'color':'red'})] # Benchmark test on a single algorithm
    task, res = test(dfid, algorithms, options) # Breakdown of the result obtained
    if isinstance(res, pd.DataFrame):
        return return_all_algorithms(task, res, res['dataframe'][0]) # If the result is a DataFrame this means that the test has been run for all available algorithms
    if task is None: 
        return [html.P(res, style={'color':'red'})] # If the task is not present it means that there was an execution error while downloading the DataFrame
    s1, s2, pipeline, timelife = res
    if pipeline[0:5] == 'Error': # If the first 5 characters of the pipeline variable are Error it means that an exception was thrown during the execution of the algorithm
        return [html.Div([
                        html.P('The execution of the benchmark for the dataframe: ' + str(dfid) + ' with: ' + algorithms + ', for: ' + str(options[algorithms]['time']) + ' minute/s throw an exception.'),
                        html.P(pipeline)
                    ], style={'color':'red'}
                )]
    # Definition of the test to be displayed containing the results of the two scores
    if(task == 'classification'): text = 'Accuracy: ' + str(s1) + '     F1 Score: ' + str(s2)
    else: text = 'RMSE: ' + str(s1) + '     R2 Score: ' + str(s2)

    # Complete display of the result
    return [html.Div([
            html.P(
                'Dataframe results ' + str(dfid) + ' by using the algorithm: ' + str(algorithms) + ' with starting running time: ' + str(options[algorithms]['time']) + ' minute/s' 
                + ' and with final running time: ' + str(timelife) + ' minute/s'
            ),
            html.P(text),
            set_body(str(algorithms), pipeline) # View the pipeline for that algorithm
    ])]

# Function for displaying the results of a Test Benchmark on all algorithms
def return_all_algorithms(task, res, name):
    # Decomposition of the resulting DataFrame
    first_scores = res.iloc[[0]]
    second_scores = res.iloc[[1]]
    pipelines = res.iloc[[2]]
    timelifes = res.iloc[[3]]

    bars = {'first': [], 'second': []}
    titles = []
    if(task == 'classification'): titles = ['Accuracy Score', 'F1 Score']
    else: titles = ['RMSE Score', 'R2 Score']
    
    # Populating the dictionary bars needed later to display the graph
    for col in first_scores.iloc[:, 1:]:
        bars['first'].append(go.Bar(y=first_scores[col], name=col))
        bars['second'].append(go.Bar(y=second_scores[col], name=col))

    return [
            html.Div([
                html.H3('Test Results form DataFrame ' + name),
                html.H4(titles[0]),
                create_table(first_scores.iloc[:, 1:]),     # Generation of the table for displaying the first score
                html.H4(titles[1]),
                create_table(second_scores.iloc[:, 1:]),    # Generation of the table for displaying the second score
                html.H4("Final Timelifes Algorithms"),
                create_table(timelifes.iloc[:, 1:]),        # Generation of the table for displaying the final life time of the algorithm
                html.Div(              # Display of the two graphs
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=go.Figure(data=bars['first'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[0]), title=dict(text = titles[0]))))),
                            dbc.Col(dcc.Graph(figure=go.Figure(data=bars['second'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[1]), title=dict(text = titles[1]))))),
                        ], align="center"
                    )
                ),
                html.H4("Pipelines"),
                html.Div(get_body_from_pipelines(pipelines, None, name)) # Visualization of algorithm pipelines
            ])
    ]

# Callback required to display a graph or not
def render_tab_content_function(active_tab, data, scores):
    if(data['scatter_'+scores[0]] is not None):
        return render_tab_content(active_tab, data, scores)
    else:
        return [None]


# Callback for managing the visualization and manipulation of collapses
def collapse_alogrithms_options_function(n1, n2, n3, n4, n5, is_open1, is_open2, is_open3, is_open4, is_open5):
    ctx = dash.callback_context

    if not ctx.triggered:
        return [False, False, False, False, False]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
    if button_id == "autosklearn-options" and n1:
        return [not is_open1, False, False, False, False]
    elif button_id == "h2o-options" and n2:
        return [False, not is_open2, False, False, False]
    elif button_id == "tpot-options" and n3:
        return [False, False, not is_open3, False, False]
    elif button_id == "mljar-options" and n4:
        return [False, False, False, not is_open4, False]
    elif button_id == "autogluon-options" and n5:
        return [False, False, False, False, not is_open5]
    return [False, False, False, False, False]