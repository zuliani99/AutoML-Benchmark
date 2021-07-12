import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from .frontend import openmlbenchmark, kagglebenchmark, testbenchmark, get_pastresultopenml, get_pastresultkaggle, home
from .utils import render_tab_content, get_store_past_bech_function, set_body, create_table, get_body_from_pipelines
from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import pandas as pd
import plotly.graph_objects as go


def render_page_content_function(pathname):
        if pathname == "/":
            return home
        elif pathname == "/openml":
            return openmlbenchmark
        elif pathname == "/kaggle":
            return kagglebenchmark
        elif pathname == "/test":
            return testbenchmark
        elif pathname == '/results-openml':
            return get_pastresultopenml()
        elif pathname == '/results-kaggle':
            return get_pastresultkaggle()
        
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

#Output('store_class_openml', 'data'), Output('store_reg_openml', 'data'), Output('store_pipelines_class_openml', 'data'), Output('store_pipelines_reg_openml', 'data'), Output('res-bench-openml-table-class', 'children'), Output('res-bench-openml-table-reg', 'children')],
def start_openml_function(nmore, ndf, options):
        if nmore is not None and ndf is not None:
            res = openml_benchmark(ndf, nmore, options)
            return get_store_past_bech_function(res, 'OpenML')
        else:
            raise PreventUpdate

def start_kaggle_function(kaggledataframe, options):
        if kaggledataframe is None:
                raise PreventUpdate
        res = kaggle_benchmark(kaggledataframe, options)
        #print('dataaaaaaa: ', res)
        return get_store_past_bech_function(res, 'Kaggle')


def start_test_function(dfid, algorithms, options):
        if dfid is None or algorithms is None:
                raise PreventUpdate
        task, res = test(dfid, algorithms, options) #task, s1, s2, pipelines, timelife      s1 = acc or rmse,  s2 = f1 or r2   oppure task, dataframe
        #print(res)
        if isinstance(res, pd.DataFrame):
            return return_all_algorithms(task, res, res['dataframe'][0])
        if task is None:
            return [html.P(res, style={'color':'red'})]
        s1, s2, pipeline, timelife = res
        if pipeline[0:5] == 'Error':
            return html.Div([
                        html.P('The execution of the benchmark for the dataframe: ' + dfid + ' whit the algorithm: ' + algorithms + ' for ' + options['time'] + ' ' + options['type'] + ' throw an exception.'),
                        html.P(pipeline)
                    ], style={'color':'red'}
                )
        if(task == 'classification'):
            text = 'Accuracy: ' + str(s1) + '     f1_score: ' + str(s2)
        else:
            text = 'RMSE: ' + str(s1) + '     r2_score: ' + str(s2)
        return [html.Div([
            html.P(
                'Dataframe results ' + str(dfid) + ' by using the algorithm: ' + str(algorithms) + ' with starting running time: ' + str(options[algorithms]['time']) + ' ' + str(options[algorithms]['type'])
                + ' and with final running time: ' + str(timelife) + ' ' + str(options[algorithms]['type'])
            ),
            html.P(text),
            set_body(str(algorithms), pipeline)
        ])]


#REFACTORING TEST BENCHMARK
def return_all_algorithms(task, res, name):
        #res.to_csv('vediamo.csv', index=False)

        first_scores = res.iloc[[0]]
        second_scores = res.iloc[[1]]
        pipelines = res.iloc[[2]]
        timelifes = res.iloc[[3]]

        #print(pipelines)

        bars = {'first': [], 'second': []}
        titles = []
        if(task == 'classification'):
            titles = ['Accuracy Score', 'F1 Score']
        else:
            titles = ['RMSE Score', 'R2 Score']
        for col in first_scores.iloc[:, 1:]:
            bars['first'].append(go.Bar(y=first_scores[col], name=col))
            bars['second'].append(go.Bar(y=second_scores[col], name=col))

        return [
            html.Div([
                html.H2('Test Results form DataFrame ' + name),
                html.H4(titles[0]),
                create_table(first_scores.iloc[:, 1:]),
                html.H4(titles[1]),
                create_table(second_scores.iloc[:, 1:]),
                html.H4("Final Timelifes Algorithms"),
                create_table(timelifes.iloc[:, 1:]),
                html.Div(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=go.Figure(data=bars['first'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[0]), title=dict(text = titles[0]))))),
                            dbc.Col(dcc.Graph(figure=go.Figure(data=bars['second'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[1]), title=dict(text = titles[1]))))),
                        ], align="center"
                    )
                ),
                html.H4("Pipelines"),
                #create_table(pipelines)
                #create_div_pipelines(pipelines)
                html.Div(get_body_from_pipelines(pipelines, name))
            ])
        ]


def render_tab_content_function(active_tab, data, scores):
    if(data['scatter_'+scores[0]] is not None):
        return render_tab_content(active_tab, data, scores)
    else:
        return [None]


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
    elif button_id == "autokeras-options" and n4:
        return [False, False, False, not is_open4, False]
    elif button_id == "autogluon-options" and n5:
        return [False, False, False, False, not is_open5]
    return [False, False, False, False, False]