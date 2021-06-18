import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from .frontend import openmlbenchmark, kagglebenchmark, testbenchmark, pastresultopenml, pastresultkaggle
from .utils import get_store_and_tables, render_tab_content
from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import pandas as pd
import plotly.graph_objects as go


def render_page_content_function(pathname):
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

def start_openml_function(nmore, ndf, options):
        #print(state_button)
        if nmore is not None and ndf is not None:
            res = openml_benchmark(ndf, nmore, options)
            #return [print_table_graphs(res)]
            return get_store_and_tables(res, 'OpenML')
        else:
            raise PreventUpdate

def start_kaggle_function(kaggledataset, options):
        if kaggledataset is not None:
            res = kaggle_benchmark(kaggledataset, options)
            return [html.P(res, style={'color':'red'})] if isinstance(res, str) else get_store_and_tables(res, 'Kaggle') #controllare la gestione degli errori
        else:
            raise PreventUpdate


def start_test_function(dfid, algorithms, options):
        if dfid is not None and algorithms is not None:
            res = test(dfid, algorithms, options)
            #print(res)
            if isinstance(res[1], pd.DataFrame):
                first_score = res[1].iloc[:1]
                second_score = res[1].iloc[1:]
                bars = {'first': [], 'second': []}
                titles = []            
                if(res[0] == 'classification'):
                    titles = ['Accuracy Score:', 'F1 Score:']
                else:
                    titles = ['RMSE Score:', 'R2 Score:']
                for col in first_score:
                    bars['first'].append(go.Bar(y=first_score[col], name=col.split('-')[0])) #attenzione alla x, ora l'ho rimossa
                    bars['second'].append(go.Bar(y=second_score[col], name=col.split('-')[0]))
                return [
                    html.Div([
                        html.H2('Test Results form DataFrame ' + str(dfid)),
                        html.H4(titles[0]),
                        dbc.Table.from_dataframe(first_score, striped=True, bordered=True, hover=True),
                        html.H4(titles[1]),
                        dbc.Table.from_dataframe(second_score, striped=True, bordered=True, hover=True),
                        html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=bars['first'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[0]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=bars['second'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[1]))))),
                                ], align="center"
                            )
                        )
                    ])
                ]
            else:
                if res[0] is None:
                    return [html.P(res[1], style={'color':'red'})]
                else:
                    if(res[0] == 'classification'):
                        text = 'Accuracy: ' + str(res[1][0]) + '     f1_score: ' + str(res[1][1])
                    else:
                        text = 'RMSE: ' + str(res[1][0]) + '     r2_score: ' + str(res[1][1])
                    return [html.Div([
                        html.P('Dataframe results ' + str(dfid) + " by using the algorithm: " + str(algorithms) + "with running time: " + str(options[algorithms]['time']) + " " + str(options[algorithms]['type'])),
                        html.P(text),
                    ])]
        else:
            raise PreventUpdate


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