# Import necessari
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from .frontend import openmlbenchmark, kagglebenchmark, testbenchmark, get_pastresultopenml, get_pastresultkaggle, home
from .utils import render_tab_content, get_store_past_bech_function, set_body, create_table, get_body_from_pipelines, checkoptions
from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import pandas as pd
import plotly.graph_objects as go

# Funzione per il rendering delle pagine dell'applicazione web
def render_page_content_function(pathname):
    return {
        '/': home,
        '/openml': openmlbenchmark,
        '/kaggle': kagglebenchmark,
        '/test': testbenchmark,
        '/results-openml': get_pastresultopenml(),
        '/results-kaggle': get_pastresultkaggle()
    }.get(pathname,
        dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )
    )

#Output('store_class_openml', 'data'), Output('store_reg_openml', 'data'), Output('store_pipelines_class_openml', 'data'), Output('store_pipelines_reg_openml', 'data'), Output('res-bench-openml-table-class', 'children'), Output('res-bench-openml-table-reg', 'children')],
# Funzione per l'esecuzione del OpenML Benchmark
def start_openml_function(ndf, nmore, options):
    if ndf is None or nmore is None or ndf < 1 or nmore < 50 or nmore > 100000:
        raise PreventUpdate
    if not checkoptions(options):  # Verifica delle opzioni degli algoritmi inserite
        return None, None, None, None, [
            html.P('Please check the algorithms options inserted', style={'color':'red'}),
            dbc.Tabs( 
                [], id="tabs-class", active_tab="", style={'hidden':'true'})],[ dbc.Tabs( 
                [], id="tabs-reg", active_tab="", style={'hidden':'true'} )]
    res = openml_benchmark(ndf, nmore, options)
    return get_store_past_bech_function(res, 'OpenML')
    

# Funzione per l'esecuzione del Kagle Benchmark
def start_kaggle_function(kaggledataframe, options):
    if kaggledataframe is None:
        raise PreventUpdate
    if not checkoptions(options): # Verifica delle opzioni degli algoritmi inserite
        return None, None, None, None, [
            html.P('Please check the algorithms options inserted', style={'color':'red'}),
            dbc.Tabs( 
                [], id="tabs-class", active_tab="", style={'hidden':'true'})],[ dbc.Tabs( 
                [], id="tabs-reg", active_tab="", style={'hidden':'true'} )]
    res = kaggle_benchmark(kaggledataframe, options)
    return get_store_past_bech_function(res, 'Kaggle')


# Funzione per l'esecuzione del Test Benchmark
def start_test_function(dfid, algorithms, options):
    if dfid is None or algorithms is None or dfid < 1:
        raise PreventUpdate
    if not checkoptions(options): # Verifica delle opzioni degli algoritmi inserite
        if algorithms == 'all': return [html.P('Please check the algorithms options inserted', style={'color':'red'})] # Test Benchmark su tutti gli algoritmi
        else: return [html.P('Please check the ' + algorithms +' options inserted', style={'color':'red'})] # Test Benchmark su un solo algoritmo
    task, res = test(dfid, algorithms, options) # Scomposizione del risultato ottenuto
    if isinstance(res, pd.DataFrame):
        return return_all_algorithms(task, res, res['dataframe'][0]) # Se il risultato è un DataFrame questo significa cheil test è stato fatto girare per tutti gli algoritmi a dispopsizione 
    if task is None: 
        return [html.P(res, style={'color':'red'})] # Se il task non è presente vuol dire che c'è stato un errore di esecuzione durante il download del DataFrame
    s1, s2, pipeline, timelife = res
    if pipeline[0:5] == 'Error': # Se i primi 5 caratteri della variabile pipeline sono Error vuol dire che è statat generata un'eccezione durante l'esecuzione dell'algoritmo
        #print(options[algorithms])
        return [html.Div([
                        html.P('The execution of the benchmark for the dataframe: ' + str(dfid) + ' whit the algorithm: ' + algorithms + ' for ' + str(options[algorithms]['time']) + ' ' + options[algorithms]['type'] + ' throw an exception.'),
                        html.P(pipeline)
                    ], style={'color':'red'}
                )]
    # Definizione del test da visualizzare contenente i risultati dei due scores
    if(task == 'classification'):
        text = 'Accuracy: ' + str(s1) + '     f1_score: ' + str(s2)
    else:
        text = 'RMSE: ' + str(s1) + '     r2_score: ' + str(s2)
    # Visualizzazione competa del risultato
    return [html.Div([
            html.P(
                'Dataframe results ' + str(dfid) + ' by using the algorithm: ' + str(algorithms) + ' with starting running time: ' + str(options[algorithms]['time']) + ' ' + options[algorithms]['type']
                + ' and with final running time: ' + str(timelife) + ' ' + str(options[algorithms]['type'])
            ),
            html.P(text),
            set_body(str(algorithms), pipeline) # Visualizzazione della pipeline relativa a quel algoritmo 
    ])]

# Funzione per la visualizzazione dei risultati di un Test Benhmark su tutti gli algoritmi
def return_all_algorithms(task, res, name):
    # Scomposizione del DataFrame risultante
    first_scores = res.iloc[[0]]
    second_scores = res.iloc[[1]]
    pipelines = res.iloc[[2]]
    timelifes = res.iloc[[3]]

    bars = {'first': [], 'second': []}
    titles = []
    if(task == 'classification'): titles = ['Accuracy Score', 'F1 Score']
    else: titles = ['RMSE Score', 'R2 Score']
    
    # Popolamento del dizionario bars necessario successivamente per la visualizzazione del grafico
    for col in first_scores.iloc[:, 1:]:
        bars['first'].append(go.Bar(y=first_scores[col], name=col))
        bars['second'].append(go.Bar(y=second_scores[col], name=col))

    return [
            html.Div([
                html.H3('Test Results form DataFrame ' + name),
                html.H4(titles[0]),
                create_table(first_scores.iloc[:, 1:]),     # Generazione della tabella per la visualizzazione del primo score
                html.H4(titles[1]),
                create_table(second_scores.iloc[:, 1:]),    # Generazione della tabella per la visualizzazione del secondo score
                html.H4("Final Timelifes Algorithms"),
                create_table(timelifes.iloc[:, 1:]),        # Generazione della tabella per la visualizzazione del tempo di vita finale dell'algoritmo
                html.Div(              # Visualizzazione dei due grafici
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=go.Figure(data=bars['first'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[0]), title=dict(text = titles[0]))))),
                            dbc.Col(dcc.Graph(figure=go.Figure(data=bars['second'], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = titles[1]), title=dict(text = titles[1]))))),
                        ], align="center"
                    )
                ),
                html.H4("Pipelines"),
                html.Div(get_body_from_pipelines(pipelines, name)) # Visualizzazione delle pipelines degli algoritmi
            ])
    ]

# Callback necessaria per visaulizzare o meno un grafico
def render_tab_content_function(active_tab, data, scores):
    if(data['scatter_'+scores[0]] is not None):
        return render_tab_content(active_tab, data, scores)
    else:
        return [None]


# Callback per la gestione della visualizzazione e manipolazione dei collapse
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