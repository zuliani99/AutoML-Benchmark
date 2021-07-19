# Import necessari
from typing import Sequence
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

# Funzione per il rendering delle pagine dell'applicazione web
def render_page_content_function(pathname):
    return {
        '/': home,
        '/openml': openmlbenchmark,
        '/kaggle': kagglebenchmark,
        '/test': testbenchmark,
        '/results-openml': get_pastresultopenml(),
        '/results-kaggle': get_pastresultkaggle()
    }.get(pathname, # In caso la pagina non sai presente verrà mostrato l'errore 404
        dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )
    )

# Funzione per l'esecuzione del OpenML Benchmark
def start_openml_function(active_tab, dfs_squence, ndf, nmore, options):
    if active_tab == 'knownID':
        # Se il benchmark è voluto su una lista di ID specificata
        return start_openml_function_knownID(dfs_squence, options)
    # Altrimenti vuol dire che si è scelta l'opzione di effettuare un benchmark su DataFrame filtrati da paramentri inseriti dall'utente
    if ndf is None or nmore is None or ndf < 1 or nmore < 50 or nmore > 100000:
        return displaying_error('')
    if not checkoptions(options):  # Verifica delle opzioni degli algoritmi inserite
        return displaying_error('Please check the algorithms options inserted') # Visualizzazione dell'errore
    return get_store_past_bech_function(openml_benchmark((ndf, nmore), options), 'start-OpenML', None) # Avvio del benchmark con le opzioni inserite dall'utente

# Funzione per l'esecuzione del OpenML Benchmark su una sequenza specifica di DataFrame
def start_openml_function_knownID(dfs_squence, options):
    # Controllo dei paramentri inseirti dall'utente
    if dfs_squence is None or dfs_squence == '': 
        return displaying_error('')
    if not check_dfs_sequence(dfs_squence):
        return displaying_error('Please make sure each ID is followed by a comma')
    res = openml_benchmark(dfs_squence, options) # Esecuzione del benchmark
    if isinstance(res, str) and res[0:5] == 'Error': # In caso di errore stampo l'eccezione ritornata
        return displaying_error(res)
    return get_store_past_bech_function(res, 'start-OpenML', None) # Visualizzazione dei risultati ottenuti
    
    

# Funzione per l'esecuzione del Kagle Benchmark
def start_kaggle_function(kaggledataframe, options):
    if kaggledataframe is None:
        return displaying_error('')
    if not checkoptions(options): # Verifica delle opzioni degli algoritmi inserite
        return displaying_error('Please check the algorithms options inserted') # Visualizzazione dell'errore
    return get_store_past_bech_function(kaggle_benchmark(kaggledataframe, options), 'start-Kaggle', None)


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
                html.Div(get_body_from_pipelines(pipelines, None, name)) # Visualizzazione delle pipelines degli algoritmi
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