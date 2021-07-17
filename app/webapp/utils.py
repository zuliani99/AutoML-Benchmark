# Import necessari
import os
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import math
from dash.exceptions import PreventUpdate
import pandas as pd

# Funzione per ottenere le date dei vecchi Benchmarks
def get_lisd_dir(test):
    lis = (os.listdir('./results/'+test))
    lis.sort()
    return [{'label': l, 'value': l} for l in lis if l != '.gitignore']

# Funzione per la lettura del file README.md
def read_markdown():
    with open('../README.md', 'r') as file:
        data = file.read()
    return data

# Funzione per la visualizzazione dell'errore relativo al mal inseirmento delle opzioni degli algoritmi
def displaying_error(error):
    return None, None, None, None, [
        html.P(error, style={'color':'red'}),
            dbc.Tabs( 
                [], id="tabs-class", active_tab="", style={'hidden':'true'})],[
            dbc.Tabs( 
                [], id="tabs-reg", active_tab="", style={'hidden':'true'})]
    
# Funzione per definizone del dizionario contenente le opzioni inseirte dell'utente
def make_options(as_tl=1, h2o_tl=1, t_tl=1, ak_tl=10, ag_tl=1, as_f=False, h2o_f=False, t_f=False, ak_f=False, ag_f=False):
    return {
        'autosklearn': {'min': 1, 'time': as_tl, 'rerun': as_f, 'type': 'minute/s'},
        'h2o': {'min': 1,'time': h2o_tl, 'rerun': h2o_f, 'type': 'minute/s'},
        'tpot': {'min': 1,'time': t_tl, 'rerun': t_f, 'type': 'minute/s'},
        'autokeras': {'min': 10,'time': ak_tl, 'rerun': ak_f, 'type': 'epoch/s'},
        'autogluon': {'min': 1,'time': ag_tl, 'rerun': ag_f, 'type': 'minute/s'},
    }

# Funzione per il controllo che le opzioni inseirte dall'utnete siano valide
def checkoptions(options):
    return all(int(value['time']) >= int(value['min']) for key, value in options.items())

# Funzione per la definizone del dizionario dato alla gestione dei collapse
def render_collapse_options(choice):
    return {
        'autosklearn': [False, True, True, True, True],
        'h2o': [True, False, True, True, True],
        'tpot': [True, True, False, True, True],
        'autokeras': [True, True, True, False, True],
        'autogluon':[True, True, True, True, False],
        'all': [False, False, False, False, False],
    }.get(choice)


# Funzione per la gestione della visualizzazione delle tabelle e grafici dei benchmarks
def get_store_past_bech_function(timestamp, type):
    type_bench = type.split('-')[1]
    if timestamp is None:
        raise PreventUpdate
    dfs = []
    scores = [('classification','acc'), ('classification','f1_score'), ('regression','rmse'), ('regression','r2_score')]
    # Memorizzazione dei file csv relativi al benchmark che si vuole prendere in esame
    for score in scores:
        if os.path.exists('./results/'+ type_bench +'/'+timestamp+'/'+ str(score[0]) +'/'+ str(score[1]) +'.csv'):
            dfs.append(pd.read_csv('./results/'+ type_bench +'/'+timestamp+'/'+ str(score[0]) +'/'+ str(score[1]) +'.csv'))
        else:
            dfs.append(None)
    # Memorizzazione dei file csv relativi alle pipelines del benchmark
    for t in ('classification', 'regression'):
        if os.path.exists('./results/'+ type_bench +'/'+timestamp+'/'+ t + '/pipelines.csv'):
            dfs.append(pd.read_csv('./results/'+ type_bench +'/'+timestamp+'/'+ t + '/pipelines.csv', sep='@').to_dict())
        else:
            dfs.append(None)
    # Memorizzazione del file csv relativo alle opzioni che ha inserito l'utente
    dfs.append(pd.read_csv('./results/'+ type_bench +'/'+timestamp+'/options.csv'))
    return get_store_and_tables(dfs, type, timestamp)


def get_store_and_tables(dfs, type, timestamp):
    res_class_acc, res_class_f1, res_reg_rmse, res_reg_r2, pipelines_class, pipelines_reg, options = dfs # Scomposizione dell'array dato a parametro

    # Definizone dei dizionari ed array che andremo a restituire a fine funzione
    store_dict = { 'class': {}, 'reg': {} }
    store_pipelines = { 'class': {}, 'reg': {} }
    tables = [[None], [None]]

    store_dict['class'], store_pipelines['class'], tables[0] = retrun_graph_table([res_class_acc, res_class_f1], pipelines_class, 'Classification Results', 'class', type.split('-')[1], options, ('Accuracy', 'F1'))
    store_dict['reg'], store_pipelines['reg'], tables[1] = retrun_graph_table([res_reg_rmse, res_reg_r2], pipelines_reg, 'Regression Results', 'reg', type.split('-')[1], options, ('RMSE', 'R2'))

    if type.split('-')[0] == 'past':
        dataframe_acc = res_class_acc['dataframe'].to_numpy() if res_class_acc is not None else None
        dataframe_reg = res_reg_rmse['dataframe'].to_numpy() if res_reg_rmse is not None else None
        all_list = os.listdir('./results/'+type.split('-')[1])
        all_list.remove('.gitignore')
        all_list.remove(timestamp)
        dataframe_comapre = get_dfs_to_compare(dataframe_acc, dataframe_reg, options.iloc[options.shape[0]-1].to_numpy(), type.split('-')[1], all_list)
        return store_dict['class'], store_dict['reg'], store_pipelines['class'], store_pipelines['reg'], tables[0], tables[1], dataframe_comapre

    return store_dict['class'], store_dict['reg'], store_pipelines['class'], store_pipelines['reg'], tables[0], tables[1]

def get_dfs_to_compare(dfs_class, dfs_reg, options, type, all_list): # quindi ora ho i df calss, df reg e tutta la lista dei becnhmarkl passati ecetto me stesso
    dfs_comapre = []
    for past_bench in all_list:
        if os.path.exists('./results/'+ type +'/'+past_bench+'/classification/acc.csv'):
            cls = (pd.read_csv('./results/'+ type +'/'+past_bench+'/classification/acc.csv')['dataframe'].to_numpy())
        else: cls = None
        if os.path.exists('./results/'+ type +'/'+past_bench+'/regression/rmse.csv'):
            reg = (pd.read_csv('./results/'+ type +'/'+past_bench+'/regression/rmse.csv')['dataframe'].to_numpy())
        else: reg = None
        piptemp = pd.read_csv('./results/'+ type +'/'+past_bench+'/options.csv')
        pip = piptemp.iloc[piptemp.shape[0]-1].to_numpy()
        
        print(cls, reg, pip)

        if (cls == dfs_class).all() and (reg == dfs_reg).all() and (pip != options).any():
            # controlliamo che abbiano differenti timelife
            print('aggiungo')
            dfs_comapre.append(past_bench)
    return dfs_comapre


# Funzione per il rendering e visualizzazione dei risultati relativi al Benchmark preso in esame
def retrun_graph_table(dfs, pipelines, title, task, t, opts, scores):
    table = [html.H3(title)]
    if (dfs[0] is None or dfs[1] is None):
        return {'scatter_'+scores[0]: None, 'histo_'+scores[0]: None, 'scatter_'+scores[1]: None, 'histo_'+scores[1]: None, 'options': None}, None, dbc.Tabs( 
                [], id="tabs-class" if task == "class" else "tabs-reg", active_tab="", style={'hidden':'true'} 
            )
    scatters = []
    histos = []
    for index, df in enumerate(dfs):
        df['pipelines'] = get_pipelines_button(df[['dataframe']], df.columns[1].split('-')[1])

        # Populamento degli array con i relativi grafici e tabelle
        for col in df.columns[1:-1]:
            if isinstance(df[col][0], float): # Aggiorno gli array solo se si tratta di un istanza di tipo float, quindi escludo le celle con valore "no value"
                scatters.append(go.Scatter(x=df['dataframe'], y=df[col], name=col.split('-')[0], mode='lines+markers'))
                histos.append(go.Bar(x=df['dataframe'], y=df[col], name=col.split('-')[0]))
        table.extend((html.H4(scores[index] + ' Score'), create_table(df)))

    table.append(
        dbc.Tabs(
            [
                dbc.Tab(label="Histograms", tab_id="histogram"),
                dbc.Tab(label="Scatter", tab_id="scatter"),
                dbc.Tab(label="Algorithm Options", tab_id="algo-options"),
            ],
            id="tabs-"+task,
            active_tab="histogram",
        ) 
    )

    # Creazione della sezione rivolta alla visualizzazione delle opzioni degli algoritmi inserite dall'utente
    opts = opts.to_dict()
    options = [
        html.Div([
            html.P(["Autosklearn -> Starting running time: " + str(opts['autosklearn'][0]) + " minute/s, Final running time: " + str(opts['autosklearn'][1]) + " minute/s"]),
            html.P(["TPOT -> starting running time " + str(opts['tpot'][0]) + " minutes/s, Final running time: " + str(opts['tpot'][1]) + " minutes/s"]),
            html.P(["H2O -> starting running time " + str(opts['h2o'][0]) + " minute/s, Final running time: " + str(opts['h2o'][1]) + " minute/s"]),
            html.P(["AutoKeras -> starting running time  " + str(opts['autokeras'][0]) + " epoch/s, Final running time: " + str(opts['autokeras'][1]) + " epoch/s"]),
            html.P(["AutoGluon -> starting running time  " + str(opts['autogluon'][0]) + " minute/s, Final running time: " + str(opts['autogluon'][1]) + " minute/s"]),
        ])
    ]

    limit = 5 if t == 'OpenML' else 6 # Limite posto a 5 se siamo nel caso di un OpenML Benchmark altrimenti a 6 nel caso di un Kaggle Benchmark perchè c'è alche la presenza del leader

    return {
        'scatter_'+scores[0]: scatters[:limit], 'histo_'+scores[0]: histos[:limit], 'scatter_'+scores[1]: scatters[limit:], 'histo_'+scores[1]: histos[limit:], 'options': options
    }, pipelines, table


# Funzione per la visualizzazione di un messaggio di errore se presente
def print_err_table(cell):
    if isinstance(cell, float) and math.isnan(cell):
        return "Error", {"border": "1px solid black", 'color': 'red'}
    return cell, {"border": "1px solid black"}


# FUnzione per la generazione delle tebelle
def create_table(df):
    return html.Div([
                html.Table(
                            [html.Tr([html.Th(col) for col in df.columns], style={"border": "1px solid black"})]
                            + [
                                html.Tr(
                                    [   
                                        html.Td(print_err_table(df.iloc[i][col])[0], style = print_err_table(df.iloc[i][col])[1])
                                        for col in df.columns
                                    ], style={"border": "1px solid black"}
                                )
                                for i in range(len(df))
                            ]
                        
                    ,
                    style={'text-align':'center', 'width': '100%', "border-collapse": "collapse", "border": "1px solid black"},
                ),
                html.Br()
            ])
    

# Funzione per la generazioen dei Modal e del bottone per la loro visualizzazione
def get_pipelines_button(dfs, task):
    return [
        [
            dbc.Modal([
                    dbc.ModalHeader("Pipelines"),
                    dbc.ModalBody(
                        id={
                            'type': "body-modal-Pipelines",
                            'index': task + '-' + str(row['dataframe']),
                        }
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close",
                            id={
                                'type': "close-modal-Pipelines",
                                'index': task + '-' + str(row['dataframe']),
                            },
                            className="ml-auto",
                            n_clicks=0,
                        )
                    ),
                ],id={
                    'type': "modal-Pipelines",
                    'index': task + '-' + str(row['dataframe']),
                },
                size="xl",
                is_open=False,
                style={"max-width": "none", "width": "90%"}
            ),html.Div(
                [
                    dbc.Button(
                        "Pipelines",
                        id={
                            'type': "open-Pipelines",
                            'index': task + '-' + str(row['dataframe']),
                        },
                        value=task + '-' + str(row['dataframe']),
                        className="mr-1",
                        n_clicks=0,
                    )
                ]
            ),
        ]
        for index, row in dfs.iterrows()
    ]


# Funzione per la visaulizzazione del bar chart o del scatter plot oppure dei timelife degli algoritmi
def render_tab_content(active_tab, data, type):
    if active_tab and data is not None:
        if active_tab == "scatter":
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[0]], layout=go.Layout(xaxis = dict(title = 'dataframes'), yaxis = dict(title = type[0]), title=dict(text = type[0].upper() + ' Score'))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[1]], layout=go.Layout(xaxis = dict(title = 'dataframes'), yaxis = dict(title = type[1]), title=dict(text = type[1].upper() + ' Score'))))),
                                ], align="center"
                            )
                        )]
        elif active_tab == "histogram":
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[0]], layout=go.Layout(xaxis = dict(title = 'dataframes'), yaxis = dict(title = type[0]), title=dict(text = type[0].upper() + ' Score'))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[1]], layout=go.Layout(xaxis = dict(title = 'dataframes'), yaxis = dict(title = type[1]), title=dict(text = type[1].upper() + ' Score'))))),
                                ], align="center"
                            )
                        )]
        else:
            return [
                data['options']
            ]
    return "No tab selected"


# Funzione per la generazione di collapse relativi alla scelta del tempo massimo di esecuzione dell'algoritmo e alla scelta di effettuare una riesecuzione in caso il tempo specificato sia insufficiente
def create_collapses(): # algo, measure, min, disabled
    options = make_options()
    return [
        dbc.Card([
                dbc.CardHeader(
                    html.H2( dbc.Button(key.upper() + " Options", color="link", id=key + "-options", disabled=False,))
                ),
                dbc.Collapse(
                    dbc.CardBody([
                            dbc.FormGroup([
                                    dbc.Label("Running time in " + val['type'], width=5),
                                    dbc.Col([
                                            dbc.InputGroup([
                                                    dbc.Input(id=key + "-timelife", type="number", value=val['min'], placeholder=val['type'], min=val['min']),
                                                    dbc.InputGroupAddon("at least " + str(val['min']), addon_type="prepend"),
                                            ]),
                                        ],width=5,
                                    ),
                                    dcc.Checklist(
                                        options=[
                                            {
                                                'label': ' Allow the algorithm to re-run with a bigger timelife?',
                                                'value': key + '-flag-rerun',
                                            },
                                        ],
                                        id=key + '-flag-rerun',
                                        labelStyle={'display': 'inline-block'},
                                    ),
                                ],
                                row=True,
                            ),
                        ]
                    ),
                    id="collapse-" + key,
                ),
            ]
        )
        for key, val in options.items()
    ]


# Funzione per la manipolazione del testo del modal pipeline
def set_body(name, pipeline):
    if (
        name == 'tpot'
        and (pipeline[0:5] == 'Error')
        or name != 'tpot'
        and name != 'dataframe'
        and (pipeline[0:5] == 'Error')
    ):
        return html.Div(pipeline, style={'color':'red'})
    elif name == 'tpot':
        ret = []
        strings = pipeline.split('\n')
        for string in strings:
            ret.extend((string, html.Br()))
        return html.Div(ret)
    elif name != 'dataframe':
        return dcc.Markdown(pipeline)
    else:
        return html.Div(pipeline)

# Funzione per la gestione del testo da aggiungere al modal pipelines
def get_body_from_pipelines(pipeline, df_name):
    df = pd.DataFrame.from_dict(pipeline) if not isinstance(pipeline, pd.DataFrame) else pipeline
    df.reset_index(drop=True, inplace=True)
    col = df.columns
    index = df.index
    condition = df['dataframe'] == df_name
    row = index[condition].tolist()
    pipeline = df.iloc[int(row[0])]
    return [html.Div([
        html.H4(name),
        set_body(name, pipeline[i]), 
        html.Br()
    ]) for i, name in enumerate(col)]


# Funzione per la gestione della visualizzazione delle pipelines  
def show_hide_pipelines_function(store_pipelines_class, store_pipelines_reg, n1, n2, value, is_open):
    if n1 or n2:
        score = value.split('-')[0]
        df_name = value.split(score+'-')[1]
        if score in ['acc', 'f1']:
            return not is_open, get_body_from_pipelines(store_pipelines_class, df_name)
        else:
            return not is_open, get_body_from_pipelines(store_pipelines_reg, df_name)
    return is_open, None


def check_dfs_sequence(dfs_sequence):
    if dfs_sequence is None:
        raise PreventUpdate
    dfs = dfs_sequence.split(',')
    for df in dfs:
        try:
            int(df)
        except:
            return False
    return True