# Import needed
import os
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import math
from dash.exceptions import PreventUpdate
import pandas as pd
import collections
import copy

# Function to get the dates of the old Benchmarks
def get_lisd_dir(test):
    lis = (os.listdir('./results/'+test))
    lis.sort()
    return [{'label': l, 'value': l} for l in lis if l != '.gitignore']

# Function for reading the README.md file
def read_markdown():
    with open('../README.md', 'r') as file:
        data = file.read()
    return data

# Function for displaying the error relating to the incorrect insertion of the algorithm options
def displaying_error(error):
    return None, None, None, None, [
        html.P(error, style={'color':'red'}),
            dbc.Tabs( 
                [], id="tabs-class", active_tab="", style={'hidden':'true'})],[
            dbc.Tabs( 
                [], id="tabs-reg", active_tab="", style={'hidden':'true'})]
    
# Function by definition of the dictionary containing the options entered by the user
def make_options(as_tl=1, h2o_tl=1, t_tl=1, mj_tl=1, ag_tl=1, as_f=False, h2o_f=False, t_f=False, mj_f=False, ag_f=False):
    return {
        'autosklearn': {'min': 1, 'default': 60, 'time': as_tl, 'rerun': as_f, 'type': 'minute/s'},
        'h2o': {'min': 1, 'default': None, 'time': h2o_tl, 'rerun': h2o_f, 'type': 'minute/s'},
        'tpot': {'min': 1, 'default': None,'time': t_tl, 'rerun': t_f, 'type': 'minute/s'},
        'mljar': {'min': 1, 'default': 60,'time': mj_tl, 'rerun': mj_f, 'type': 'minute/s'},
        'autogluon': {'min': 1, 'default': None,'time': ag_tl, 'rerun': ag_f, 'type': 'minute/s'},
    }

# Function for checking that the options entered by the user are valid
def checkoptions(options):
    return not any(
        ((key in ['autosklearn', 'mljar'] and value['time'] is None))
        or ((value['time'] is not None) and (value['time'] < value['min']))
        for key, value in options.items()
    )
    #return all(int(value['time']) >= int(value['min']) for key, value in options.items())

# Function for the definition of the dictionary given to the management of collapses
def render_collapse_options(choice):
    return {
        'autosklearn': [False, True, True, True, True],
        'h2o': [True, False, True, True, True],
        'tpot': [True, True, False, True, True],
        'mljar': [True, True, True, False, True],
        'autogluon':[True, True, True, True, False],
        'all': [False, False, False, False, False],
    }.get(choice)

# Function for modifying the dropdown relative to comparable benchmarks
def modify_dropdown_comparedf_function(timestamp, comapre_list, type):
    if(timestamp is None):
        raise PreventUpdate
    dfs = get_dfs_from_timestamp([timestamp], type)
    dfs_compare = get_dfs_from_timestamp(comapre_list, type) if comapre_list is not None else [None]

    dataframe_acc = dfs[0][0]['dataframe'].to_list() if dfs[0][0] is not None else [None]
    dataframe_reg = dfs[0][2]['dataframe'].to_list() if dfs[0][2] is not None else [None]

    all_list = os.listdir('./results/'+ type)
    all_list.remove('.gitignore')
    all_list.remove(timestamp)
    if dfs_compare is not None:
        for df in dfs_compare:
            if df in all_list: all_list.remove(df)

    #return [get_dfs_to_compare(dataframe_acc, dataframe_reg, dfs[0][7].iloc[0].to_list(), type, all_list)]
    return [get_dfs_to_compare(dataframe_acc, dataframe_reg, dfs[0][8].iloc[0].to_list(), type, all_list)]

# Function to get all the data of a benchmark given the timestamp
def get_dfs_from_timestamp(timestamp, type_bench): # Return a list of lists of dataframes
    dfs = []
    scores = [('classification','acc'), ('classification','f1_score'), ('regression','rmse'), ('regression','r2_score')]
    for ts in timestamp:
        df = []
        # Storage of csv files related to the benchmark to be examined
        for score in scores:
            if os.path.exists('./results/'+ type_bench +'/'+ts+'/'+ str(score[0]) +'/'+ str(score[1]) +'.csv'):
                df.append(pd.read_csv('./results/'+ type_bench +'/'+ts+'/'+ str(score[0]) +'/'+ str(score[1]) +'.csv'))
            else: df.append(None)

        # Storage of csv files related to benchmark pipelines
        for t in ('classification', 'regression'):
            if os.path.exists('./results/'+ type_bench +'/'+ts+'/'+ t + '/pipelines.csv'):
                df.append(pd.read_csv('./results/'+ type_bench +'/'+ts+'/'+ t + '/pipelines.csv', sep='@').to_dict())
            else: df.append(None)

        for t in ('classification', 'regression'):
            if os.path.exists('./results/'+ type_bench +'/'+ts+'/'+ t + '/timelife_end.csv'):
                df.append(pd.read_csv('./results/'+ type_bench +'/'+ts+'/'+ t + '/timelife_end.csv'))
            else: df.append(None)

        # Storage of the csv file relating to the options entered by the user
        df.append(pd.read_csv('./results/'+ type_bench +'/'+ts+'/options_start.csv'))
        dfs.append(df)

    return dfs


# Function for managing the display of tables and graphs of the benchmarks
def get_store_past_bech_function(timestamp, type, compare_with):
    type_bench = type.split('-')[1]
    if timestamp is None:
        raise PreventUpdate
    return get_store_and_tables(get_dfs_from_timestamp([timestamp], type_bench), type, compare_with)


# Function for the merge of the dataframes of the benchmarks to be compared
def combile_dfs(df_from, dfs_comapre):
    to_return = copy.copy(df_from)
    for df in dfs_comapre: # For all algorithms I have to buy with to return
        for index in range(len(to_return)):
            if(to_return[index] is not None and df[index] is not None):
                # Concaetno to_return [index] with df [index]
                if(isinstance(to_return[index], dict)):
                    to_return[index] = ((pd.DataFrame.from_dict(to_return[index]).append(pd.DataFrame.from_dict(df[index]))).reset_index(drop=True)).to_dict()
                else:
                    to_return[index] = to_return[index].append(df[index]).reset_index(drop=True)

    return to_return


# Function to get the components for displaying the benchmark data
def get_store_and_tables(dfs, type, compare_with):
    res_class_acc, res_class_f1, res_reg_rmse, res_reg_r2, pipelines_class, pipelines_reg, options_end_class, options_end_reg, options_start = dfs[0]# Decomposition of the array given as a parameter
    dfs_compare = get_dfs_from_timestamp(compare_with, type.split('-')[1]) if compare_with is not None else None # List of comparable data frames

    # Definition of dictionaries and arrays that we will return at the end of the function
    store_dict = { 'class': {}, 'reg': {} }
    store_pipelines = { 'class': {}, 'reg': {} }
    tables = [[None], [None]]

    if dfs_compare is not None:
        # If I have benchmarks to compare I have to update the variables
        res_class_acc, res_class_f1, res_reg_rmse, res_reg_r2, pipelines_class, pipelines_reg, options_end_class, options_end_reg, options_start = combile_dfs(dfs[0], dfs_compare)

    store_dict['class'], store_pipelines['class'], tables[0] = retrun_graph_table([res_class_acc, res_class_f1], pipelines_class, 'Classification Results', 'class', type.split('-')[1], options_start, options_end_class, options_end_reg, ('Accuracy', 'F1'))
    store_dict['reg'], store_pipelines['reg'], tables[1] = retrun_graph_table([res_reg_rmse, res_reg_r2], pipelines_reg, 'Regression Results', 'reg', type.split('-')[1], options_start, options_end_class, options_end_reg, ('RMSE', 'R2'))

    return store_dict['class'], store_dict['reg'], store_pipelines['class'], store_pipelines['reg'], tables[0], tables[1]


# Function to set benchmarks comparable to the one just selected
def get_dfs_to_compare(dfs_class, dfs_reg, options_start, type, all_list):
    dfs_comapre = []
    for past_bench in all_list:
        if os.path.exists('./results/'+ type +'/'+past_bench+'/classification/acc.csv'):
            cls = (pd.read_csv('./results/'+ type +'/'+past_bench+'/classification/acc.csv')['dataframe'].to_list())
        else: cls = [None]
        if os.path.exists('./results/'+ type +'/'+past_bench+'/regression/rmse.csv'):
            reg = (pd.read_csv('./results/'+ type +'/'+past_bench+'/regression/rmse.csv')['dataframe'].to_list())
        else: reg = [None]
        time_limit = (pd.read_csv('./results/'+ type +'/'+past_bench+'/options_start.csv')).iloc[0].to_list()
        
        
        #if collections.Counter(cls) == collections.Counter(dfs_class) and collections.Counter(reg) == collections.Counter(dfs_reg) and collections.Counter(pip) != collections.Counter(options_end):

        # Ora ho messo che posso comparare dei benchmark aventi gli stessi dataframe ma con start time limit diiferenti, conmfronti effettuati tutti sul primo benchmark scelto
        # Devo decidere se implementare il confronto tra tutti quelli selezionati
        
        # Check if the benchmark is comparable to the one selected initially
        if collections.Counter(cls) == collections.Counter(dfs_class) and collections.Counter(reg) == collections.Counter(dfs_reg) and collections.Counter(time_limit) != collections.Counter(options_start):
            dfs_comapre.append({'label': past_bench, 'value': past_bench})
    return dfs_comapre


# Function for rendering and displaying the results relating to the Benchmark under consideration
def retrun_graph_table(dfs, pipelines, title, task, t, options_start, options_end_class, options_end_reg, scores):
    if (dfs[0] is None or dfs[1] is None):
        return {'scatter_'+scores[0]: None, 'histo_'+scores[0]: None, 'scatter_'+scores[1]: None, 'histo_'+scores[1]: None, 'options': None}, None, dbc.Tabs( 
                [], id="tabs-class" if task == "class" else "tabs-reg", active_tab="", style={'hidden':'true'} 
            )
            
    options_end = options_end_class if task == 'class' else options_end_reg
    table = [html.H3('Timelifes algorithms'), html.H4('Initial time limit'), create_table(options_start), html.H4('Actual time spent on computation'), create_table(options_end), html.H3(title)]
    scatters = []
    histos = []
    for index, df in enumerate(dfs):
        df['pipelines'] = get_pipelines_button(df[['date', 'dataframe']], df.columns[2].split('-')[1])

        # Population of arrays with related graphics and tables
        for col in df.columns[2:-1]:
            if isinstance(df[col][0], float): # Update arrays only if it is a float instance, so I exclude cells with "no value"
                scatters.append(go.Scatter(x=(df['date'], df['dataframe']), y=df[col], name=col.split('-')[0], mode='lines+markers'))
                histos.append(go.Bar(x=(df['date'], df['dataframe']), y=df[col], name=col.split('-')[0]))
        table.extend((html.H4(scores[index] + ' Score'), create_table(df)))

    table.append(
        dbc.Tabs(
            [
                dbc.Tab(label="Histograms", tab_id="histogram"),
                dbc.Tab(label="Scatter", tab_id="scatter"),
            ],
            id="tabs-"+task,
            active_tab="histogram",
        ) 
    )

    limit = 5 if t == 'OpenML' else 6 # Limit set to 5 if we are in the case of an OpenML Benchmark otherwise to 6 in the case of a Kaggle Benchmark because there is also the presence of the leader

    return {
        'scatter_'+scores[0]: scatters[:limit], 'histo_'+scores[0]: histos[:limit], 'scatter_'+scores[1]: scatters[limit:], 'histo_'+scores[1]: histos[limit:]
    }, pipelines, table


# Function for displaying an error message if present
def print_err_table(cell):
    if isinstance(cell, float) and math.isnan(cell):
        return "Error", {"border": "1px solid black", 'color': 'red'}
    return cell, {"border": "1px solid black"}


# Function for generating tables
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
    

# Function for the generation of Modals and the button for their display
def get_pipelines_button(dfs, task):
    return [
        [
            dbc.Modal([
                    dbc.ModalHeader("Pipelines"),
                    dbc.ModalBody(
                        id={
                            'type': "body-modal-Pipelines",
                            'index': task + '@' + str(row['date']) + '@' + str(row['dataframe']),
                        }
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close",
                            id={
                                'type': "close-modal-Pipelines",
                                'index': task + '@' + str(row['date']) + '@' + str(row['dataframe']),
                            },
                            className="ml-auto",
                            n_clicks=0,
                        )
                    ),
                ],id={
                    'type': "modal-Pipelines",
                    'index': task + '@' + str(row['date']) + '@' + str(row['dataframe']),
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
                            'index': task + '@' + str(row['date']) + '@' + str(row['dataframe']),
                        },
                        value=task + '@' + str(row['date']) + '@' + str(row['dataframe']),
                        className="mr-1",
                        n_clicks=0,
                    )
                ]
            ),
        ]
        for index, row in dfs.iterrows()
    ]


# Function for displaying the bar chart or the scatter plot or the timelife of the algorithms
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
    return "No tab selected"


# Function for the generation of collapses related to the choice of the maximum execution time of the algorithm and to the choice of performing a re-execution in case the specified time is insufficient
def create_collapses():
    options = make_options()
    return [
        dbc.Card([
                dbc.CardHeader(
                    html.H2( dbc.Button(key.upper() + " Options", color="link", id=key + "-options", disabled=False,))
                ),
                dbc.Collapse(
                    dbc.CardBody([
                            dbc.FormGroup([
                                    dbc.Label("Running time in " + val['type'] + ", (default value: " + str(val['default']) + ")", width=5),
                                    dbc.Col([
                                            dbc.InputGroup([
                                                    dbc.Input(id=key + "-timelife", type="number", value=val['default'], placeholder=val['type'], min=val['min']),
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


# Function for manipulating the text of the modal pipeline
def set_body(name, pipeline):
    if pipeline[0:5] == 'Error':
        return html.Div(pipeline, style={'color':'red'})
    elif name in ['tpot', 'autosklearn']:
        ret = []
        strings = pipeline.split('\n')
        for string in strings:
            ret.extend((string, html.Br()))
        return html.Div(ret)
    elif name == 'dataframe':
        return html.Div(pipeline)
    elif name == 'mljar':
        #print(type(pipeline), pipeline)
        #return html.Div([dbc.Table(dcc.Markdown(pipeline[0])), html.H6('Ensemble Statistics'), dbc.Table(dcc.Markdown(pipeline[1])), dbc.Table(dcc.Markdown(pipeline[2]))])
        return dcc.Markdown(pipeline)
    else:
        return dbc.Table(dcc.Markdown(pipeline))

# Function for managing the text to be added to the modal pipelines
def get_body_from_pipelines(pipeline, date, df_name):
    df = pd.DataFrame.from_dict(pipeline) if not isinstance(pipeline, pd.DataFrame) else pipeline
    df.reset_index(drop=True, inplace=True)
    col = df.columns
    index = df.index
    condition = (df['date'] == date) & (df['dataframe'] == df_name) if date is not None else (df['dataframe'] == df_name)
    row = index[condition].to_list()
    pipeline = df.iloc[int(row[0])]
    return [html.Div([
        html.H4(name),
        html.Div([set_body(name, pipeline[i])]), 
        html.Br()
    ]) for i, name in enumerate(col)]


# Function for managing the visualization of the pipelines
def show_hide_pipelines_function(store_pipelines_class, store_pipelines_reg, n1, n2, value, is_open):
    if n1 or n2:
        score, date, df_name = value.split('@')
        if score in ['acc', 'f1']:
            return not is_open, get_body_from_pipelines(store_pipelines_class, date, df_name)
        else:
            return not is_open, get_body_from_pipelines(store_pipelines_reg, date, df_name)
    return is_open, None

# Function for checking the correct structure of the Dataframe IDs sequence
def check_dfs_sequence(dfs_sequence):
    if dfs_sequence is None:
        raise PreventUpdate
    dfs = dfs_sequence.split(',')
    for df in dfs:
        try: int(df)
        except: return False
    return True