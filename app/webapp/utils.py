import os
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import pandas as pd

def get_lisd_dir(test):
    lis = (os.listdir('./results/'+test))
    dropdown = []
    for l in lis:
        if l != '.gitignore':
            dropdown.append({'label': l, 'value': l})
    return dropdown

def get_pipelines_button(dfs, task):
    ret = []
    for index, row in dfs.iterrows():
        ret.append([dbc.Modal(
                [
                    dbc.ModalHeader("Pipelines"),
                    dbc.ModalBody(id={
                                'type': "body-modal-Pipelines",
                                'index': task+'-'+str(row['dataset'])
                            }),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id={
                                'type': "close-modal-Pipelines",
                                'index': task+'-'+str(row['dataset'])
                            }, className="ml-auto", n_clicks=0
                        )
                    ),
                ],
                id={
                    'type': "modal-Pipelines",
                    'index': task+'-'+str(row['dataset'])
                },
                size="lg",
                is_open=False,
            ),html.Div([dbc.Button("Pipelines", id={
                'type': "open-Pipelines",
                'index': task+'-'+str(row['dataset'])
            }, value=task+'-'+str(row['dataset']), className="mr-1", n_clicks=0)])])
    return ret

def retrun_graph_table(dfs, pipelines, title, task, t, opts, scores):
    scatters = []
    histos = []
    table = [html.H3(title)]
    if(dfs[0] is None or dfs[1] is None):
        return {'scatter_'+scores[0]: None, 'histo_'+scores[0]: None, 'scatter_'+scores[1]: None, 'histo_'+scores[1]: None, 'options': None}, None, dbc.Tabs( [], id="tabs-class", active_tab="", style={'hidden':'true'} )
    else:
        for df in dfs:
            df['pipelines'] = get_pipelines_button(df[['dataset']], df.columns[1].split('-')[1])
            table.append(dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True))
            for col in df.columns[1:-1]:
                scatters.append(go.Scatter(x=df['dataset'], y=df[col], name=col.split('-')[0], mode='lines+markers'))
                histos.append(go.Bar(x=df['dataset'], y=df[col], name=col.split('-')[0]))
        

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

        opts = opts.to_dict()
        options = [
            html.Div([
                html.P(["Running time for Autosklearn: " + str(opts['autosklearn'][0]) + " minute/s"]),
                html.P(["Running time for TPOT: " + str(opts['tpot'][0]) + " generation/s"]),
                html.P(["Running time for H2O: " + str(opts['h2o'][0]) + " minute/s"]),
                html.P(["Running time for AutoKeras: " + str(opts['autokeras'][0]) + " epoch/s"]),
                html.P(["Running time for AutoGluon: " + str(opts['autogluon'][0]) + " minute/s"]),
            ])
        ]

        
        limit = 5 if t == 'OpenML' else 6 # 6 perchè c'è il leader per i kaggle
        return {
            'scatter_'+scores[0]: scatters[:limit], 'histo_'+scores[0]: histos[:limit], 'scatter_'+scores[1]: scatters[limit:], 'histo_'+scores[1]: histos[limit:], 'options': options
        }, pipelines, table


def get_store_and_tables(dfs, type):
    res_class_acc, res_class_f1, res_reg_rmse, res_reg_r2, pipelines_class, pipelines_reg, options = dfs # scomposizione
    store_dict= { 'class': {}, 'reg': {} }
    store_pipelines = { 'class': {}, 'reg': {} }
    tables = [[None], [None]]

    store_dict['class'], store_pipelines['class'], tables[0] = retrun_graph_table([res_class_acc, res_class_f1], pipelines_class, 'Classification Results', 'class', type, options, ('acc', 'f1'))
    store_dict['reg'], store_pipelines['reg'], tables[1] = retrun_graph_table([res_reg_rmse, res_reg_r2], pipelines_reg, 'Regression Results', 'reg', type, options, ('rmse', 'r2'))

    return store_dict['class'], store_dict['reg'], store_pipelines['class'], store_pipelines['reg'], tables[0], tables[1]


def get_store_past_bech_function(timestamp, type):
    if timestamp is not None:
        dfs = []
        scores = [('classification','acc'), ('classification','f1_score'), ('regression','rmse'), ('regression','r2_score')]
        for score in scores:
            #print('./results/'+ type +'/'+timestamp+'/'+ score[0] +'/'+ score[1] +'.csv')
            if os.path.exists('./results/'+ type +'/'+timestamp+'/'+ str(score[0]) +'/'+ str(score[1]) +'.csv'):
                dfs.append(pd.read_csv('./results/'+ type +'/'+timestamp+'/'+ str(score[0]) +'/'+ str(score[1]) +'.csv'))
            else:
                dfs.append(None)
        for t in ('classification', 'regression'):
            dfs.append(pd.read_csv('./results/'+ type +'/'+timestamp+'/'+ t + '/pipelines.csv', sep='@').to_dict())
        dfs.append(pd.read_csv('./results/'+ type +'/'+timestamp+'/options.csv'))
        return get_store_and_tables(dfs, type)
    else:
        raise PreventUpdate



def render_tab_content(active_tab, data, type): #pathname
    if active_tab and data is not None:
        if active_tab == "scatter":
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[0]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[0]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['scatter_'+type[1]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[1]))))),
                                ], align="center"
                            )
                        )]
        elif active_tab == "histogram":
            return [html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[0]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[0]))))),
                                    dbc.Col(dcc.Graph(figure=go.Figure(data=data['histo_'+type[1]], layout=go.Layout(xaxis = dict(title = 'Datasets'), yaxis = dict(title = type[1]))))),
                                ], align="center"
                            )
                        )]
        else:
            return [
                data['options']
            ]

    return "No tab selected"


def create_collapse(algo, measure, min, disabled):
    return dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H2(
                                            dbc.Button(
                                                algo + " Options",
                                                color="link",
                                                id=algo.lower()+"-options",
                                                disabled=disabled
                                            )
                                        )
                                    ),
                                    dbc.Collapse(
                                        dbc.CardBody([
                                            dbc.FormGroup([
                                                dbc.Label("Runnung time in "+measure,  width=5),
                                                dbc.Col([
                                                    dbc.InputGroup([
                                                        dbc.Input( id=algo.lower()+"-timelife", type="number", value=min, placeholder=measure, min=min, max=100000),
                                                        dbc.InputGroupAddon("at least " + str(min), addon_type="prepend")]
                                                    ),
                                                ], width=5),
                                            ],row=True),
                                        ]), id="collapse-"+algo.lower()
                                    ),
                                ]
                            )
def render_collapse_options(choice):
    return {
        'autosklearn': [False, True, True, True, True],
        'h2o': [True, False, True, True, True],
        'tpot': [True, True, False, True, True],
        'autokeras': [True, True, True, False, True],
        'autogluon':[True, True, True, True, False],
        'all': [False, False, False, False, False],
    }.get(choice)


def get_body_for_modal(pipeline, df_name):
    body = []
    df = pd.DataFrame.from_dict(pipeline)
    print('sono su get_body_for_modal')
    print(df)
    print('diodio')
    print(df_name)
    col = df.columns
    index = df.index
    condition = df['dataset'] == df_name
    row = index[condition].tolist()
    print(condition, row)
    pipeline = df.iloc[int(row[0])]
    for i, name in enumerate(col[0:]):
        body.append(html.Div([
        html.H4(name),
        html.P(pipeline[i])
    ]))
    return body


def show_hide_pipelines_function(store_pipelines_class, store_pipelines_reg, n1, n2, value, is_open):
        if n1 or n2:
            score = value.split('-')[0]
            df_name = value.split('-')[1]
            print('ho splittato sta roba: ' + value)
            if score in ['acc', 'f1']:
                return not is_open, get_body_for_modal(store_pipelines_class, df_name)
            else:
                return not is_open, get_body_for_modal(store_pipelines_reg, df_name)
        return is_open, None


def make_options(as_tl, h2o_tl, t_tl, ak_tl, ag_tl):
    return {
            'autosklearn': as_tl,
            'h2o': h2o_tl,
            'tpot': t_tl,
            'autokeras': ak_tl,
            'autogluon': ag_tl 
        }