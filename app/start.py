#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import os
os.environ['KAGGLE_USERNAME'] = "zullle" # username from the json file
os.environ['KAGGLE_KEY'] = "24df22da033e9547780e278280a6ae2b" # key from the json file

from webapp.frontend import sidebar, openmlbenchmark, kagglebenchmark, testbenchmark, pastresultopenml, pastresultkaggle
from webapp.backend import render_page_content_function, start_openml_function, start_kaggle_function, start_test_function, render_tab_content_function, collapse_alogrithms_options_function
from webapp.utils import get_store_past_bech_function 


def start():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    #server = app.server

    algorithms = ['autosklearn', 'h2o', 'tpot', 'autokeras', 'autogluon']

    CONTENT_STYLE = {
        "marginLeft": "22rem",
        "marginRight": "2rem",
        "padding": "2rem 1rem",
    }

    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div([
        dcc.Location(id="url"), sidebar,
        dcc.Store(id="store_class_openml"), dcc.Store(id="store_reg_openml"),
        dcc.Store(id="store_class_kaggle"), dcc.Store(id="store_reg_kaggle"),
        dcc.Store(id="store_class_results_openml"), dcc.Store(id="store_reg_results_openml"),
        dcc.Store(id="store_class_results_kaggle"), dcc.Store(id="store_reg_results_kaggle"),
        content
    ])

    app.validation_layout=html.Div([openmlbenchmark, kagglebenchmark, testbenchmark, pastresultopenml, pastresultkaggle])


    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        return render_page_content_function(pathname)


    #populiamo i due store
    @app.callback(
        [Output('store_class_openml', 'data'), Output('store_reg_openml', 'data'), Output('res-bench-openml-table-class', 'children'), Output('res-bench-openml-table-reg', 'children')],
        [Input('submit-openml', 'n_clicks')],
        [State('nmore', 'value'), State('ndf', 'value'), State("autosklearn-timelife", "value"), State("h2o-timelife", "value"), State("tpot-timelife", "value"), State("autokeras-timelife", "value"), State("autogluon-timelife", "value")]
    )
    def start_openml(n_clicks, nmore, ndf, as_tl, h2o_tl, t_tl, ak_tl, ag_tl):
        options = {
            'autosklearn': as_tl,
            'h2o': h2o_tl,
            'tpot': t_tl,
            'autokeras': ak_tl,
            'autogluon': ag_tl 
        }
        return start_openml_function(nmore, ndf, options)



    #populiamo i due store
    @app.callback(
        [Output('store_class_kaggle', 'data'), Output('store_reg_kaggle', 'data'), Output('res-bench-kaggle-table-class', 'children'), Output('res-bench-kaggle-table-reg', 'children')],
        [Input('submit-kaggle', 'n_clicks')],
        [State('kaggledataset', 'value'), State("autosklearn-timelife", "value"), State("h2o-timelife", "value"), State("tpot-timelife", "value"), State("autokeras-timelife", "value"), State("autogluon-timelife", "value")]
    )
    def start_kaggle(n_clicks, kaggledataset, as_tl, h2o_tl, t_tl, ak_tl, ag_tl):
        options = {
            'autosklearn': as_tl,
            'h2o': h2o_tl,
            'tpot': t_tl,
            'autokeras': ak_tl,
            'autogluon': ag_tl 
        }
        return start_kaggle_function(kaggledataset, options)


    @app.callback(
        [Output('res-bench-test', 'children')],
        [Input('submit-test', 'n_clicks')],
        [State('dfid', 'value'), State('algorithms', 'value')]
    )
    def start_test(n_clicks, dfid, algorithms):
        return start_test_function(dfid, algorithms)



#qui aggiorno i store di class e reg e stampo inizialmente le tabelle con i relativi risultati
#OPNEML
    @app.callback(
        [Output('store_class_results_openml', 'data'), Output('store_reg_results_openml', 'data'), Output('result-past-bench-openml-table-class', 'children'), Output('result-past-bench-openml-table-reg', 'children')],
        [Input('pastresultopenml', 'value')]
    )
    def get_store_past_bech_openml(timestamp):
        return get_store_past_bech_function(timestamp, 'OpenML')

#KAGGLE
    @app.callback(
        [Output('store_class_results_kaggle', 'data'), Output('store_reg_results_kaggle', 'data'), Output('result-past-bench-kaggle-table-class', 'children'), Output('result-past-bench-kaggle-table-reg', 'children')],
        [Input('pastresultkaggle', 'value')]
    )
    def get_store_past_bech_kaggle(timestamp):
        return get_store_past_bech_function(timestamp, 'Kaggle')


#modfico stra scatter e histogram i risultati di classificazione
    @app.callback([Output('res-bench-openml-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_openml', 'data')])
    def render_tab_content_class(active_tab, store_class_openml):
        return render_tab_content_function(active_tab, store_class_openml, ('class_acc', 'class_f1'))

    @app.callback([Output('res-bench-kaggle-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_kaggle', 'data')])
    def render_tab_content_class(active_tab, store_class_kaggle):
        return render_tab_content_function(active_tab, store_class_kaggle, ('class_acc', 'class_f1'))

    @app.callback([Output('result-past-bench-openml-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_results_openml', 'data')])
    def render_tab_content_class(active_tab, store_class_results_openml):
        return render_tab_content_function(active_tab, store_class_results_openml, ('class_acc', 'class_f1'))

    @app.callback([Output('result-past-bench-kaggle-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_results_kaggle', 'data')])
    def render_tab_content_class(active_tab, store_class_results_kaggle):
        return render_tab_content_function(active_tab, store_class_results_kaggle, ('class_acc', 'class_f1'))


#modfico stra scatter e histogram i risultati di regressione
    @app.callback([Output('res-bench-openml-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_openml', 'data')])
    def render_tab_content_reg(active_tab, store_reg_openml):
        return render_tab_content_function(active_tab, store_reg_openml, ('reg_rmse', 'reg_r2'))


    @app.callback([Output('res-bench-kaggle-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_kaggle', 'data')])
    def render_tab_content_reg(active_tab, store_reg_kaggle):
        return render_tab_content_function(active_tab, store_reg_kaggle, ('reg_rmse', 'reg_r2'))

    @app.callback([Output('result-past-bench-openml-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_results_openml', 'data')])
    def render_tab_content_reg(active_tab, store_reg_results_openml):
        return render_tab_content_function(active_tab, store_reg_results_openml, ('reg_rmse', 'reg_r2'))


    @app.callback([Output('result-past-bench-kaggle-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_results_kaggle', 'data')])
    def render_tab_content_reg(active_tab, store_reg_results_kaggle):
        return render_tab_content_function(active_tab, store_reg_results_kaggle, ('reg_rmse', 'reg_r2'))

    @app.callback(
        [Output(f"collapse-{algo}", "is_open") for algo in algorithms],
        [Input(f"{algo}-options", "n_clicks") for algo in algorithms],
        [State(f"collapse-{algo}", "is_open") for algo in algorithms],
    )
    def collapse_alogrithms_options(n1, n2, n3, n4, n5, is_open1, is_open2, is_open3, is_open4, is_open5):
        return collapse_alogrithms_options_function(n1, n2, n3, n4, n5, is_open1, is_open2, is_open3, is_open4, is_open5)


    '''host='0.0.0.0', port=8050, '''
    app.run_server(host='0.0.0.0', port=8050, debug=True)

if __name__ == '__main__':
    start()
    