#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import needed
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import os
os.environ['KAGGLE_USERNAME'] = "zullle" # Username for using the Kaggle API
os.environ['KAGGLE_KEY'] = "24df22da033e9547780e278280a6ae2b" # Key for using the Kaggle API

from webapp.frontend import sidebar, openmlbenchmark, kagglebenchmark, testbenchmark, get_pastresultopenml, get_pastresultkaggle
from webapp.backend import render_page_content_function, start_openml_function, start_kaggle_function, start_test_function, render_tab_content_function, collapse_alogrithms_options_function
from webapp.utils import get_store_past_bech_function, render_collapse_options, show_hide_pipelines_function, make_options, modify_dropdown_comparedf_function

def start():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    
    app.title = 'AutoML Benchmark'

    algorithms = ['autosklearn', 'h2o', 'tpot', 'autokeras', 'autogluon']

    # Style of the page
    CONTENT_STYLE = {
        "marginLeft": "22rem",
        "marginRight": "2rem",
        "padding": "2rem 1rem",
    }

    content = html.Div(id="page-content", style=CONTENT_STYLE)

    # Define the layout of the web application
    app.layout = html.Div([
        dcc.Location(id="url"), sidebar,
        dcc.Store(id="store_class_openml"), dcc.Store(id="store_reg_openml"),
        dcc.Store(id="store_class_kaggle"), dcc.Store(id="store_reg_kaggle"),
        dcc.Store(id="store_class_results_openml"), dcc.Store(id="store_reg_results_openml"),
        dcc.Store(id="store_class_results_kaggle"), dcc.Store(id="store_reg_results_kaggle"),
        dcc.Store(id="store_pipelines_class_openml"), dcc.Store(id="store_pipelines_reg_openml"),
        dcc.Store(id="store_pipelines_class_kaggle"), dcc.Store(id="store_pipelines_reg_kaggle"),
        dcc.Store(id="store_pipelines_results_class_openml"), dcc.Store(id="store_pipelines_results_reg_openml"),
        dcc.Store(id="store_pipelines_results_class_kaggle"), dcc.Store(id="store_pipelines_results_reg_kaggle"),
        content
    ])

    app.validation_layout=html.Div([openmlbenchmark, kagglebenchmark, testbenchmark, get_pastresultopenml(), get_pastresultkaggle()])

    # Callback for rendering sections
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname): return render_page_content_function(pathname)


    # Callback for populating the Store components and tables related to an OpenML Benchmark
    @app.callback(
        [Output('store_class_openml', 'data'), Output('store_reg_openml', 'data'), Output('store_pipelines_class_openml', 'data'), Output('store_pipelines_reg_openml', 'data'), Output('res-bench-openml-table-class', 'children'), Output('res-bench-openml-table-reg', 'children')],
        [Input('submit-openml', 'n_clicks')],
        [State("openmlbenchmark-tabs", "active_tab"), State("dfs-sequence", "value"), State('ndf', 'value'), State('nmore', 'value'), State("autosklearn-timelife", "value"), State("h2o-timelife", "value"), State("tpot-timelife", "value"), State("autokeras-timelife", "value"), State("autogluon-timelife", "value"),
        State("autosklearn-flag-rerun", "value"), State("h2o-flag-rerun", "value"), State("tpot-flag-rerun", "value"), State("autokeras-flag-rerun", "value"), State("autogluon-flag-rerun", "value")]
    )
    def start_openml(n_clicks, active_tab, dfs_squence, ndf, nmore, as_tl, h2o_tl, t_tl, ak_tl, ag_tl, as_f, h2o_f, t_f, ak_f, ag_f):
        return start_openml_function(active_tab, dfs_squence, ndf, nmore, make_options(as_tl, h2o_tl, t_tl, ak_tl, ag_tl, as_f, h2o_f, t_f, ak_f, ag_f))



    # Callback for populating the Store components and tables related to a Kaggle Benchmark
    @app.callback(
        [Output('store_class_kaggle', 'data'), Output('store_reg_kaggle', 'data'), Output('store_pipelines_class_kaggle', 'data'), Output('store_pipelines_reg_kaggle', 'data'), Output('res-bench-kaggle-table-class', 'children'), Output('res-bench-kaggle-table-reg', 'children')],
        [Input('submit-kaggle', 'n_clicks')],
        [State('kaggledataframe', 'value'), State("autosklearn-timelife", "value"), State("h2o-timelife", "value"), State("tpot-timelife", "value"), State("autokeras-timelife", "value"), State("autogluon-timelife", "value"),
        State("autosklearn-flag-rerun", "value"), State("h2o-flag-rerun", "value"), State("tpot-flag-rerun", "value"), State("autokeras-flag-rerun", "value"), State("autogluon-flag-rerun", "value")]
    )
    def start_kaggle(n_clicks, kaggledataframe, as_tl, h2o_tl, t_tl, ak_tl, ag_tl, as_f, h2o_f, t_f, ak_f, ag_f):
        return start_kaggle_function(kaggledataframe, make_options(as_tl, h2o_tl, t_tl, ak_tl, ag_tl, as_f, h2o_f, t_f, ak_f, ag_f))


    # Callback to run the Test Benchmark
    @app.callback(  
        [Output('res-bench-test', 'children')],
        [Input('submit-test', 'n_clicks')],
        [State('dfid', 'value'), State('algorithms', 'value'), State("autosklearn-timelife", "value"), State("h2o-timelife", "value"), State("tpot-timelife", "value"), State("autokeras-timelife", "value"), State("autogluon-timelife", "value"),
        State("autosklearn-flag-rerun", "value"), State("h2o-flag-rerun", "value"), State("tpot-flag-rerun", "value"), State("autokeras-flag-rerun", "value"), State("autogluon-flag-rerun", "value")]
    )
    def start_test(n_clicks, dfid, algorithms, as_tl, h2o_tl, t_tl, ak_tl, ag_tl, as_f, h2o_f, t_f, ak_f, ag_f):
        return start_test_function(dfid, algorithms, make_options(as_tl, h2o_tl, t_tl, ak_tl, ag_tl, as_f, h2o_f, t_f, ak_f, ag_f))


    # Populating the Stores needed to view the results of old OpenML Benchmarks
    @app.callback(
        [Output('store_class_results_openml', 'data'), Output('store_reg_results_openml', 'data'),Output('store_pipelines_results_class_openml', 'data'), Output('store_pipelines_results_reg_openml', 'data'), Output('result-past-bench-openml-table-class', 'children'), Output('result-past-bench-openml-table-reg', 'children')],
        [Input('submit-search-openml', 'n_clicks')],
        [State('pastresultopenml', 'value'), State('pastresultopenml-comapre', 'value')]
    )
    def get_store_past_bech_openml(n_clicks, timestamp, compare_with): return get_store_past_bech_function(timestamp, 'past-OpenML', compare_with)

    # Callback for changing the Dropdown of comparable markings
    @app.callback(
        [Output('pastresultopenml-comapre', 'options')],
        [Input('pastresultopenml', 'value'), Input('pastresultopenml-comapre', 'value')]
    )
    def modify_dropdown_comparedf(timestamp, comapre_list): return modify_dropdown_comparedf_function(timestamp, comapre_list, 'OpenML')



    # Populating the Stores needed to view the results of old Kaggle Benchmarks
    @app.callback(
        [Output('store_class_results_kaggle', 'data'), Output('store_reg_results_kaggle', 'data'),Output('store_pipelines_results_class_kaggle', 'data'), Output('store_pipelines_results_reg_kaggle', 'data'), Output('result-past-bench-kaggle-table-class', 'children'), Output('result-past-bench-kaggle-table-reg', 'children')],
        [Input('submit-search-kaggle', 'n_clicks')],
        [State('pastresultkaggle', 'value'), State('pastresultkaggle-comapre', 'value')]
    )
    def get_store_past_bech_kaggle(n_clicks, timestamp, compare_with): return get_store_past_bech_function(timestamp, 'past-Kaggle', compare_with)
    
    # Callback for changing the Dropdown of comparable markings
    @app.callback(
        [Output('pastresultkaggle-comapre', 'options')],
        [Input('pastresultkaggle', 'value'), Input('pastresultkaggle-comapre', 'value')]
    )
    def modify_dropdown_comparedf(timestamp, comapre_list): return modify_dropdown_comparedf_function(timestamp, comapre_list, 'Kaggle')


    # Callbacks related to chart refresh to display Kaggle and OpenML classification results
    @app.callback([Output('res-bench-openml-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_openml', 'data')])
    def render_tab_content_class(active_tab, store_class_openml): return render_tab_content_function(active_tab, store_class_openml, ('Accuracy', 'F1')) if store_class_openml is not None else [None]

    @app.callback([Output('res-bench-kaggle-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_kaggle', 'data')])
    def render_tab_content_class(active_tab, store_class_kaggle): return render_tab_content_function(active_tab, store_class_kaggle, ('Accuracy', 'F1')) if store_class_kaggle is not None else [None]

    @app.callback([Output('result-past-bench-openml-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_results_openml', 'data')])
    def render_tab_content_class(active_tab, store_class_results_openml): return render_tab_content_function(active_tab, store_class_results_openml, ('Accuracy', 'F1')) 

    @app.callback([Output('result-past-bench-kaggle-graph-class', 'children')], [Input("tabs-class", "active_tab"), Input('store_class_results_kaggle', 'data')])
    def render_tab_content_class(active_tab, store_class_results_kaggle): return render_tab_content_function(active_tab, store_class_results_kaggle, ('Accuracy', 'F1'))


    # Callbacks related to graph refresh to display Kaggle and OpenML regression results
    @app.callback([Output('res-bench-openml-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_openml', 'data')])
    def render_tab_content_reg(active_tab, store_reg_openml): return render_tab_content_function(active_tab, store_reg_openml, ('RMSE', 'R2')) if store_reg_openml is not None else [None]

    @app.callback([Output('res-bench-kaggle-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_kaggle', 'data')])
    def render_tab_content_reg(active_tab, store_reg_kaggle): return render_tab_content_function(active_tab, store_reg_kaggle, ('RMSE', 'R2')) if store_reg_kaggle is not None else [None]

    @app.callback([Output('result-past-bench-openml-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_results_openml', 'data')])
    def render_tab_content_reg(active_tab, store_reg_results_openml): return render_tab_content_function(active_tab, store_reg_results_openml, ('RMSE', 'R2'))

    @app.callback([Output('result-past-bench-kaggle-graph-reg', 'children')], [Input("tabs-reg", "active_tab"), Input('store_reg_results_kaggle', 'data')])
    def render_tab_content_reg(active_tab, store_reg_results_kaggle): return render_tab_content_function(active_tab, store_reg_results_kaggle, ('RMSE', 'R2'))

    # Callback for managing collapses related to algorithm options
    @app.callback(
        [Output(f"collapse-{algo}", "is_open") for algo in algorithms],
        [Input(f"{algo}-options", "n_clicks") for algo in algorithms],
        [State(f"collapse-{algo}", "is_open") for algo in algorithms],
    )
    def collapse_alogrithms_options(n1, n2, n3, n4, n5, is_open1, is_open2, is_open3, is_open4, is_open5): return collapse_alogrithms_options_function(n1, n2, n3, n4, n5, is_open1, is_open2, is_open3, is_open4, is_open5)

    @app.callback(
        [Output(f"{algo}-options", "disabled") for algo in algorithms],
        [Input("algorithms", "value")],
    )
    def disable_buttons_collapse(choice): return render_collapse_options(choice)

    # Callback for managing the display of the modal in which the pipelines of the algorithms related to that result are visible
    @app.callback(
        [Output({"type":"modal-Pipelines", "index": MATCH}, "is_open"), Output({"type": 'body-modal-Pipelines', "index": MATCH}, 'children')],
        [Input({"type": "open-Pipelines", "index": MATCH}, "n_clicks"), Input({"type": "close-modal-Pipelines", "index": MATCH}, "n_clicks"), Input({"type": "open-Pipelines", "index": MATCH}, "value"), Input("url", "pathname"),
        Input('store_pipelines_class_openml', 'data'), Input('store_pipelines_reg_openml', 'data'),
        Input('store_pipelines_class_kaggle', 'data'), Input('store_pipelines_reg_kaggle', 'data'),
        Input('store_pipelines_results_class_openml', 'data'), Input('store_pipelines_results_reg_openml', 'data'),
        Input('store_pipelines_results_class_kaggle', 'data'), Input('store_pipelines_results_reg_kaggle', 'data'),],
        [State({"type":"modal-Pipelines", "index": MATCH}, "is_open")]
    )
    def show_hide_pipelines(n1, n2, value, path, s1, s2, s3, s4, s5, s6, s7, s8, is_open): 
        stores = {
            "/openml": [s1,s2],
            "/kaggle": [s3,s4],
            '/results-openml': [s5,s6],
            '/results-kaggle': [s7,s8],
        }
        s = stores.get(path)
        if s is not None:
            return show_hide_pipelines_function(s[0], s[1], n1, n2, value, is_open)
        else:
            return None, None

    app.run_server(debug=True)
    
    

if __name__ == '__main__':
    start() # Starting the application
    
