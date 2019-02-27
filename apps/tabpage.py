import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output
import os

from app import dash_app
from apps import Tab1, Tab2, StreamingAnalytics


import pandas as pd


layout =html.Div([

            html.Div([
                dcc.Tabs(
                id="tabs",
                style={"height": "20", "verticalAlign": "middle"},
                children=[
                    dcc.Tab(label="Tab1", value="Tab1_tab"),
                    dcc.Tab(label="Tab2", value="Tab2_tab"),
                    dcc.Tab(label="Streaming Analytics", value="StreamingAnalytics_tab"),
                ],
                value="StreamingAnalytics_tab",
            )], className="row tabs_div"),

            html.Div(id="tab_content", className="row", style={"margin": "2% 3%"}),





        ], className='row', style={'margin': '0%'})


@dash_app.callback(Output("tab_content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "Tab1_tab":
        return Tab1.layout
    elif tab == "Tab2_tab":
        return Tab2.layout
    elif tab == "StreamingAnalytics_tab":
        return StreamingAnalytics.layout
    else:
        return Tab1.layout