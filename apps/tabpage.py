import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output
import os

from app import dash_app
from apps import tabOpps, tabLeads, tabNew
from apps import tabLeads

import pandas as pd


layout =html.Div([

            html.Div([
                dcc.Tabs(
                id="tabs",
                style={"height": "20", "verticalAlign": "middle"},
                children=[
                    dcc.Tab(label="Opportunities", value="opportunities_tab"),
                    dcc.Tab(label="Leads", value="leads_tab"),
                    dcc.Tab(id="cases_tab",label="Cases", value="cases_tab"),
                ],
                value="leads_tab",
            )], className="row tabs_div"),

            html.Div(id="tab_content", className="row", style={"margin": "2% 3%"}),





        ], className='row', style={'margin': '0%'})


@dash_app.callback(Output("tab_content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "opportunities_tab":
        return tabOpps.layout
    elif tab == "cases_tab":
        return tabLeads.layout
    elif tab == "leads_tab":
        return tabNew.layout
    else:
        return tabOpps.layout