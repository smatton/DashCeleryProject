import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output
import os

from app import dash_app
import pandas as pd
from appConfig import sasfile_folder, cas_server_types

#sas_files = os.listdir(sasfile_folder)
sas_files = os.listdir()

server_options = [{'label': x, 'value': x} for x in cas_server_types]



# Define your layout for the page, this gets imported in index.py to render the
# page using the layout
layout = html.Div([
    html.Div([
    dcc.Dropdown(
        options=server_options,
        value='laxno'
    ),
    dcc.Checklist(
        options=server_options,
        values=['laxno']
    ),
    html.Iframe(sandbox='', srcDoc="in the Iframe"),

    ],className='row'),
    html.Button('Submit', id='submit-job'),
    html.Button('Refresh Results', id='refresh-button'),
    html.Div(id='table-dropdown-container')
],className='row')
