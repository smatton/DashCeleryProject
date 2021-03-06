import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output
import os
from app import dash_app
from appConfig import file_folder, cas_server_types


server_options = [{'label': x, 'value': x} for x in cas_server_types]

def getFiles():
    files = os.listdir(file_folder)
    file_options = [{'label': x, 'value': x} for x in files]

    return file_options

# Define your layout for the page, this gets imported in index.py to render the
# page using the layout
def makeIframe(file_name):

    if file_name is not None:
        f = os.path.join(file_folder, file_name)
        _, extension = os.path.splitext(f)
        message1 = ''
        with open(f) as file1:
            for line in file1:
                if extension.lower() == '.xml':
                    line = line.replace("<", "&lt").replace(">", "&gt")
                message1 = message1 + line + '<BR>'


        return html.Iframe(srcDoc=message1, style={'width':"50%", "height":400})
    else:
        return None

layout = html.Div([
    html.Div([
    dcc.Dropdown(
        id='file_drop',
        options=getFiles(),

    ),
    dcc.Checklist(
        options=server_options,
        values=['laxno']
    ),
    html.Div(id='iframe-div', className='six columns'),


    ], className='row'),
    html.Div(children=[
    html.Button('Submit', id='submit-job'),
    html.Button('Refresh Results', id='refresh-button'),
    dcc.Link('tabpages', href='/tabpage'),
    html.Div(id='table-dropdown-container')], className='row'
    )
],className='row')


@dash_app.callback(
    Output('iframe-div', 'children'),
    [Input('file_drop', 'value')]
)

def updateIframe(file_name):

    return makeIframe(file_name)


@dash_app.callback(
    Output('file_drop', 'options'),
    [Input('refresh-button', 'n_clicks')]
)
def updateFiles(n_clicks):
    print(n_clicks)
    return getFiles()
