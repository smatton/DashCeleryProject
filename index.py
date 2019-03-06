import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
import os
from app import dash_app, server
from apps import page1, tabpage

dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # header
    html.Div([

        html.Span("APP TITLE", className='app-title'),
        # can use the following to add image to your banner
        # html.Div([
        #     html.Img(src='', height="100%")], style={'float':"right", 'height':"100%"}, className="row header"),
     ], className="row header"),
    html.Div(id='page-content', className='row', style={"margin": "2% 3%"}),

    html.Link(href="https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css", rel="stylesheet"),

    # this needs to be imported in the index.py for multi-page app to load dependencies
    html.Div(dt.DataTable(data=[{}]), style={'display': 'none'})

], className='row', style={"margin": "0%"})


@dash_app.callback(Output('page-content', 'children'),
                   [Input('url', 'pathname')])
def display_page(pathname):
    if pathname in ['/', '/index']:
        return page1.layout
    elif pathname == '/tabpage':
        return tabpage.layout
    else:
        return '404'


if __name__ == '__main__':
    dash_app.run_server(debug=True)
