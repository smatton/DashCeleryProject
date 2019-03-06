import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
from appConfig import worker_log
import os
from apps.page1 import makeIframe
from app import dash_app
from mytasks import add



layout = html.Div([
    html.Div([
        html.H1("Pick two numbers to add"),
        dcc.Input(id='first-number', type='text'),
        dcc.Input(id='second-number', type='text'),
        html.H2("Run asynchronous or synchronous"),
        dcc.RadioItems(
            id='asynch-choice',
            options=[{"label": 'Asynchronous', "value": 'async'},
                     {"label": 'Synchronous', "value": 'sync'}],
            value='sync'

        )



    ],className='row'),
    html.Div([
        html.Button("Run", id='run-button'),
        html.Div(id='results-div', style={'display': 'none'}),
        html.Div(id='log-div')
    ],className='row')


], className='container')



@dash_app.callback(Output("results-div", "children"),
                   [Input('run-button', 'n_clicks')],
                    [State('asynch-choice','value'),
                     State('first-number','value'),
                     State('second-number','value')]
)
def runAdd(n_clicks, async_choice, a, b):
    if n_clicks is not None and a is not None and b is not None:
        if async_choice == 'async':
            #run
            add.delay(float(a), float(b))
            return 'asynch'
        else:
            #run synchronously

            return add(float(a), float(b))
    else:
        return None

@dash_app.callback(Output("log-div","children"),
                   [Input("results-div", 'children')])
def updateLogDiv(results):
    if results is not None:

        if results == 'asynch':

            return makeIframe(worker_log)
        else:
            return html.P(results)
    else:
        return None
