import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State



dash_app = dash.Dash(__name__)
server = dash_app.server
dash_app.config.suppress_callback_exceptions = True
