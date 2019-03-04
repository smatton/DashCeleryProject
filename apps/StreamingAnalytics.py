import dash
import pickle
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import codecs
import json
import plotly.graph_objs as go
import dash_table as dt
from dash.dependencies import Input, Output, State
import os
import time
import pandas as pd
from appConfig import model_folder
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import f1_score
from sklearn import svm
from app import dash_app
from mytasks import trainModel



def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}

def Card(children, **kwargs):
    return html.Section(
        children,
        style=_merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
        }, kwargs.get('style', {})),
        **_omit(['style'], kwargs)
)
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

def makeDrift(startcenters=[[2, 2], [-2, -2]], endcenters=[[-2, 2], [2, -2]], n_per_drift=25, n_total=500,
              outliers_fraction=0.15):
    n_outliers = int(outliers_fraction * n_per_drift)
    n_inliers = n_per_drift - n_outliers

    n_steps = int(n_total / n_inliers)

    centers_list = []

    if len(startcenters) == len(endcenters):
        num_startcenters = len(startcenters)
        if sum([len(x) for x in startcenters]) / num_startcenters == len(startcenters[0]):
            start_center_dims = len(startcenters[0])
        else:
            sys.exit("dims across startcenters not consistent")
        num_endcenters = len(endcenters)
        if sum([len(x) for x in endcenters]) / num_endcenters == len(endcenters[0]):
            end_center_dims = len(endcenters[0])
        else:
            sys.exit("dims across endcenters not consistent")

        if end_center_dims == start_center_dims:

            tmp_centers = np.empty((n_steps, start_center_dims), dtype=np.float64)
        else:
            sys.exit("dims between startcenters and endcenters not consistent")

        for i, s_cent in enumerate(startcenters):
            for j, s_dim in enumerate(s_cent):
                if s_dim - endcenters[i][j] == 0:
                    tmp_centers[:, j] = s_dim
                else:
                    tmp_centers[:, j] = np.linspace(s_dim, endcenters[i][j], n_steps)

            centers_list.append([list(x) for x in tmp_centers])
    else:
        print("start and end dims not the same")
    blobs_params = dict(n_samples=n_inliers, n_features=start_center_dims)
    np.random.seed(0)
    k = 0
    for z in range(n_steps):
        tmp_blob = make_blobs(centers=[centers_list[0][z], centers_list[1][z]], cluster_std=[0.5, 0.5],
                              **blobs_params)[0]
        inlier_lab = np.repeat(1, tmp_blob.shape[0]).astype(np.float64)
        outlier_lab = np.repeat(-1, n_outliers).astype(np.float64)
        labs = np.concatenate([inlier_lab, outlier_lab], axis=0)
        labs = labs.reshape((labs.shape[0], 1))
        X = np.concatenate([tmp_blob, np.random.uniform(low=-6, high=6,
                                                  size=(n_outliers, 2))], axis=0)
        X = np.hstack([X, labs])
        np.random.shuffle(X)
        if k == 0:
            dataset = X.copy()
        else:
            dataset = np.concatenate([dataset, X], axis=0)
        k += 1

    return dataset

dataset = makeDrift(n_total=50)



layout = html.Div(children=[
    html.H1(id="model-update-info"),
    dcc.Interval(id='interval-update', n_intervals=0),
    html.Div(id='model-div', style={'display': 'none'}),
    html.Div(id='f1-score'),
    html.Div(children=[
        html.Div(id='live-update-text'),
        html.Div([
        dcc.Graph(id='live-update-graph')
        ], className='six columns'),
        html.Div([
        Card([
        dcc.Dropdown(
                id='dropdown-interval-control',
                options=[
                    {'label': 'No Updates', 'value': 'no'},
                    {'label': 'Slow Updates', 'value': 'slow'},
                    {'label': 'Regular Updates', 'value': 'regular'},
                    {'label': 'Fast Updates', 'value': 'fast'}
                ],
                value='no',
                clearable=False,
                searchable=False
        ),
            dcc.RadioItems(
                id = 'f1-threshold',
                options=[
                    {'label': 'F1 Threshold 90%', 'value': 0.90},
                    {'label': 'F1 Threshold 80%', 'value': 0.80},
                    {'label': 'F1 Threshold 60%', 'value': 0.60}
                ],
                value=0.90
            )
        ], className='three columns'),
        html.Div(
        dcc.Slider(
            id='window-size',
            min=-0,
            max=25,
            marks={i: '{} Recent'.format(i) for i in range(26)},
            step=1,
            value=10,
        ))])

    ], className='row')

], className='row')


@dash_app.callback(Output("f1-score", "children"),
        [Input('interval-update', 'n_intervals'), Input('window-size', 'value')],
    [State('model-div', 'children')])
def reportF1(n, window, model):

    if model is not None and n is not None:
        with open(model, 'rb') as file:
            s = pickle.load(file)
        if window == 0:
            window_data = dataset[:n, :]
        else:
            if n - window < 0:
                window_data = dataset[0:n, :]
            else:
                window_data = dataset[n - window:n, :]

        y_pred = s.predict(window_data[:, :2])

        acc = f1_score(window_data[:,2], y_pred)
    else:
        acc = 0

    return acc

@dash_app.callback(Output("model-update-info", "children"),
                   [Input("f1-score", "children")],
                   [State('f1-threshold', 'value'),
                    State('interval-update', 'n_intervals'),
                    State('window-size', 'value'),
                    State('model-update-info', "children")]
)
def runModel(score, threshold, n_interval, window, train_state):
    print("runmodel", train_state)
    if train_state is not None:
        state_list = train_state.split()
        if score < threshold and n_interval > 0:

            if len(state_list) > 0:
                if state_list[0] == 'none' or state_list[0] == 'done':
                    if window == 0:
                        window_data = dataset[:n_interval, :]
                    else:
                        if n_interval - window < 0:
                            window_data = dataset[0:n_interval, :]
                        else:
                            window_data = dataset[n_interval - window:n_interval, :]

                    b = window_data.tolist()
                    data_string = json.dumps(b)
                    trainModel.delay(data_string, 0.30, str(n_interval))
                    return "training svm_model_{}.pkl".format(n_interval)
                elif state_list[0] == 'training':
                    return train_state
                else:
                    return 'none'
            else:
                return 'none'
        elif n_interval > 0:
            # check state
            if state_list[0] == "training":
                models_in_folder = os.listdir(model_folder)
                if state_list[1] in models_in_folder:
                    return "done {}".format(state_list[1])
                else:
                    return train_state
            elif state_list[0] == 'done':
                return train_state
            else:
                return "none"
        else:
            return "none"
    else:
        return "none"
@dash_app.callback(
    Output('live-update-graph', 'figure'),
    [Input('interval-update', 'n_intervals'), Input('window-size', 'value')],
    [State('model-div', 'children')]
)
def updatePredictionPlot(n, window, model, dataset = dataset):

    if model is None or n is None:
        return None
    if n > dataset.shape[0]:
        return None

    if window == 0:
        window_data = dataset[:n, :]
    else:
        if n-window < 0:
            window_data = dataset[0:n, :]
        else:
            window_data = dataset[n-window:n, :]



    print("I load model, {}".format(model))
    with open(model, 'rb') as file:
        s = pickle.load(file)

    #y_pred = s.predict(window_data[:, :2])

    Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    red = 'rgb(244, 66, 66)'
    blue = 'rgb(17,157, 255)'
    colors = [blue if x == 1 else red for x in window_data[:, 2]]

    trace0 = go.Contour(
        x=np.linspace(-7, 7, 150),
        y=np.linspace(-7, 7, 150),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo='none',
        contours=dict(
            showlines=False,
            type='constraint',
            operation='=',
            #value=scaled_threshold,
        ),

        line=dict(
            color='#222222'
        )
    )
    trace1 = go.Scatter(
        x=window_data[:, 0],
        y=window_data[:, 1],
        marker=dict(color=colors),
        mode='markers'
    )

    data = [trace0, trace1]

    layout = go.Layout(

        xaxis=dict(
            # scaleanchor="y",
            # scaleratio=1,
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        hovermode='closest',
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),

    )
    fig = go.Figure(data=data, layout=layout)
    return fig

@dash_app.callback(Output('live-update-text', 'children'),
              [Input('interval-update', 'n_intervals')])
def update_metrics(n):
    print('{}'.format(str(n)))
    return html.P('{}'.format(str(n)))

@dash_app.callback(Output('interval-update', 'interval'),
              [Input('dropdown-interval-control', 'value')])
def update_interval_log_update(interval_rate):
    if interval_rate == 'fast':
        return 200

    elif interval_rate == 'regular':
        return 500

    elif interval_rate == 'slow':
        return 750

    # Refreshes every 24 hours
    elif interval_rate == 'no':
        return 24 * 60 * 60 * 1000

@dash_app.callback(Output('model-div', 'children'),
               [Input('interval-update', 'n_interval')]
                   ,[State('model-update-info', 'children')]
                )
def dumpModel(n_interval, train_state):
    print("dumpModel", train_state)
    if n_interval is None:
        print("I initialized model")
        # initialize model with init data 15% outliers
        blobs_params = dict(n_samples=170, n_features=2)
        tmp_blob = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
                              **blobs_params)[0]
        inlier_lab = np.repeat(1, tmp_blob.shape[0]).astype(np.float64)
        outlier_lab = np.repeat(-1, 30).astype(np.float64)
        labs = np.concatenate([inlier_lab, outlier_lab], axis=0)
        labs = labs.reshape((labs.shape[0], 1))
        x = np.concatenate([tmp_blob, np.random.uniform(low=-6, high=6,
                                                        size=(30, 2))], axis=0)
        x = np.hstack([x, labs])
        np.random.shuffle(x)

        s = svm.OneClassSVM(nu=0.15, kernel="rbf",
                            gamma=0.1)
        s.fit(x[:,:2])
        model_file = "svm_model.pkl"
        model_path = os.path.join(model_folder, model_file)
        with open(model_path, 'wb') as file:
            pickle.dump(s, file)
    elif train_state.split()[0] == "done":
        print("I am loading finished model")
        model_file = "{}".format(train_state.split()[1])
        model_path = os.path.join(model_folder, model_file)
        print(model_path)
        return model_path
    else:
        return None
    return model_path


