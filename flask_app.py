from __future__ import print_function
import os
import sys
import random
import time
import datetime
from collections import deque
from collections import OrderedDict
import numpy as np
import logging
import logzero
from logzero import logger
# import socket
import subprocess
import zmq

#*** to run this app you need pip install Flask and Flask-SocketIO and eventlet and ***#
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import json
from threading import Thread, Lock


logzero.loglevel(logging.INFO)
app_logger = logger

# Socket to talk to server
context = zmq.Context()
subscriber = context.socket(zmq.SUB)

subscriber.connect("tcp://localhost:5556")
subscriber_filter = ''

# Python 2 - ascii bytes to unicode str
if isinstance(subscriber_filter, bytes):
    subscriber_filter = subscriber_filter.decode('ascii')
subscriber.setsockopt_string(zmq.SUBSCRIBE, subscriber_filter)



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app)
thread = None

def background_thread():
    """send server generated events to clients."""
    while True:
        socketio.sleep(10)
        # string = subscriber.recv_string()
        string = '%.3f' % random.random()
        #data_string = update_graph(string)
        app_logger.info(string)
        datastr = json.loads(string)
        socketio.emit('new_data', datastr) #  ,namespace='/')

@socketio.on('connect')  # , namespace='/')
def test_connect():
    global thread
    if thread is None:
        thread =  socketio.start_background_task(target=background_thread)
    print('got to test_connect, thread started')
    # emit('my_response', {'data': 'Connected'}, namespace='/')
    socketio.emit('connect', {'data': 'connected to server'})  # , namespace='/')

@socketio.on('disconnect')  #, namespace='/')
def test_disconnect():
    print('Client disconnected', request.sid)

@socketio.on('my_event')  #, namespace='/')
def my_event():
    print('my_event', request.sid)

@app.route("/", methods=['GET', 'POST'])
def plot():
    # This is a terrible hack to enable logging and updating the plot
    num_curves = 1

    plotTitle = 'Flask socketio plotly test app'
    ndata = []
    for idx, name in enumerate(['graph']):
        ndata.append({'x': [], 'y': [], 'type': 'scatter',
                            'mode':'markers+lines', 'name':'%s' % name})

    # Updates the data for the table
    newdata , x = update_table()

    # Create a traces and layout to plot
    data=[]
    figure={}
    for j in range(num_curves):
        trace = go.Scattergl(x=ndata[j]['x'], y=ndata[j]['y'],name=ndata[j]['name'])
        data.append(trace)
    layout = go.Layout(title=plotTitle, \
        yaxis={'title':'Random', 'type':'linear'}, xaxis={'title':'Time'})

    # PlotlyJSONEncoder converts objects to their JSON equivalents
    figure=dict(data=data, layout=layout)
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # Add dropdown for fridge states

    # try to connect to clients
    test_connect()

    return render_template('webFridge.html', graphJSON=graphJSON, \
                    newdata=newdata, async_mode=socketio.async_mode)


def update_graph(data_string):
    sensor_names = fridge.get_sensor_names()
    data_string_list = data_string.split(',')
    x = datetime.datetime.fromtimestamp(float(data_string_list[0]))
    newdata = {}
    figure = []
    for i, name in enumerate(fridge.config['graph']):
        idx = sensor_names.index(name) + 2
        newdata['y']= float(data_string_list[idx])
        newdata['x'] = x
        newdata['name'] = name
        figure.append(newdata)
    return figure

def update_table():
    x = datetime.datetime.now()
    newdata = {'data1': random.random()}

    return newdata, x


if __name__ == "__main__":
    ip, port_ = '0.0.0.0', '45000'

    # Start Flask app
    socketio.run(app, host=ip, port=port_, use_reloader=True)
