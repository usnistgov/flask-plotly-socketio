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

try:
    from eventlet import monkey_patch as monkey_patch
    monkey_patch()
except ImportError:
    try:
        from gevent.monkey import patch_all
        patch_all()
    except ImportError:
        pass

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

# Socket to talk to subscribe to temperatures
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5556")
subscriber_filter = ''
# Python 2 - ascii bytes to unicode str
if isinstance(subscriber_filter, bytes):
    subscriber_filter = subscriber_filter.decode('ascii')
subscriber.setsockopt_string(zmq.SUBSCRIBE, subscriber_filter)

poller = zmq.Poller()
poller.register(subscriber, zmq.POLLIN)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, async_mode=None)
thread = None

with open('2017-10-10-12-19-14.csv', 'r') as f:
    labels = f.readline()
    # labels = labels[1:]
    labels = labels.split(',')
    labels = [label.strip() for label in labels]
    print('number of label', len(labels))
for l in labels:
    print(len(l), l, type(l))

graph = ['40K', '4K', '1K', 'switch', 'pump']
def background_thread():
    global labels
    """send server generated events to clients."""
    socketio.sleep(1)
    while True:
        if True:
            socketio.sleep(0)
            socks = dict(poller.poll())
            # print('poll socks', socks)
            if subscriber in socks:
                data_string = subscriber.recv_string()
                # print('subscriber message:', data_string)
                data = data_string.split(',')
                # print('len of data', len(data))
        else:
            socketio.sleep(10)
            data_string = '%.2f, %f' % (time.time(), random.random())
            data = data_string.split(',')
        # app_logger.info(data_string)
        x = datetime.datetime.fromtimestamp(float(data[0]))
        # x = datetime.datetime.now()
        x = x.strftime('%Y-%m-%d %H:%M:%S')
        timestamp = x
        y = []
        x = []
        #for i, name in enumerate(fridge.config['graph']):
            # idx = self.sensor_names.index(name) + 1
            # value = float(data_string_list[idx])
        for i, name in enumerate(graph):
            idx = labels.index(name)
            value = float(data[idx])
            if np.isnan(value):
                value = -1
            y.append([value])
            x.append([timestamp])
        # print(x,y)
        datastr = {'x': x, 'y': y}
        # print('datastr', datastr)
        socketio.emit('new_data', datastr) #  ,namespace='/')

@socketio.on('connect', namespace='/')
def test_connect():
    global thread
    if thread is None:
        thread =  socketio.start_background_task(target=background_thread)
        print('got to test_connect, thread started')
    print('passing sensor_names to client')
    # emit('my_response', {'data': 'Connected'}, namespace='/')
    socketio.emit('connect', graph)  # , namespace='/')

@socketio.on('disconnect')  #, namespace='/')
def test_disconnect():
    print('Client disconnected', request.sid)

@socketio.on('my_event')  #, namespace='/')
def my_event(message):
    print('my_event', request.sid)
    print('message', message)

@app.route("/", methods=['GET', 'POST'])
def plot():
    plotTitle = 'Flask socketio plotly test app'

    # Updates the data for the table
    newdata = load_data(graph)
    print(newdata)
    # Create a traces and layout to plot
    data=[]
    figure={}
    # for j in range(num_curves):
    for curve in newdata:
        trace = go.Scattergl(x=curve['x'], y=curve['y'],
                             name=curve['name'], mode=curve['mode'])
        data.append(trace)
    layout = go.Layout(title=plotTitle, \
        yaxis={'title':'Random', 'type':'linear'}, xaxis={'title':'Time'})

    # PlotlyJSONEncoder converts objects to their JSON equivalents
    figure=dict(data=data, layout=layout)
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # Add dropdown for fridge states

    # try to connect to clients
    print('try to connect to clients')
    test_connect()

    return render_template('webFridge.html', graphJSON=graphJSON,
                           async_mode=socketio.async_mode)


def load_data(graph):
    newdata = []
    for idx, name in enumerate(graph):
        print('load_data:', name)
        newdata.append({'x': [], 'y': [], 'type': 'scatter',
                            'mode':'markers+lines', 'name':'%s' % name})
    return newdata




if __name__ == "__main__":
    ip, port_ = '0.0.0.0', '45000'

    # Start Flask app
    socketio.run(app, host=ip, port=port_, debug=True, use_reloader=True)
