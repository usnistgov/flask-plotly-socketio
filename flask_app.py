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
import fridge_machine

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
if 'piServer2' not in parentdir:
    parentdir = os.path.join(parentdir, 'piServer2')
sys.path.insert(0,parentdir)
import data_logger
import client
import lttb
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
import preallocate

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

state_subscriber = context.socket(zmq.SUB)
state_subscriber.connect("tcp://localhost:5557")
state_subscriber.setsockopt_string(zmq.SUBSCRIBE, subscriber_filter)

poller = zmq.Poller()
poller.register(subscriber, zmq.POLLIN)
poller.register(state_subscriber, zmq.POLLIN)

fridge_client = context.socket(zmq.REQ)
fridge_client.connect("tcp://localhost:5555")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, async_mode=None)
thread = None

if False:
    with open('./logs/2017-10-10-12-19-14.csv', 'r') as f:
        labels = f.readline()
    LOG_PATH = './logs/'
else:
    labels = client.client('127.0.0.1', 50326, 'getlabels')
    LOG_PATH = '../logs/'
labels = labels.split(',')
labels = [label.strip() for label in labels]
for l in labels:
    print(len(l), l, type(l))
graph = ['40K', '4K', '1K', 'switch', 'pump']
table = ['40K', '4K', '1K', 'switch', 'pump', 'hp', 'hs', 'relays']
DATA = None
SIZE = 100000
def load_history():
    # global data_history, logfile_prefix
    # if 'logfile_prefix' in fridge.config['logfile_prefix']:
    #     prefix = fridge.config['logfile_prefix']
    # else:
    #     prefix = ''
    #  Find previous log file

    previous_log_filename = data_logger.getlast.getlast(LOG_PATH, '*.csv')
    print('log_filename', previous_log_filename)
    #  Create logger object
    if '-' in previous_log_filename:
        logfile_prefix = previous_log_filename.split('-')[:2]
        print('logfile_prefix', logfile_prefix)
        try:
            if len(logfile_prefix[1]) == 4:  # Second item is the year
                logfile_prefix = logfile_prefix[0]
            else:
                logfile_prefix = logfile_prefix[0][:-4]  # strip year
                print('logfile_prefix', logfile_prefix)
        except:
            pass
    else:
        logfile_prefix = ''
    filename = LOG_PATH + previous_log_filename
    tail = subprocess.Popen(['tail', '-10000', filename],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = tail.communicate()
    # with open(filename, 'r') as f:
    #    history = deque(f, history_start)
    history_start = 100000
    history = deque(iter(output.decode().splitlines()), history_start)
    data_history = np.genfromtxt(history, invalid_raise=False, delimiter=',')
    print(type(data_history), data_history.shape)
    data = preallocate.PreallocatedArray(np.zeros((SIZE, data_history.shape[1])))
    data[:data_history.shape[0], :data_history.shape[1]] = data_history
    data.length = data_history.shape[0]
    try:
        last_point = history.pop()
    except:
        last_point = None
    app_logger.info('last_point, %r' % last_point)
    return data, filename

DATA, FILENAME = load_history()

def background_thread():
    global labels, graph, table, DATA
    """send server generated events to clients."""
    socketio.sleep(1)
    while True:
        socketio.sleep(1)
        socks = dict(poller.poll())
        # print('poll socks', socks)
        if subscriber in socks:
            data_string = subscriber.recv_string()
            # print('subscriber message:', data_string)
            data = data_string.split(',')
            # print('len of data', len(data))
            app_logger.debug(data_string)
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
                idx = labels.index(name.lower()) + 1
                value = float(data[idx])
                if np.isnan(value):
                    value = -1
                y.append([value])
                x.append([timestamp])
            # print(x,y)
            datastr = {'x': x, 'y': y}
            # print('datastr', datastr)
            table_dict = {}
            for name in table:
                idx = labels.index(name.lower()) + 1
                value = float(data[idx])
                if np.isnan(value):
                    value = -1
                table_dict[name] = value
            alldata = {'graph': datastr, 'table': table_dict}
            socketio.emit('new_data', alldata) #  ,namespace='/')
            time, data = data_string.split(',', 1)
            new_row = np.fromstring(data_string, sep=',')
            new_row2 = np.zeros(len(new_row)+1)
            new_row2[0] = new_row[0]
            new_row2[1] = float('nan')
            new_row2[2:] = new_row[1:]
            DATA.append_row(new_row2)
            # print('background DATA.shape', DATA.shape, DATA.length)

        elif state_subscriber in socks:
            data_string = state_subscriber.recv_string()
            app_logger.debug('fridge is in state: %s' % data_string)
            socketio.emit('fridge_state', data_string)

@socketio.on('connect', namespace='/')
def test_connect():
    global thread, graph, table
    if thread is None:
        thread =  socketio.start_background_task(target=background_thread)
        print('got to test_connect, thread started')
    print('passing sensor_names to client')
    # emit('my_response', {'data': 'Connected'}, namespace='/')
    data = {'graph': graph, 'table': table}
    socketio.emit('connect', data)  # , namespace='/')

@socketio.on('disconnect')  #, namespace='/')
def test_disconnect():
    print('Client disconnected', request.sid)

@socketio.on('my_event')  #, namespace='/')
def my_event(message):
    print('my_event', request.sid)
    print('message', message)

@socketio.on('switch')  #, namespace='/')
def switch_event(message):
    print('my_event', request.sid)
    print('switch message', type(message), message)
    client_message = ' '.join(['set_heater', message['name'],
            '%s' % message['state']])
    print('client_message', client_message)
    fridge_client.send_string(client_message)
    msg = fridge_client.recv_string()

@socketio.on('state')  #, namespace='/')
def switch_event(message):
    print('my_event', request.sid)
    print('state message', type(message), message)
    #  This sends the data... should see if timesout...
    fridge_client.send_string('set_state '+message)
    msg = fridge_client.recv_string()

@app.route("/", methods=['GET', 'POST'])
def plot():
    global DATA
    plotTitle = 'Flask socketio plotly test app'

    # Updates the data for the table
    newdata, filename = load_data(graph)
    # Create a traces and layout to plot
    print('filename', filename)
    data=[]
    figure={}
    # for j in range(num_curves):
    for curve in newdata:
        trace = go.Scattergl(x=curve['x'], y=curve['y'],
                             name=curve['name'], mode=curve['mode'])
        data.append(trace)
    layout = go.Layout(title=plotTitle, \
        yaxis={'title':'Random', 'type':'log'}, xaxis={'title':'Time'})

    # PlotlyJSONEncoder converts objects to their JSON equivalents
    figure=dict(data=data, layout=layout)
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # try to connect to clients
    print('try to connect to clients')
    test_connect()

    return render_template('webFridge.html', graphJSON=graphJSON,
                           states = fridge_machine.Fridge.fridge_states,
                           async_mode=socketio.async_mode)


def load_data(graph):
    global DATA
    print('load_data DATA.shape', DATA.shape, DATA.length)
    print('building graphs')
    # history, filename = load_history()
    history = DATA
    filename = FILENAME
    newdata = []
    x = [datetime.datetime.fromtimestamp(d).strftime('%y-%m-%d %H:%M:%S')
         for d in history[:, 0]]

    for idx, name in enumerate(graph):
        lower_labels = [label.lower() for label in labels]
        idx = lower_labels.index(name.lower()) + 2  # Could be 2 if there is human readable date
        if len(history[:, 0]) > 1000:
            hdata = np.column_stack((history[:, 0], history[:, idx]))
            downsize = lttb.downsample(hdata, n_out=1000)
            assert downsize.shape == (1000, 2)
            x = [datetime.datetime.fromtimestamp(d).strftime('%y-%m-%d %H:%M:%S')
                    for d in downsize[:, 0]]
            y = list(downsize[:,1])
        else:
            y = list(history[:, idx])

        print('load_data:', name, idx)
        newdata.append({'x': x, 'y': y, 'type': 'scatter',
                            'mode':'markers+lines', 'name':'%s' % name})
    print('done building graphs')
    return newdata, filename




if __name__ == "__main__":
    ip, port_ = '0.0.0.0', '45000'

    # Start Flask app
    #  socketio.run(app, host=ip, port=port_, debug=True, use_reloader=True)
    socketio.run(app, host=ip, port=port_, debug=True, use_reloader=False)
    # socketio.run(app, host=ip, port=port_)
