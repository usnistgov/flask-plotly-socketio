from __future__ import print_function
import os
import sys
import time
import datetime
from collections import deque
import logging
import logzero
from logzero import logger
# import socket
import subprocess
import zmq
import fridge_machine
import ruamel
from ruamel.yaml import YAML
import numpy as np
#*** to run this app you need pip install Flask and Flask-SocketIO and eventlet and ***#
from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import json
import preallocate

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
if 'piServer2' not in parentdir:
    parentdir = os.path.join(parentdir, 'piServer2')
sys.path.insert(0,parentdir)
import data_logger
import client
import lttb

from eventlet import monkey_patch as monkey_patch
monkey_patch()


yaml = YAML()

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
TESTING = False
if len(sys.argv) == 3:
    TESTING = True
    print('TESTING')
if TESTING:
    with open('./logs/2017-10-10-12-19-14.csv', 'r') as f:
        labels = f.readline()
        labels = labels.split(',', 1)[-1]
    LOG_PATH = './logs/'
    table = ['40K', '4K', '1K', 'switch', 'pump', 'hp', 'hs']
    LABEL_OFFSET = 1
    YAML_FILE = './config.yaml'
else:
    labels = client.client('127.0.0.1', 50326, 'getlabels')
    LOG_PATH = '../logs/'
    table = ['40K', '4K', '1K', 'switch', 'pump', 'hp', 'hs', 'relays']
    LABEL_OFFSET = 2
    YAML_FILE = '../config.yaml'
with open(YAML_FILE, 'r') as f:
    CONFIG = yaml.load(f)
labels = labels.split(',')
labels = [label.strip() for label in labels]
for l in labels:
    print(len(l), l, type(l))
graph = ['40K', '4K', '1K', 'switch', 'pump']
DATA = None
SIZE = 100000
HR_date = False
def load_history():
    # global data_history, logfile_prefix
    # if 'logfile_prefix' in fridge.config['logfile_prefix']:
    #     prefix = fridge.config['logfile_prefix']
    # else:
    #     prefix = ''
    #  Find previous log file
    global HR_date, logfile_prefix
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
    tail = subprocess.Popen(['tail', '-50000', filename],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = tail.communicate()
    # with open(filename, 'r') as f:
    #    history = deque(f, history_start)
    history_start = 100000
    history = deque(iter(output.decode().splitlines()), history_start)
    data_history = np.genfromtxt(history, invalid_raise=False, delimiter=',')
    print(type(data_history), data_history.shape)
    #  Check if Human Readable data is the second column
    print(data_history[0, 1])
    if np.isnan(data_history[0, 1]):
        HR_date = True
    else:
        HR_date = False
    print('HR_date: ' + '%s' % HR_date)
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

def on_new_log_pressed():
    global logfile_prefix
    head = ''
    if len(logfile_prefix) > 0:
        print('logfile_prefix:', logfile_prefix)
        if logfile_prefix.endswith('-'):
            logfile_prefix = logfile_prefix[:-1]
        head = logfile_prefix.rstrip('0123456789')
    print('head:', head)
    if len(head) > 0:
        print(logfile_prefix, head, len(head))
        tail = int(logfile_prefix[len(head):])
        print('new file name', head, tail+1)
        logfile_prefix = head + '%d-' % (tail + 1)
        print(logfile_prefix)
    filename = LOG_PATH + logfile_prefix + \
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.csv')
    print('new log file name: ' + filename)
    with open(filename, 'w') as csvfile:
        csvfile.write('')

    if False:
        create_log(filename, logger=logger_config['logger'])
        self.logfile_label.set_text(filename)
        self.logfile_label.attributes['download'] = os.path.basename(filename)
        self.logfile_label._filename = filename

def background_thread():
    global labels, graph, table, DATA, HR_date
    """send server generated events to clients."""
    print('starting background thread')
    socketio.sleep(1)
    while True:
        socketio.sleep(1)
        socks = dict(poller.poll())
        # print('poll socks', socks)
        if subscriber in socks:
            data_string = subscriber.recv_string()
            if data_string.startswith('file'):
                print('data_string:', data_string)
                socketio.emit('update_log_name', os.path.basename(data_string))
                return
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
            def get_idx(labels, name):
                idx = None
                offset = 1
                try:
                    idx = labels.index(name.lower()) + offset
                except ValueError:
                    try:
                        idx = labels.index(name.upper()) + offset
                    except Exception as e:
                        # print(e)
                        pass
                except Exception as e:
                    print(e)
                # if idx is None:
                #     print('Can not find ', name)
                return idx
            for i, name in enumerate(graph):
                idx = get_idx(labels, name)
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
                idx = get_idx(labels, name)
                # idx = labels.index(name.lower()) + 1
                if idx is not None:
                    value = float(data[idx])
                    if np.isnan(value):
                        value = -1
                else:
                    value = -2
                table_dict[name] = value
            alldata = {'graph': datastr, 'table': table_dict}
            socketio.emit('new_data', alldata) #  ,namespace='/')
            new_row = np.fromstring(data_string, sep=',')
            # print('len new_row', len(new_row), HR_date)
            if HR_date:
                new_row2 = np.zeros(len(new_row)+1)
                new_row2[0] = new_row[0]
                new_row2[1] = float('nan')
                new_row2[2:] = new_row[1:]
            else:
                new_row2 = new_row
            DATA.append_row(new_row2)
            # print('background DATA.shape', DATA.shape, DATA.length)

        elif state_subscriber in socks:
            data_string = state_subscriber.recv_string()
            app_logger.debug('fridge is in state: %s' % data_string)
            socketio.emit('fridge_state', data_string)

@socketio.on('connect', namespace='/')
def test_connect():
    global thread, graph, table
    if (thread is None):  # and (not TESTING):
        thread =  socketio.start_background_task(target=background_thread)
    print("sending names to client")
    data = {'graph': graph, 'table': table, 'log_filename': os.path.basename(FILENAME)}
    socketio.emit('connect', data)  # , namespace='/')

@socketio.on('disconnect')  #, namespace='/')
def test_disconnect():
    print('Client disconnected', request.sid)

@socketio.on('my_event')  #, namespace='/')
def my_event(message):
    print('my_event', request.sid)
    print('message', message)
    # if not TESTING:
    if True:
        client_message = 'get_recycle_hour'
        fridge_client.send_string(client_message)
        recycle_hour = int(float(fridge_client.recv_string()))

        client_message = 'get_next_recycle_time'
        fridge_client.send_string(client_message)
        next_recycle_time = eval(fridge_client.recv_string())
        next_recycle_time = next_recycle_time.strftime('%Y-%m-%d %H:%M:%S')

        message = {'recycle_hour': recycle_hour, 'next_time': next_recycle_time}
        socketio.emit('update_recycle', message) 

@socketio.on('new_logfile')  #, namespace='/')
def new_logfile():
    print('Create new log file')
    on_new_log_pressed()

@socketio.on('unzoom')  #, namespace='/')
def unzoom(message):
    global graph

    graphJSON = build_graphjson()
    socketio.emit('zoom_graph', graphJSON)

@socketio.on('zoom')  #, namespace='/')
def zoom(message):
    global graph
    # print('zoom', request.sid)
    # print('message', type(message), message)
    format_str = '%Y-%m-%d %H:%M:%S.%f'
    start = datetime.datetime.strptime(message['xaxis.range[0]'], format_str)
    stop = datetime.datetime.strptime(message['xaxis.range[1]'], format_str)
    start = time.mktime(start.timetuple())
    stop = time.mktime(stop.timetuple())
    print('timestamps for range', start, stop, stop-start)

    newdata, filename = load_data(graph, data_slice = (start, stop))
    data=[]
    figure={}
    for curve in newdata:
        trace = go.Scattergl(x=curve['x'], y=curve['y'],
                             name=curve['name'], mode=curve['mode'])
        data.append(trace)
    layout = go.Layout(  # title=plotTitle, \
        yaxis={'title':'Temperatures', 'type':'log'}, xaxis={'title':'Time'},
        margin={ 'l': 50, 'r': 50, 'b': 50, 't': 5, 'pad': 0 },
        )

    # PlotlyJSONEncoder converts objects to their JSON equivalents
    figure=dict(data=data, layout=layout)
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    socketio.emit('zoom_graph', graphJSON)

@socketio.on('set_recycle_hour')  #, namespace='/')
def set_recycle_hour(message):
    print('my_event', request.sid)
    print('recycle_hour message', type(message), message)
    fridge_client.send_string('set_recycle_hour '+message)    
    msg = fridge_client.recv_string()

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

@app.route("/logfile/")
def download_logfile():
    try:
        print('sending', os.path.basename(FILENAME))
        return send_file(FILENAME, as_attachment=True, attachment_filename=os.path.basename(FILENAME))
    except Exception as e:
        return str(e)

@app.route("/", methods=['GET', 'POST'])
def plot():
    global DATA, graph
    plotTitle = 'Flask socketio plotly test app'
    # graphJSON = build_graphjson()
    layout = go.Layout(  # title=plotTitle, \
        yaxis={'title':'Temperatures', 'type':'log'}, xaxis={'title':'Time'},
        margin={ 'l': 50, 'r': 50, 'b': 50, 't': 5, 'pad': 0 },
        )
    trace= go.Scattergl(x=[0,1], y=[1,0], name='temp')
    # PlotlyJSONEncoder converts objects to their JSON equivalents
    figure=dict(data=[trace], layout=layout)
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # # Updates the data for the table
    # newdata, filename = load_data(graph)
    # # Create a traces and layout to plot
    # print('filename', filename)
    # data=[]
    # figure={}
    # # for j in range(num_curves):
    # for curve in newdata:
    #     trace = go.Scattergl(x=curve['x'], y=curve['y'],
    #                          name=curve['name'], mode=curve['mode'])
    #     data.append(trace)
    # layout = go.Layout(  # title=plotTitle, \
    #     yaxis={'title':'Temperatures', 'type':'log'}, xaxis={'title':'Time'},
    #     margin={ 'l': 50, 'r': 50, 'b': 50, 't': 5, 'pad': 0 },
    #     )
    #
    # # PlotlyJSONEncoder converts objects to their JSON equivalents
    # figure=dict(data=data, layout=layout)
    # graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # try to connect to clients
    print('try to connect to clients')
    test_connect()
    if False:
        client_message = 'get_recycle_hour'
        fridge_client.send_string(client_message)
        recycle_hour = int(float(fridge_client.recv_string()))

        client_message = 'get_next_recycle_time'
        fridge_client.send_string(client_message)
        next_recycle_time = eval(fridge_client.recv_string())
        next_recycle_time = next_recycle_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        recycle_hour = -1 
        next_recycle_time = 'fix me'
    return render_template('webFridge.html', graphJSON=graphJSON,
                           states = fridge_machine.Fridge.fridge_states,
                           recycle_hour = recycle_hour, 
                           next_recycle_time = next_recycle_time,
                           async_mode=socketio.async_mode,
                           room=CONFIG['location'],
                           name=CONFIG['name'],
                          )


def load_data(graph, data_slice=None):
    global DATA
    print('load_data DATA.shape', DATA.shape, DATA.length)
    print('building graphs')
    # history, filename = load_history()
    history = DATA
    filename = FILENAME
    newdata = []
    if data_slice is None:
        data_slice = range(DATA.length)
    else:
        start = np.where(history[:,0]>data_slice[0])[0][0]
        stop = np.where(history[:,0]<data_slice[1])[0][-1]
        data_slice = range(start, stop)
        # print('data_slice:', data_slice, stop-start)
        # bool_ = np.logical_and(history[:,0] > data_slice[0], history[:,0] <
        #                data_slice[1])
        # data_slice = np.where(bool_)[0]
    x = [datetime.datetime.fromtimestamp(d).strftime('%y-%m-%d %H:%M:%S')
         for d in history[data_slice, 0]]
    use_lttb = len(x) > 1000
    for idx, name in enumerate(graph):
        lower_labels = [label.lower() for label in labels]
        idx = lower_labels.index(name.lower()) + LABEL_OFFSET # Could be 2 if there is human readable date
        # y = history[data_slice, idx]
        # print('y', y)
        # if len(history[data_slice, 0]) > 1000:
        if use_lttb:
            print('data_slice len', len(data_slice))
            hdata = np.column_stack((history[data_slice, 0], history[data_slice, idx]))
            print('hdata.shape', hdata.shape)
            downsize = lttb.downsample(hdata, n_out=1000)
            assert downsize.shape == (1000, 2)
            x = [datetime.datetime.fromtimestamp(d).strftime('%y-%m-%d %H:%M:%S')
                    for d in downsize[:, 0]]
            y = list(downsize[:,1])
        else:
            y = list(history[data_slice, idx])

        print('load_data:', name, idx)
        newdata.append({'x': x, 'y': y, 'type': 'scatter',
                            'mode':'markers+lines', 'name':'%s' % name})
    print('done building graphs')
    return newdata, filename

def build_graphjson(data_slice=None):

    global graph
    if data_slice is None:
        newdata, filename = load_data(graph)
    else:
        newdata, filename = load_data(graph, data_slice)
    # Create a traces and layout to plot
    print('filename', filename)
    data=[]
    figure={}
    # for j in range(num_curves):
    for curve in newdata:
        trace = go.Scattergl(x=curve['x'], y=curve['y'],
                             name=curve['name'], mode=curve['mode'])
        data.append(trace)
    layout = go.Layout(  # title=plotTitle, \
        yaxis={'title':'Temperatures', 'type':'log'}, xaxis={'title':'Time'},
        margin={ 'l': 50, 'r': 50, 'b': 50, 't': 5, 'pad': 0 },
        )

    # PlotlyJSONEncoder converts objects to their JSON equivalents
    figure=dict(data=data, layout=layout)
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def check_change(**kwargs):
    """ Check if config.yaml has changed"""
    kwargs = kwargs['kwargs']
    # print('check_change kwargs', kwargs)
    if not hasattr(check_change, 'filename'):
        check_change.filename = kwargs['filename']
    if not hasattr(check_change, 'mtime'):
        check_change.mtime = os.stat(check_change.filename).st_mtime

    new_mtime = os.stat(check_change.filename).st_mtime
    changed = new_mtime != check_change.mtime
    if changed:
        print('old, new', '%.2f'%time.time(), check_change.mtime, new_mtime)
        check_change.mtime = new_mtime
    return changed


def signal_modified_yaml(**kwargs):
    print('start to check for modified yaml')
    print('kwargs', kwargs)
    while True:
        socketio.sleep(1)
        # print('done sleeping')
        changed = check_change(**kwargs)
        if changed:
            # print('put something on queue')
            # QUEUE.put(changed)
            print('emit update')
            socketio.emit('update', CONFIG, namespace='/config')


yaml_thread = None

@socketio.on('connect', namespace='/config')
def socketio_connect():
    global CONFIG, yaml_thread
    data = CONFIG
    socketio.emit('connect', data, namespace='/config')
    if yaml_thread is None:
        print('Start background yaml_thread')
        thread = socketio.start_background_task(target=signal_modified_yaml,
                                                kwargs={'filename': YAML_FILE} )

@socketio.on('disconnect', namespace='/config')
def socketio_disconnect():
    print('Client disconnected', request.sid)

def update(d, new):
    updated = False
    for k in d.keys():
        if isinstance(d[k], ruamel.yaml.comments.CommentedMap):
            updated |= update(d[k], new[k])
        else:
            if d[k] != new[k]:
                print('replace', k, d[k], new[k])
                d[k] = new[k]
                updated = True
    return updated


@socketio.on('new_config', namespace='/config')
def socketio_update(msg):
    """ handle new_config socketio event, yaml was updated on web page
    """
    global yaml, YAML_FILE, CONFIG
    print('new_config', request.sid)
    updated = update(CONFIG, msg)
    if updated:
        # print('new CONFIG', CONFIG)
        with open(YAML_FILE, 'w') as f:
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.dump(CONFIG, f)


@app.route('/config')
def config_endpoint():
    """ /config endpoint of website """
    return render_template('jsoneditor.html')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ip, port_ = '0.0.0.0', sys.argv[1]
    else:
        ip, port_ = '0.0.0.0', '50000'

    # Start Flask app
    #  socketio.run(app, host=ip, port=port_, debug=True, use_reloader=True)
    socketio.run(app, host=ip, port=port_, debug=True, use_reloader=False)
    # socketio.run(app, host=ip, port=port_)
