import zmq
import random
import time
import threading
fname = 'logs/2017-10-10-12-19-14.csv'
fname = 'logs/2018-11-03-16-42-33.csv'

def serve():
    print('serve thread started')
    with open(fname, 'r') as f:
        labels = f.readline()
        labels = labels[1:]
        labels = labels.strip()
        labels = labels.split(',', 1)
        labels = labels[1]
        print(labels)
    server = context.socket(zmq.REP)
    server.bind("tcp://*:50326")
    while True:
        request = server.recv_string()
        print('request: ' + request)
        server.send(labels.encode())

server_thread = threading.Thread(target=serve)
server_thread.start()

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

with open(fname, 'r') as f:
    count = 0
    while True:
        line = f.readline()
        if len(line)==0:
            break
        if not line.startswith('#'):
            socket.send_string(line[:-1])
            print(count, line[:-1])
            count += 1
            time.sleep(10)
