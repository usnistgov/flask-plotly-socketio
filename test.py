import zmq
import random
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

with open('2017-10-10-12-19-14.csv', 'r') as f:
    count = 0
    while True:
        line = f.readline()
        if len(line)==0:
            break
        if not line.startswith('#'):
            socket.send_string(line[:-1])
            print(count, line[:-1])
            count += 1
            time.sleep(1)
