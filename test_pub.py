
#   random number publisher
#   Binds PUB socket to tcp://*:5556
#

import zmq
import random
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

while True:
    r = random.random()
    socket.send_string("%.3f" % (r))
    time.sleep(1)
