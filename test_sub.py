#
#   Weather update client
#   Connects SUB socket to tcp://localhost:5556
#   Collects weather updates and finds avg temp in zipcode
#

import sys
import zmq

#  Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

socket.connect("tcp://localhost:5555")

# Subscribe to zipcode, default is NYC, 10001
sub_filter = ''

# Python 2 - ascii bytes to unicode str
if isinstance(sub_filter, bytes):
    sub_filter = sub_filter.decode('ascii')
socket.setsockopt_string(zmq.SUBSCRIBE, sub_filter)


while True:
    string = socket.recv_string()
    print(string)
