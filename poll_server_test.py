import zmq

# Socket to talk to server
context = zmq.Context()
subscriber = context.socket(zmq.SUB)

subscriber.connect("tcp://localhost:5556")
subscriber_filter = ''

# Python 2 - ascii bytes to unicode str
if isinstance(subscriber_filter, bytes):
    subscriber_filter = subscriber_filter.decode('ascii')
subscriber.setsockopt_string(zmq.SUBSCRIBE, subscriber_filter)

server = context.socket(zmq.REP)
server.bind("tcp://*:5555")

poller = zmq.Poller()
poller.register(server, zmq.POLLIN )
poller.register(subscriber, zmq.POLLIN)

while True:
    try:
        socks = dict(poller.poll())
    except KeyboardInterrupt:
        break

    if server in socks:
        message = server.recv_string()
        msg_out = 'server message:' +  message
        print(msg_out)
        server.send_string(msg_out)
    if subscriber in socks:
        data_string = subscriber.recv_string()
        print('subscriber message:', data_string)
        # state_machine.update(data_string)

