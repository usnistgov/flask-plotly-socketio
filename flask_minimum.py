from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
        return 'Hello, World!'

ip, port_ = '0.0.0.0', 45000
app.run(host=ip, port=port_, use_reloader=True)
