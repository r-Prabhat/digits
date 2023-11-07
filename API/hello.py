from flask import Flask

app = Flask(__name__)
@app.route("/<name>")
def hello_world(name):
    return "<p>Hello, World</p>"
# @app.route("/<a>/<b>")
# def sumOfTwoNumber(a, b):
#     return str(int(a)+int(b))
