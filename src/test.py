from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def denoise():
    return "<p>Hello, World!</p>"
