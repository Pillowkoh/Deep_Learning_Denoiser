from flask import Flask, render_template, request, send_from_directory, send_file
from io import StringIO
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("start.html")

app.config["AUDIO_INPUTS"] = 'inputs'
app.config["AUDIO_OUTPUTS"] = 'outputs'

@app.route("/denoise", methods=['GET','POST'])
def denoise():
    if request.method == "POST":
        audio_input = request.files["audio-input"]
        if audio_input.filename != '':
            print("IN UPLOAD:", audio_input)
            audio_input.save(os.path.join(app.config["AUDIO_INPUTS"], audio_input.filename))
            # TO DO: convert input audio to model-required form
            # TO DO: denoise with model
            # TO DO: convert back to web-readable form
            # TO DO: save to AUDIO_OUTPUTS
            audio_input.save(os.path.join(app.config["AUDIO_OUTPUTS"], audio_input.filename)) # remove
            return render_template("denoise.html", audio_output=audio_input.filename)
            
    return render_template("denoise.html")

@app.route("/playaudio/<name>")
def playaudio(name):
    path = app.config["AUDIO_OUTPUTS"] +'/'+ name
    return send_from_directory(app.config["AUDIO_OUTPUTS"], name)

@app.route("/download/<name>")
def download(name):
    path = app.config["AUDIO_OUTPUTS"] +'/'+ name
    return send_file(path, as_attachment=True)