from flask import Flask, render_template, request, send_from_directory, send_file
from io import StringIO
import os
from denoiser_app import AudioDenoiser

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
            audio_input.save(os.path.join(app.config["AUDIO_INPUTS"], audio_input.filename))

            in_fp = os.path.join(os.path.join(app.config["AUDIO_INPUTS"], audio_input.filename))
            out_fn = audio_input.filename[:-4] + '_denoised' + '.wav'
            out_fp = os.path.join(os.path.join(app.config["AUDIO_OUTPUTS"], out_fn))
            
            denoiser = AudioDenoiser(in_fp)
            denoiser.denoise(out_fp)

            return render_template("denoise.html", audio_output=out_fn)
            
    return render_template("denoise.html")

@app.route("/download/<name>")
def download(name):
    print("NAME:", name)
    path = app.config["AUDIO_OUTPUTS"] +'/'+ name
    return send_file(path, as_attachment=True)