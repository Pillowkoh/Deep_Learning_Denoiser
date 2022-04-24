from flask import Flask, render_template, request, send_from_directory, send_file
from io import StringIO
import os
from denoiser_app import AudioDenoiser

app = Flask(__name__)

# Start page
@app.route("/")
def home():
    return render_template("start.html")

# paths to input and output folders where audio files will be saved
app.config["AUDIO_INPUTS"] = 'inputs'
app.config["AUDIO_OUTPUTS"] = 'outputs'

# Main page to denoise files
@app.route("/denoise", methods=['GET','POST'])
def denoise():
    # If the form on the page is submitted
    if request.method == "POST": 

        # Request files in the file input segment (name: 'audio-input') of the form on the page
        audio_input = request.files["audio-input"]

        # If a file is submitted i.e. when the filename is not blank
        if audio_input.filename != '':

            # Save audio file into inputs folder
            audio_input.save(os.path.join(app.config["AUDIO_INPUTS"], audio_input.filename))

            # Generate input filepath, output filename, and output filepath
            in_fp = os.path.join(os.path.join(app.config["AUDIO_INPUTS"], audio_input.filename))
            out_fn = audio_input.filename[:-4] + '_denoised' + '.wav'
            out_fp = os.path.join(os.path.join(app.config["AUDIO_OUTPUTS"], out_fn))
            
            # Send file to Denoiser using input filepath
            denoiser = AudioDenoiser(in_fp)

            # Set denoised file to output filepath
            denoiser.denoise(out_fp)

            # Render the template with output filename as parameter, to generate download link
            return render_template("denoise.html", audio_output=out_fn)
    
    # "GET" method i.e. on first render of the route
    return render_template("denoise.html")

# Download the denoised file
@app.route("/download/<name>")
def download(name):
    print("NAME:", name)
    path = app.config["AUDIO_OUTPUTS"] +'/'+ name
    return send_file(path, as_attachment=True)