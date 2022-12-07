from flask import Flask, request, render_template,jsonify,request
import os
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import functions as fn
import soundfile
import io
import numpy as np
import base64

from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = "./"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
@app.route("/",methods=['GET','POST'])
def index():
    if(request.method=='GET'):
        features_list=fn.extract_features('./my-rec.wav')
        prediction=fn.apply_model(features_list)
        img,fig=fn.plot_melspectrogram('./my-rec.wav')
        fig.colorbar(img,format="%+2.f")
        spectro= plt.savefig('./static/spectro.png')
        spectro=True
        # fig = Figure()
        # ax = fig.subplots()
        # ax.plot([1, 2])
        # # Save it to a temporary buffer.
        # buf = io.BytesIO()
        # fig.savefig(buf, format="png")
        # # Embed the result in the html output.
        # data = base64.b64encode(buf.getbuffer()).decode("ascii")
        # return f"<img src='data:image/png;base64,{data}'/>"
    else:
        if 'data' in request.files:
            url = "http://127.0.0.1:5000/"
          

            file = request.files['data']
            
            # Write the data to a file.
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Jump back to the beginning of the file.
            file.seek(0)
            
            # Read the audio data again.
            data, samplerate = soundfile.read(file)
            with io.BytesIO() as fio:
                soundfile.write(
                    fio, 
                    data, 
                    samplerate=samplerate, 
                    
                    format='wav'
                )
                data = fio.getvalue()

            soundfile.write('my-rec.wav', data, samplerate)

            with open("my-rec.wav", 'rb') as fp:
                audio = fp.read()

            result = request.post(url, data=audio)
            features_list=result.text
            prediction=""
            

    return render_template('index.html',features=prediction,spectro=spectro)


if __name__ == '__main__':        

    app.run(debug=True)