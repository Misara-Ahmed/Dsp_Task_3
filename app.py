from flask import Flask, request, render_template,jsonify,request
import os
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import functions as fn
import soundfile
import io
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = "./"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
@app.route("/",methods=['GET','POST'])
def index():
    if(request.method=='GET'):
        features_list=fn.extract_features('./Recording (3).wav')
        prediction=fn.apply_model(features_list)
    else:
        if 'data' in request.files:
            url = "http://127.0.0.1:5000/"
            key = 'REST API KEY'
            headers = {
                "Content-Type": "application/octet-stream",
                "Transfer-Encoding":"chunked",
                "Authorization": "KakaoAK " + key,
            }

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

            soundfile.write('my-rec2.wav', data, samplerate)

            with open("my-rec2.wav", 'rb') as fp:
                audio = fp.read()

            result = request.post(url, headers=headers, data=audio)
            features_list=result.text
            prediction=""
        
    return render_template('index.html',features=prediction)


if __name__ == '__main__':        

    app.run(debug=True)