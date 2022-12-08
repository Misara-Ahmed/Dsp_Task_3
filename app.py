from flask import Flask, request, render_template,request
import os
import matplotlib.pyplot as plt
import functions as fn
import audio_features
import soundfile
import io
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = "./"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
@app.route("/",methods=['GET','POST'])
def index():
    if(request.method=='GET'):
        
        features_list=audio_features.feature_extraction_array('./my-rec.wav')
        voice_prediction,speech_prediction=fn.apply_model(features_list)
        fn.plot_melspectrogram('./my-rec.wav')
        
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
            voice_prediction=""
            speech_prediction=""
            

    return render_template('index.html',voice_prediction= voice_prediction,speech_prediction=speech_prediction)


if __name__ == '__main__':        

    app.run(debug=True)