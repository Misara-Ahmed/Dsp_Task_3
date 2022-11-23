from flask import Flask, request, render_template,jsonify
import os
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import functions as fn
app = Flask(__name__)
@app.route("/")
def index():
        # parent_dir = os.path.dirname(os.path.abspath(__file__))
        # # Custom REACT-based component for recording client audio in browser
        # build_dir = os.path.join(parent_dir, "rec\\frontend\\build")
    
    #song=ipd.Audio(file)
    df=fn.extract_features()

    return render_template('index.html',features=df)


if __name__ == '__main__':        

    app.run(debug=True)