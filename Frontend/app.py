import io
import os
import sys

from flask import Flask, render_template, request, send_file
from PIL import Image



from model import Model

app = Flask(__name__)

model = Model()



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/selectionMade", methods=["POST"])
def data():
    midiFile = model.predict(request)
    print("request successful")
    return True
    
@app.route('/download', methods=["FETCH"])
def dowload():
    return send_file('Melody_Generated.mid', as_attachment = True)

if __name__ == '__main__':
   app.run()