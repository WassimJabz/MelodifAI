import io
import os
import sys

from flask import Flask, render_template, request, send_file, make_response
from PIL import Image



from model import Model

app = Flask(__name__)

model = Model()



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/selectionMade", methods=["GET","POST"])
def data():
    midiFile = model.predict(request)
    
    print(int(request.data))
    return "request successful"
    
@app.route('/download', methods=["GET","POST"])
def dowload():
    return send_file('Melody_Generated.mid', as_attachment = True)

if __name__ == '__main__':
   app.run()