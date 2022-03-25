import io
import torch
from flask import Flask, render_template, request
from PIL import Image
from model import Model, transformation

app = Flask(__name__)

model = Model()
model.load_state_dict(torch.load("weights.pth"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    im = Image.open(io.BytesIO(request.data)).convert("L")
    im = transformation(im).unsqueeze(0)
    with torch.no_grad():
        preds = model(im)
        preds = torch.argmax(preds, axis=1)
        print(preds[0].item())
        return {"data": preds[0].item()}
