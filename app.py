from flask import Flask, render_template, request, jsonify
import numpy as np
from base64 import b64decode, b64encode
from PIL import Image
import io
from whitenoise import WhiteNoise

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import torch
import numpy as np

class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.cnn_layers = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding="valid"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.1),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(128, 256, kernel_size=3, padding="valid"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.1),
        torch.nn.MaxPool2d(3),
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.LazyLinear(),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.LazyLinear(),
        torch.nn.Softmax()
    )
  def forward(self, X):
      result = self.cnn_layers(X)
      return result

device = torch.device('cpu')
cnn = CNN()
cnn.load_state_dict(torch.load("./model/model.pth", map_location=device))
cnn.eval()

app = Flask(__name__, static_url_path='', static_folder="./static")
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/")

uri = "mongodb+srv://phucthuhigh:phucthuhigh1701@handwritting.fxzvn.mongodb.net/?retryWrites=true&w=majority&appName=handwritting"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.handwritting

dataset = db.handwritting

def preprocess_image(image_data):
    # Convert base64 to image and preprocess image
    image_bytes = b64decode(image_data.split(",")[1])
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((28, 28))
    img = img.convert("L")
    img_array = np.invert(np.array(img)).reshape((1, 1, 28, 28))

    # Convert image to base64 and store to mongodb
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    image_bytes_scale = b64encode(buffered.getvalue())
    return image_bytes_scale, img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    img_data = request.json["image_data"]
    _, img_array = preprocess_image(img_data)
    img = torch.tensor(img_array, dtype=torch.float32)
    img = img.to(device)
    pred = cnn(img)
    print(pred)
    pred = torch.argmax(pred, dim=1)
    return jsonify({"success": True, "data": pred.item()})

@app.route("/save", methods=["POST"])
def save():
    try:
        img_data = request.json["image_data"]
        img_label = request.json["label"]
        image_bytes_scale, _ = preprocess_image(img_data)
        dataset.insert_one({"label": img_label,"imgURL": image_bytes_scale})
        print(img_label)
        return jsonify({"success": True})
    except:
        return jsonify({"success": False}), 404

if __name__ == "__main__":
    app.run(debug=True)