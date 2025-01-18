from flask import Flask, render_template, request, jsonify
import numpy as np
from base64 import b64decode, b64encode
from PIL import Image
import io
import datetime

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)

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
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img = img.convert("L")

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
    return jsonify({"success": True})

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