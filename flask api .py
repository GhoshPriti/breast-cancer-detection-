from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

app = Flask(__name__)
# Custom sensitivity function
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Load trained model
model = load_model("cancer_cell_model.keras", compile=False)

# Class labels
classes = ["benign", "malignant", "normal"]


@app.route("/")
def home():
    return '''
    <h2>Breast Cancer Detection System</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file"><br><br>
        <input type="submit" value="Predict">
    </form>
    '''


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join("temp.png")
    file.save(filepath)

    # Load image
    img = image.load_img(filepath, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Prediction
    prediction = model.predict(img)
    index = np.argmax(prediction)
    result = classes[index]
    confidence = float(np.max(prediction)) * 100

    return f"""
    <h2>Prediction Result</h2>
    <h3>Class: {result}</h3>
    <h3>Confidence: {confidence:.2f}%</h3>
    <br><a href="/">Upload Another Image</a>
    """


if __name__ == "__main__":
    app.run(debug=True)
