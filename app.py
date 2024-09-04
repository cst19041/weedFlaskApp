import base64
import numpy as np
import io
import os
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

import tensorflow
import joblib
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import InputLayer

app = Flask(__name__)
CORS(app)

# Load the models
cnn_model = load_model('models/final_model.h5')
rf_model = joblib.load('models/random_forest_model.pkl')

print(" * Loaded models...")
print(f' * Tensorflow Version:  {tensorflow.__version__}')

import sklearn
print(sklearn.__version__)

# Define the layer for feature extraction
layer_name = 'global_average_pooling2d'
intermediate_layer_model = Model(
[cnn_model.inputs],
[cnn_model.get_layer(layer_name).output])

def extract_features_final(intermediate_layer_model, img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    features = intermediate_layer_model.predict(img)
    return features.flatten()  # Flatten

def predict_label_final(img_path, threshold=0.5):
    img = load_img(img_path, target_size=(255, 255))
    features = extract_features_final(intermediate_layer_model, img)
    
    # Get probabilities for each class
    prediction_probs = rf_model.predict_proba([features])
    
    # Find the class with the highest probability
    max_prob = np.max(prediction_probs)
    predicted_label = rf_model.classes_[np.argmax(prediction_probs)]
    
    # Evaluate confidence
    if max_prob >= threshold:
        return predicted_label, max_prob
    else:
        return "Uncertain", max_prob

# get_model()

@app.route("/predict", methods=["POST"])
def predict():
    
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    
    print(f' * image:  {image}')

    # Save the image to a temporary file
    temp_image_path = "C:/Users/Jeevake/Weed/testimage.jpg"
    image.save(temp_image_path)

    # Example usage
#     img_path = 'C:/Users/Jeevake/Weed/Weed/TestData/Cyperus Rotundusare/Cyperus_Rotundusare_17.jpg'
    predicted_label, confidence = predict_label_final(temp_image_path)

    print(f'Predicted label for the image: {predicted_label}')
    print(f'Confidence level: {confidence}')
    
    if predicted_label == 0 and confidence > 0.50:
        output_name = "Cyperus Rotundusare"
    elif predicted_label == 1 and confidence > 0.50:
        output_name = "Echinocola  Crusgulli"
    elif predicted_label == 2 and confidence > 0.50:
        output_name = "Echinocola Colona"
    elif predicted_label == 3 and confidence > 0.50:
        output_name = "Ludwigia Perennis"
    elif predicted_label == 4 and confidence > 0.50:
        output_name = "Monochoria Vaginalis"
    else:
        output_name = "Uncertain"

    response = {
        'prediction': {
            'output': output_name,
        }
    }
    return jsonify(response)
