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

# Global variables to hold models, will be lazily loaded
cnn_model = None
rf_model = None
intermediate_layer_model = None

def load_models():
    global cnn_model, rf_model, intermediate_layer_model
    
    if cnn_model is None:
        cnn_model = load_model('models/final_model.h5')
    
    if rf_model is None:
        rf_model = joblib.load('models/random_forest_model.pkl')
    
    if intermediate_layer_model is None:
        layer_name = 'global_average_pooling2d'
        intermediate_layer_model = tf.keras.models.Model(
            [cnn_model.inputs],
            [cnn_model.get_layer(layer_name).output]
        )
    print(" * Models loaded successfully")
    return cnn_model, rf_model, intermediate_layer_model

def extract_features_final(intermediate_layer_model, img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    features = intermediate_layer_model.predict(img)
    return features.flatten()  # Flatten

def predict_label_final(image, threshold=0.5):
    cnn_model, rf_model, intermediate_layer_model = load_models()
    
    img = image.resize((255, 255))  
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
    try: 
        # Ensure an image file is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file in the request'}), 400

        # Retrieve the image file
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Open the image file
        image = Image.open(file.stream)
        print(f' * Image: {image}')

        # Example usage
        predicted_label, confidence = predict_label_final(image)

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

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
