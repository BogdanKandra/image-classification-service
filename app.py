import io
from flask import abort, Flask, jsonify, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as preproc_image


# Paths management
MODEL_PATH = 'model.h5'

# Application and model instantiation
APP = Flask(__name__)
MODEL = load_model(MODEL_PATH)

# Binding routes
@APP.route('/classification_api', methods=['POST'])
def predict():
    # Get the image stream from the request
    image_stream = request.files['image'].read()

    # Process the image to be compatible as input for VGG
    image = Image.open(io.BytesIO(image_stream))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = preproc_image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    vgg_input = vgg16.preprocess_input(image_array)

    # Run the prediction
    predictions = MODEL.predict(vgg_input)
    predicted_classes = vgg16.decode_predictions(predictions, top=10)

    # Return the jsonified output
    response = {}
    predictions_list = []
    for (imagenet_id, label, likelihood) in predicted_classes[0]:
        prediction = {'label': label, 'likelihood': float(round(likelihood, 3))}
        predictions_list.append(prediction)
    response['predictions'] = predictions_list

    return jsonify(response)


# Run the application
if __name__ == '__main__':
    APP.run(host='localhost', port=8060, debug=True)


# curl -F image=@daisy.jpg http://localhost:8060/classification_api