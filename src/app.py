import io
import os
from flask import abort, Flask, jsonify, render_template, request, Response, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
from PIL import Image
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as preproc_image


# Paths management
if os.path.basename(os.getcwd()) == 'src':
    MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()), 'model.h5')
else:
    MODEL_PATH = 'model.h5'

# Application and model instantiation
APP = Flask(__name__)
MODEL = load_model(MODEL_PATH)

# Swagger blueprint registration
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'

swagger_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config = {
        'app_name': 'Image Classification Service'
    }
)

APP.register_blueprint(swagger_blueprint)

# Binding routes
@APP.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@APP.route('/classifyImage', methods=['POST'])
def classify():
    # Get the file stream from the request
    file_stream = request.files['image'].read()

    # Check whether the file stream represents an image
    try:
        image = Image.open(io.BytesIO(file_stream))
    except:
        return Response('Invalid file type supplied', status=400)

    # Process the image to be compatible as input for VGG
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
    APP.run(host='0.0.0.0', port=8060, debug=True, use_reloader=False)
