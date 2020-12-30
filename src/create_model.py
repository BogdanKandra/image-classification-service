import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.applications import vgg16


# Setting the project directory path, for easy access to resources
project_directory_path = os.getcwd()
while os.path.basename(project_directory_path) not in ['app', 'image-classification-service']:
    project_directory_path = os.path.dirname(project_directory_path)

# Constants
MODEL_PATH = os.path.join(project_directory_path, 'model.h5')


def create_model():
    ''' Loads the VGG16 network from Keras '''
    vgg16_model = vgg16.VGG16()

    return vgg16_model


if __name__ == '__main__':
    # If model does not exist, create it and serialize it
    if not os.path.exists(MODEL_PATH):
        model = create_model()
        model.save(MODEL_PATH)
    else:
        print('>>> Model already exists!')
