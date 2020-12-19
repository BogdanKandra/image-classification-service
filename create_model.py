import os
from tensorflow.keras.applications import vgg16


# Constants
MODEL_PATH = 'model.pkl'

def create_model():
    ''' Loads the VGG16 network from Keras '''
    vgg16_model = vgg16.VGG16()

    return vgg16_model

if __name__ == '__main__':
    # If model does not exist, create it and serialize it
    if not os.path.exists(MODEL_PATH):
        model = create_model()
        model.save('model.h5')
