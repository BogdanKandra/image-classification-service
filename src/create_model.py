import os
from tensorflow.keras.applications import vgg16


# Constants
if os.path.basename(os.getcwd()) == 'src':
    MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()), 'model.h5')
else:
    MODEL_PATH = 'model.h5'

def create_model():
    ''' Loads the VGG16 network from Keras '''
    vgg16_model = vgg16.VGG16()

    return vgg16_model


if __name__ == '__main__':
    # If model does not exist, create it and serialize it
    if not os.path.exists(MODEL_PATH):
        model = create_model()
        model.save('model.h5')
    else:
        print('>>> Model already exists!')
