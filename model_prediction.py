import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

# Load Keras' VGG16 model that was pre-trained against the ImageNet database
vgg16_model = vgg16.VGG16()

# Load an image file, resizing it to 224x224 pixels (original input size required by this model)
img = image.load_img('bay.jpg', target_size=(224, 224))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
x_vgg = vgg16.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
vgg_predictions = vgg16_model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
vgg_predicted_classes = vgg16.decode_predictions(vgg_predictions, top=10)

print('This is an image of:')

print('According to VGG:')
for imagenet_id, name, likelihood in vgg_predicted_classes[0]:
    print(' - {}: {:2f} likelihood'.format(name, likelihood))
