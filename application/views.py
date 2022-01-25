from django.shortcuts import render
from . models import Profile

# Create your views here.

def home_view(request):
    profiles = Profile.objects.all().first()

    context = {'profiles': profiles}

    img_path = str(profiles.dog_image)

    classify_dog_breed(img_path)

    return render(request, 'application/home.html', context)



# Define the model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image

import pickle

with open("model/dog_names", "rb") as fp:   # Unpickling
    dog_names = pickle.load(fp)


### Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('model/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']


### Define the model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


ResNet50_model = Sequential()
ResNet50_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet50_model.add(Dense(133, activation='softmax'))

#ResNet50_model.summary()


# Compile the model 
ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Load the model
ResNet50_model.load_weights('model/weights.best.ResNet50.hdf5')

from tensorflow.keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_mod = ResNet50(weights='imagenet')

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
  # returns prediction vector for image located at img_path
  img = preprocess_input(path_to_tensor(img_path))
  return np.argmax(ResNet50_mod.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
  prediction = ResNet50_predict_labels(img_path)
  return ((prediction <= 268) & (prediction >= 151)) 

def path_to_tensor(img_path):
  #loads RGB image as PIL.Image.Image type
  img = image.load_img(img_path, target_size=(224, 224))
  #convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
  x = image.img_to_array(img)
  #convert 3D tensor into 4D tensor with shape (1, 224, 224, 3)
  return np.expand_dims(x, axis=0)

def extract_Resnet50(tensor):
  return ResNet50(weights='imagenet', include_top=False, pooling = "avg").predict(preprocess_input(tensor))

def dog_breed(img_path):
    #extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    #obtain predicted vector
    predicted_vector = ResNet50_model.predict(bottleneck_feature) #shape error occurs here
    #return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


### A function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

#img_path = dog_files_short[2]
def classify_dog_breed(img_path):
    #determine the predicted dog breed
    breed = dog_breed(img_path)
    #display the image
    #img = cv2.imread(img_path)
    #cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(cv_rgb)
    #plt.show()
    #display relevant predictor result
    if dog_detector(img_path):
      print(str(breed).split('.')[-1])
    else:
      print("Not a dog")