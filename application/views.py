from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from . models import Profile
from django.core import serializers
from django.http import JsonResponse
import json

from . forms import CustomUserCreationForm

from django.contrib.auth.decorators import login_required

from django.contrib.auth.models import User


from django.core.files import File
from django.urls import reverse_lazy

# Create your views here.

def login_view(request):
    if request.method == 'POST':
        username= request.POST['username']
        password = request.POST['password']

        try:
            user = User.objects.get(username=username)
        except:
            print("Username does not exist")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            print("logged in")

            return redirect('home')
        else:
            print('Username or password is incorrect')
           
    return render(request, 'application/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        first_name = request.POST.get('firstname')
        last_name = request.POST.get('lastname')
        location = request.POST.get('location')
        password = request.POST.get('password')
        user_image = request.FILES['userimage']
        
        dog_name = request.POST.get('dogname')
        dog_age = request.POST.get('dogage')
        dog_description = request.POST.get('dogdescription')
        dog_image = request.FILES['dogimage'] # For form field checking only

        if username and first_name and last_name and user_image and location and password and dog_name and dog_age and dog_description and dog_image: # Check if fields are filled out
            user_obj = User.objects.create(username=username, first_name=first_name, last_name=last_name) # Save User object, then Profile is automatically created
            user_obj.set_password(password)
            user_obj.save()

            # Update Profile user image
            # Source: https://stackoverflow.com/questions/8822755/how-to-save-an-image-using-django-imagefield
            up_file_user = request.FILES['userimage']
            destination_user = open('/tmp/' + up_file_user.name , 'wb+')
            for chunk in up_file_user.chunks():
                destination_user.write(chunk)
            destination_user.close()
            img_user = Profile.objects.get(username=username)
            img_user.user_image.save(up_file_user.name, File(open('/tmp/' + up_file_user.name, 'rb')))
            img_user.save()
            
            # Update Profile object
            prof_obj = Profile.objects.get(username=username)
            prof_obj.location = location
            prof_obj.dog_name = dog_name
            prof_obj.dog_age = dog_age
            prof_obj.dog_description = dog_description
            prof_obj.save()

            # Update Profile dog image
            # Source: https://stackoverflow.com/questions/8822755/how-to-save-an-image-using-django-imagefield
            up_file = request.FILES['dogimage']
            destination = open('/tmp/' + up_file.name , 'wb+')
            for chunk in up_file.chunks():
                destination.write(chunk)
            destination.close()
            img = Profile.objects.get(username=username)
            img.dog_image.save(up_file.name, File(open('/tmp/' + up_file.name, 'rb')))
            img.save()

            # Identify dog breed
            current_user_profile = Profile.objects.get(username=username)
            img_path = str(current_user_profile.dog_image)
            user_dog_breed = classify_dog_breed(img_path)

            # Store dog_breed in databse
            current_user_profile.dog_breed = user_dog_breed
            current_user_profile.save()

            # Login the user
            user = authenticate(username=username, password=password)
            if user is not None:
                if user.is_active:
                    login(request, user)

            # Redirect to home
            return redirect('home')
        else:
            print('Fill out all the fields')
        
    return render(request, 'application/register.html')

# Store here profiles
profiles_dog_breed = []

@login_required(login_url=reverse_lazy("login"))
def home_view(request):
    profiles_dog_breed.clear() # Reset value
    # Get current user profile
    current_user = request.user
    current_user_profile = Profile.objects.get(user=current_user)

    # Get dog breed of user
    user_dog_breed = current_user_profile.dog_breed

    # Loop through all users except current, store profiles with same dog breed
    profiles = Profile.objects.all().exclude(username=current_user_profile.username)

    for i in profiles:
        # Identify dog breed
        i_dog_breed = i.dog_breed

        # Compare loop dog breed to user dog breed 
        if i_dog_breed == user_dog_breed:
            profiles_dog_breed.append(i)

    print("User's dog breed:", user_dog_breed)
    print("Users with same breed:",profiles_dog_breed)  
    print("Length:", len(profiles_dog_breed))    

    context = {'current_user_profile': current_user_profile, 'profiles_dog_breed_len': len(profiles_dog_breed)}

    return render(request, 'application/home.html', context)

def dog_cards(request):
    profiles_serializers = serializers.serialize("json", profiles_dog_breed)

    response = json.loads(profiles_serializers)
    return JsonResponse({'response': response}, status=200)


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
      #print(str(breed).split('.')[-1])
      return str(breed).split('.')[-1]
    else:
      #print("Not a dog")
      return "not dog"

    