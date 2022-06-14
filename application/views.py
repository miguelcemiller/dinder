from site import USER_SITE
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.core import serializers
from django.http import JsonResponse
import json

from django.contrib.auth.decorators import login_required

from django.contrib.auth.models import User
from . models import Profile
from . models import Message

from django.core.files import File
from django.urls import reverse_lazy

import ast
# Create your views here.

def register_view(request):

    if request.method == 'POST':
        name = request.POST.get('name')
        age = [request.POST.get('year'), (request.POST.get('month'))]
        gender = request.POST.get('gender')
        image = request.FILES['image']
        username = request.POST.get('username')
        password = request.POST.get('password')
        province = request.POST.get('province')
        city = request.POST.get('city')
        about = request.POST.get('about')

        # Create User object -> Profile object is created
        user_obj = User.objects.create(username=username) 
        user_obj.set_password(password)
        user_obj.save()

        # Update Profile object
        prof_obj = Profile.objects.get(username=username)
        prof_obj.password = password
        prof_obj.name = name
        prof_obj.year = age[0]
        prof_obj.month = age[1]
        prof_obj.gender = gender
        prof_obj.province = province
        prof_obj.city = city
        prof_obj.about = about
        prof_obj.save()

        # Update Profile object image # Source: https://stackoverflow.com/questions/8822755/how-to-save-an-image-using-django-imagefield
        up_file_user = request.FILES['image']
        destination_user = open('/tmp/' + up_file_user.name , 'wb+')
        for chunk in up_file_user.chunks():
            destination_user.write(chunk)
        destination_user.close()
        prof_obj.image.save(up_file_user.name, File(open('/tmp/' + up_file_user.name, 'rb')))
        prof_obj.save()
        
        # Identify breed
        img_path = str(prof_obj.image)
        user_dog_breed = classify_dog_breed(img_path)
        # Store dog_breed in database
        prof_obj.breed = user_dog_breed
        prof_obj.save()

        print(user_dog_breed)

        # Redirect to login
        return redirect('login')
    return render(request, 'application/register.html')



'''CNN start'''
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
''' CNN end'''

def login_view(request):
    if request.user.is_authenticated:
        if request.user.is_superuser == False:
            return redirect('home')

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

# Check login credentials
post_login_username = ""
post_login_password = ""
def post_login_cred(request):
    global post_login_username, post_login_password
    post_login_username = request.POST['username']
    post_login_password = request.POST['password']

    print(post_login_username, post_login_password)

    response = {
        'username' : post_login_username,
        'password' : post_login_password
    }

    return JsonResponse(response)

def check_login_cred(request):
    global post_login_password, post_login_password
    user = authenticate(request, username=post_login_username, password=post_login_password)

    val = ''
    if user is not None:
        val = True
    else:
        val = False
    
    response = {'val': val}
    return JsonResponse(response, status=200)

@login_required(login_url=reverse_lazy("login"))
def home_view(request):
    # If superuser, redirect to login
    if request.user.is_superuser:
        return redirect('login')

    user = request.user
    user_profile = Profile.objects.get(user=user)
    
    if user_profile.matched_usernames:
        matched_usernames = ast.literal_eval(user_profile.matched_usernames)
        matches_len = len(matched_usernames)
    else: 
        matched_usernames = []
        matches_len = len(matched_usernames)

    context = {'user_profile' : user_profile, 'matches_len': matches_len}
    return render(request, 'application/home.html', context)

def get_matched_usernames(request):
    user = request.user
    user_profile = Profile.objects.get(username=user)

    data = user_profile.matched_usernames
    print("This is the data", data)

    if data:
        response = ast.literal_eval(data)
    else:
        response = []

    return JsonResponse({'response': response}, status=200)

def post_clicked_user_username(request):
    clicked_username = request.POST['clickedUserUsername']
    clicked_user_profile = Profile.objects.get(username=clicked_username)
    user_profile = Profile.objects.get(username=request.user)

    to_POST = request.POST['toPOST']

    print("to_POST value: ",to_POST)

    if to_POST == 'true':
        # Add to database username if not included already
        if user_profile.matched_usernames:
            matched_usernames_list = ast.literal_eval(user_profile.matched_usernames)
            if clicked_username not in matched_usernames_list and clicked_username is not request.user:
                matched_usernames_list.append(clicked_username)
                user_profile.matched_usernames = matched_usernames_list
                user_profile.save()
            matches_len = len(matched_usernames_list)
        else:
            matched_usernames_list = []
            matched_usernames_list.append(clicked_username)
            user_profile.matched_usernames = matched_usernames_list
            user_profile.save()
            matches_len = len(matched_usernames_list)
    else:
        matches_len = 0 # set to 0 for navbar only

    # If username_list_clicked is empty
    if clicked_user_profile.username_list_clicked:
        username_list_clicked = json.loads(clicked_user_profile.username_list_clicked)
    else: 
        username_list_clicked = []


    data = [{'username': clicked_user_profile.username}, {'name': clicked_user_profile.name}, {'image': str(clicked_user_profile.image)}, {'username_list_clicked': username_list_clicked}, {'matches_len': matches_len}]
    if data:
        response = data
    else:
        response = []

    return JsonResponse({'response': response}, status=200)

@login_required(login_url=reverse_lazy("login"))
def matches_view(request):
    # If superuser, redirect to login
    if request.user.is_superuser:
        return redirect('login')

    user_profile = Profile.objects.get(username=request.user)

    if user_profile.matched_usernames:
        matched_usernames = ast.literal_eval(user_profile.matched_usernames)
        matches_len = len(matched_usernames)
    else: 
        matched_usernames = []
        matches_len = len(matched_usernames)
    
    matches_obj = []
    # Get matched_usernames data (image, name, username)
    if matches_len:
        for username in matched_usernames:
            match_profile = Profile.objects.get(username=username)
            match_profile_dict = {
                'image': str(match_profile.image),
                'name': match_profile.name,
                'username': match_profile.username
            }
            matches_obj.append(match_profile_dict)
    print(matches_obj)

    context = {'user_profile': user_profile, 'matches_len': matches_len, 'matches_obj': str(matches_obj)}
    return render(request, 'application/matches.html', context)

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required(login_url=reverse_lazy("login"))
def profile_view(request):
    # If superuser, redirect to login
    if request.user.is_superuser:
        return redirect('login')
    
    user_profile = Profile.objects.get(user=request.user)

    if user_profile.matched_usernames:
        matched_usernames = ast.literal_eval(user_profile.matched_usernames)
        matches_len = len(matched_usernames)
    else: 
        matched_usernames = []
        matches_len = len(matched_usernames)

    context = {'user_profile' : user_profile, 'matches_len': matches_len}
    return render(request, 'application/profile.html', context)

def profile_match_view(request, slug):
    # If superuser, redirect to login
    if request.user.is_superuser:
        return redirect('login')
    
    user_profile = Profile.objects.get(username=request.user)
    clicked_user_profile = Profile.objects.get(username=slug)

    # If not matched, (in the case of just pasting the url message/husky)
    if str(slug) not in user_profile.matched_usernames and str(request.user) not in clicked_user_profile.matched_usernames:
        return redirect('matches')

    if user_profile.matched_usernames:
        matched_usernames = ast.literal_eval(user_profile.matched_usernames)
        matches_len = len(matched_usernames)
    else: 
        matched_usernames = []
        matches_len = len(matched_usernames)

    
    context = {'user_profile': user_profile, 'matches_len': matches_len, 'clicked_user_profile': clicked_user_profile}
    return render(request, 'application/profile-match.html', context)

def save_changes(request):
    name = request.POST['name']
    year = request.POST['year']
    month = request.POST['month']
    gender = request.POST['gender']
    username = request.POST['username']
    password = request.POST['password']
    province = request.POST['province']
    city = request.POST['city']
    about = request.POST['about']

    # Update User
    user = request.user
    user.set_password(password)
    user.save()

    # Update Profile
    Profile.objects.filter(username=username).update(name=name)
    Profile.objects.filter(username=username).update(year=year)
    Profile.objects.filter(username=username).update(month=month)
    Profile.objects.filter(username=username).update(gender=gender)
    Profile.objects.filter(username=username).update(password=password)
    Profile.objects.filter(username=username).update(province=province)
    Profile.objects.filter(username=username).update(city=city)
    Profile.objects.filter(username=username).update(about=about)

    # Login user
    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        print("logged in")

    response = {
        'success' : 'success'
    }
    return JsonResponse(response)

def same_breed(request):
    user_profile = Profile.objects.get(username=request.user)

    # User breed
    breed = user_profile.breed
    # User gender
    gender = user_profile.gender

    # Same breed profiles, different username, opposite gender
    same_breed_profiles = Profile.objects.filter(breed=breed).exclude(username=request.user).exclude(gender=gender)

    data = {}
    if len(same_breed_profiles) == 0:
        print("no users found")
        response = []
    else: 
        data = serializers.serialize('json', same_breed_profiles)
        response = json.loads(data)

    return JsonResponse({'response': response}, status=200)

def cross_breed(request):
    user_profile = Profile.objects.get(username=request.user)

    # User breed
    breed = user_profile.breed
    # User gender
    gender = user_profile.gender

    # Cross breed profiles, different username, opposite gender
    cross_breed_profiles = Profile.objects.filter().exclude(breed=breed).exclude(username=request.user).exclude(gender=gender)

    data = {}
    if len(cross_breed_profiles) == 0:
        print("no users found")
        response = []
    else: 
        data = serializers.serialize('json', cross_breed_profiles)
        response = json.loads(data)
    
    return JsonResponse({'response': response}, status=200)

def post_username_choice_list(request):
    username_list = request.POST['usernamelist']
    user = request.user

    # Update Profile username list
    Profile.objects.filter(username=user).update(username_list_clicked=username_list)

    response = {
        'success' : username_list
    }
    return JsonResponse(response)

def get_username_choice_list(request):
    user_profile = Profile.objects.get(username=request.user)

    data = user_profile.username_list_clicked

    if data:
        response = json.loads(data)
    else:
        response = []

    return JsonResponse({'response': response}, status=200)


post_username = ""
def post_username(request):
    global post_username
    post_username = request.POST['username']

    response = {
        'username' : post_username,
    }
    return JsonResponse(response)


def check_username(request):
    global post_username 
    val = User.objects.filter(username=post_username).exists()

    response = {'val': val}
    return JsonResponse(response, status=200)

@login_required(login_url=reverse_lazy("login"))
def message_view(request, slug):
    # If superuser, redirect to login
    if request.user.is_superuser:
        return redirect('login')
    
    user_profile = Profile.objects.get(username=request.user)
    clicked_user_profile = Profile.objects.get(username=slug)

    # If not matched, (in the case of just pasting the url message/husky)
    if str(slug) not in user_profile.matched_usernames and str(request.user) not in clicked_user_profile.matched_usernames:
        return redirect('matches')

    if user_profile.matched_usernames:
        matched_usernames = ast.literal_eval(user_profile.matched_usernames)
        matches_len = len(matched_usernames)
    else: 
        matched_usernames = []
        matches_len = len(matched_usernames)


    # Search for sender, receiver group
    sender = str(request.user)
    receiver = str(slug)

    group = sender + " to " + receiver
    group_alt = receiver + " to " + sender
    groups = [group, group_alt]

    group_messages = Message.objects.filter(group__in=groups).order_by('order')

    # Send message
    if request.method == 'POST':
        message = request.POST['message']
        next_order = len(group_messages) + 1
        
        obj = Message.objects.create(group=group)
        obj.sender = sender
        obj.receiver = receiver
        obj.message = message
        obj.order = next_order
        obj.save()

    group_messages = Message.objects.filter(group__in=groups).order_by('order')
    

    context = {'user_profile': user_profile, 'matches_len': matches_len, 'clicked_user_profile': clicked_user_profile, 'group_messages': group_messages, 'sender': sender}
    return render(request, 'application/message.html', context)

    
@login_required(login_url=reverse_lazy("login"))
def puppy_view(request, slug):
    # If superuser, redirect to login
    if request.user.is_superuser:
        return redirect('login')
    
    user_profile = Profile.objects.get(username=request.user)
    clicked_user_profile = Profile.objects.get(username=slug)

    # If not matched, (in the case of just pasting the url message/husky)
    if str(slug) not in user_profile.matched_usernames and str(request.user) not in clicked_user_profile.matched_usernames:
        return redirect('matches')

    if user_profile.matched_usernames:
        matched_usernames = ast.literal_eval(user_profile.matched_usernames)
        matches_len = len(matched_usernames)
    else: 
        matched_usernames = []
        matches_len = len(matched_usernames)

    context = {'user_profile': user_profile, 'matches_len': matches_len, 'clicked_user_profile': clicked_user_profile}
    return render(request, 'application/puppy.html', context)
