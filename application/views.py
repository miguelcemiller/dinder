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

        # Redirect to login
        return redirect('login')
    return render(request, 'application/register.html')


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

    
