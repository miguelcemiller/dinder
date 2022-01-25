from django.shortcuts import render

from . models import Profile

# Create your views here.

def home_view(request):
    profiles = Profile.objects.all().first()

    context = {'profiles': profiles}

    print(profiles.dog_image)

    return render(request, 'application/home.html', context)