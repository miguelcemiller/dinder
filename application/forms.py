from django.forms import ModelForm
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

from . models import Profile

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = Profile
        fields = ['first_name', 'last_name', 'username', 'user_image']