from django.db import models
import uuid
from django.contrib.auth.models import User

from django.db.models.signals import pre_save, post_save, post_delete
from django.dispatch import receiver

# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    username = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    name = models.CharField(max_length=200, blank=True, null=True)
    image = models.ImageField(null=True, blank=True, upload_to='static/images/users/')
    gender = models.CharField(max_length=100, blank=True, null=True)
    year = models.IntegerField(blank=True, null=True)
    month = models.IntegerField(blank=True, null=True)
    about = models.TextField(blank=True, null=True)
    province = models.CharField(max_length=200, blank=True, null=True)
    city = models.CharField(max_length=200, blank=True, null=True)

    breed = models.CharField(max_length=200, blank=True, null=True)

    username_list_clicked = models.TextField(blank=True, null=True)
    matched_usernames = models.TextField(blank=True, null=True)
    
    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True, primary_key=True, editable=False)

    def __str__(self):
        if self.user:
            return str(self.user.username) or ''

class Message(models.Model):
    group = models.CharField(max_length=100, blank=True, null=True)
    sender = models.CharField(max_length=100, blank=True, null=True)
    receiver = models.CharField(max_length=100, blank=True, null=True)
    message = models.TextField(blank=True, null=True)
    order = models.IntegerField(blank=True, null=True)

    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True, primary_key=True, editable=False)

    def __str__(self):
        return str(self.group) or ''

# Create Profile when User is created
@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        user = instance
        profile = Profile.objects.create(
            user = user,
            username = user.username,
        )
    print("this is fired off")

# Delete User when Profile is deleted
@receiver(post_delete, sender=Profile)
def delete_student(sender, instance, **kwargs):
    user = instance.user
    user.delete()
