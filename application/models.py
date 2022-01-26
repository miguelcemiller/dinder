from django.db import models
import uuid
from django.contrib.auth.models import User

from django.db.models.signals import pre_save, post_save, post_delete
from django.dispatch import receiver

# Create your models here.
LOCATIONS = (
    ('Alaminos', 'Alaminos'),
    ('Bay', 'Bay'),
    ('Biñan', 'Biñan'),
    ('Cabuyao', 'Cabuyao'),
    ('Calamba', 'Calamba'),
    ('Calauan', 'Calauan'),
    ('Cavinti', 'Cavinti'),
    ('Famy', 'Famy'),
    ('Kalayaan', 'Kalayaan'),
    ('Liliw', 'Liliw'),
    ('Los Baños', 'Los Baños'),
    ('Luisiana', 'Luisiana'),
    ('Lumban', 'Lumban'),
    ('Mabitac', 'Mabitac'),
    ('Magdalena', 'Magdalena'),
    ('Majayjay', 'Majayjay'),
    ('Nagcarlan', 'Nagcarlan'),
    ('Paete', 'Paete'),
    ('Pagsanjan', 'Pagsanjan'),
    ('Pakil', 'Pakil'),
    ('Pangil', 'Pangil'),
    ('Pila', 'Pila'),
    ('Rizal', 'Rizal'),
    ('San Pablo', 'San Pablo'),
    ('San Pedro', 'San Pedro'),
    ('Santa Cruz', 'Santa Cruz'),
    ('Santa Maria', 'Santa Maria'),
    ('Santa Rosa', 'Santa Rosa'),
    ('Siniloan', 'Siniloan'),
    ('Victoria', 'Victoria'),
)

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    username = models.CharField(max_length=25, blank=False, null=True)
    first_name = models.CharField(max_length=200, blank=True, null=True)
    last_name = models.CharField(max_length=200, blank=True, null=True)
    user_image = models.ImageField(null=True, blank=True, upload_to='static/images/users/')

    dog_name = models.CharField(max_length=200, blank=True, null=True)
    dog_age = models.IntegerField(blank=True, null=True)
    dog_description = models.TextField(blank=True, null=True)
    dog_image = models.ImageField(null=True, blank=True, upload_to='static/images/dogs/')

    location = models.CharField(max_length=20, choices=LOCATIONS, blank=False, null=False)

    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True, primary_key=True, editable=False)

    def __str__(self):
        return str(self.user.username)

# Create Profile when User is created
@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        user = instance
        profile = Profile.objects.create(
            user = user,
            username = user.username
        )
    print("this is fired off")

# Delete User when Profile is deleted
@receiver(post_delete, sender=Profile)
def delete_student(sender, instance, **kwargs):
    user = instance.user
    user.delete()