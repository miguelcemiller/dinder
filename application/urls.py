from django.urls import path
from django.shortcuts import redirect
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('cards', views.dog_cards, name='dog_cards'),
]