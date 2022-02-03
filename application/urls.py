from django.urls import path
from django.shortcuts import redirect
from . import views
from django.views.generic import RedirectView

urlpatterns = [
    path('', RedirectView.as_view(pattern_name='home')),
    path('home', views.home_view, name='home'),
    path('login', views.login_view, name='login'),
    path('register', views.register_view, name='register'),
    path('logout', views.logout_view, name='logout'),

    path('cards', views.dog_cards, name='dog_cards'),
]