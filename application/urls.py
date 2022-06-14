from django.urls import path
from django.shortcuts import redirect
from . import views
from django.views.generic import RedirectView

urlpatterns = [
    path('home', views.home_view, name='home'),
    path('', RedirectView.as_view(pattern_name='home')),
    path('home/', RedirectView.as_view(pattern_name='home')),

    path('login', views.login_view, name='login'),
    path('login/', RedirectView.as_view(pattern_name='login')),

    path('logout', views.logout_view, name='logout'),

    path('register', views.register_view, name='register'),
    path('register/', RedirectView.as_view(pattern_name='register')),

    path('profile', views.profile_view, name='profile'),
    path('profile/', RedirectView.as_view(pattern_name='profile')),
    path('profile/<slug:slug>', views.profile_match_view, name='profile_match_view'),
    path('profile/<slug:slug>/', RedirectView.as_view(pattern_name='profile_match_view')),

    # Check if username exists already (Register)
    path('post_username', views.post_username, name='post_username'),
    path('check_username', views.check_username, name='check_username'),

    # Save changes (Profile)
    path('save_changes', views.save_changes, name='save_changes'),

    # Check login credentials (Login)
    path('post_login_cred', views.post_login_cred, name='post_login_cred'),
    path('check_login_cred', views.check_login_cred, name='check_login_cred'),

    # Get same breed and cross breed (Home)
    path('same_breed', views.same_breed, name='same_breed'),
    path('cross_breed', views.cross_breed, name='cross_breed'),

    # POST and GET username list clicked (Home)
    path('post_username_choice_list', views.post_username_choice_list, name='post_username_choice_list'),
    path('get_username_choice_list', views.get_username_choice_list, name='get_username_choice_list'),

    # (Home)
    path('post_clicked_user_username', views.post_clicked_user_username, name='post_clicked_user_username'),

    # (Matches)
    path('matches', views.matches_view, name='matches'),

    path('get_matched_usernames', views.get_matched_usernames, name='get_matched_usernames'),

    # (Message)
    path('message/<slug:slug>', views.message_view, name="message"),
    path('message/<slug:slug>/', RedirectView.as_view(pattern_name='message')),

    # (Prediction)
    path('puppy/<slug:slug>', views.puppy_view, name="puppy"),
    path('puppy/<slug:slug>/', RedirectView.as_view(pattern_name='puppy')),
]