from django.urls import path
from .views import *
urlpatterns = [
    path('',landing_page,name="login"),
]
