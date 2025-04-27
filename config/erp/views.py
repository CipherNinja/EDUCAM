from django.shortcuts import render

# Create your views here.

def landing_page(request):
    __RESP__ = {}
    return render(
        request,
        "login.html",
        __RESP__
    )