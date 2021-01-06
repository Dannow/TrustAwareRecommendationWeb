import numpy as np
from django.http import JsonResponse
from django.shortcuts import HttpResponse
def login(request):
    a = np.array([[1,2,3],[4,5,6]])
    return HttpResponse(a)