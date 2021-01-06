import numpy as np
from online.RecommendedItem import *
from django.shortcuts import HttpResponse

def getUserHaveRating(userID):
    k = 0
    userHaveRating = []
    # 获得评分矩阵
    userItemRating = dataImport()
    # 获得对应userID有评分的电影
    for i in range(1, userItemRating.shape[0]):
        if userItemRating[i][userID] != -1:
            userHaveRating.append(i)
            userHaveRating.append(-1)
            k += 1
        if k == 4:
            return userHaveRating


def UserHaveRatingMain(request):
    userHaveRating = getUserHaveRating(2)
    return HttpResponse(userHaveRating)