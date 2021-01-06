import numpy as np

# 用户平均评分（不用归一化）
def UserRatingMeanNotNormalized(userItemRating,userID):    # userID:用户ID（从1开始）
    # 有评分的商品数目
    ratingNumber = 0
    # 累计评分
    sumRating = 0
    # 遍历每个商品查询有评分的商品，并进行累计
    for i in range(userItemRating.shape[0]):
        if userItemRating[i][userID] != -1:
            ratingNumber += 1
            sumRating += userItemRating[i][userID]
    # 返回平均评分
    return sumRating / ratingNumber


# 计算目标用户对未知商品的评分（不用归一化）
def UserRatingPrediction(userItemRating,itemID,userID,neighborhood):     #userItemRating：评分矩阵（3953 X 6041），其中第一行、第一列没用；trustValues：（(MXN) X 2）；itemID:商品ID；userID：目标用户ID；neighborhood：邻居用户集合（K X 2）
    molecular = 0
    denominator = 0
    for i in range(neighborhood.shape[0]):
        # 取出邻居用户ID
        neighborhoodID = neighborhood[i][1].astype(int)
        if userItemRating[itemID][neighborhoodID] != -1:
            # 计算邻居用户的平均评分
            neighborhoodRatingMean = UserRatingMeanNotNormalized(userItemRating,neighborhoodID)
            # 分子部分的累加操作
            molecular += neighborhood[i][0] * (userItemRating[itemID][neighborhoodID] - neighborhoodRatingMean)
            # 分母部分的累加操作
            denominator += neighborhood[i][0]
    # 目标用户的平均评分
    userRatingMean = UserRatingMeanNotNormalized(userItemRating,userID)
    # 判断是否有全部邻居都没有对该商品评分的可能
    if (denominator != 0):
        return userRatingMean + (molecular / denominator)
    else:
        return -999999
