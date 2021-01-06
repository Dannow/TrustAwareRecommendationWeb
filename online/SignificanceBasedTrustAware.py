import numpy as np

# 找整个矩阵的最大最小值
def MaxAndMin(userItemRating):
    max = -1
    min = 9999
    # 找最大最小值
    for i in range(userItemRating.shape[0]):
        for j in range(userItemRating.shape[1]):
            # 最大值
            if userItemRating[i][j] > max:
                max = userItemRating[i][j]
            # 最小值
            if -1 < userItemRating[i][j] < min:
                min = userItemRating[i][j]
    return max,min


# 归一化
def Normalized(itemRatingValue,max,min):    # userItemRating：评分矩阵（3953 X 6041）；itemValue：商品（一个数）
    # 归一化公式
    r = (itemRatingValue - min) / (max - min)
    return r


# 商品对用户集的重要性
def Significance(userItemRating,itemID,result,max,min):  # itemID：商品ID，从1开始（一个数），result：被推荐的用户所在的簇(n X 28)
    significance = 0
    # 遍历簇里的用户
    for i in range(len(result)):
        # 获得用户ID
        userID = result[i][27]
        # 有评分时进行累加
        if userItemRating[itemID][userID] != -1:
            significance += Normalized(userItemRating[itemID][userID], max, min)
    return significance / len(result)


# 用户平均归一化后的评分
def UserRatingMean(userItemRating,userID,max,min):    # userID:用户ID（从1开始）
    # 有评分的商品数目
    ratingNumber = 0
    # 累计评分
    sumRating = 0
    # 遍历每个商品查询有评分的商品，并进行累计
    for i in range(userItemRating.shape[0]):
        if userItemRating[i][userID] != -1:
            ratingNumber += 1
            sumRating += Normalized(userItemRating[i][userID], max, min)
    # 返回平均评分
    return sumRating / ratingNumber


# 计算预测评分
def Predict(userItemRating, user_u, user_v, itemRatingValue, max1, min1):  # user_u:用户u的ID；itemRatingValue：候选用户v对i商品的评分
    # 用户u的平均评分
    user_u_rating = UserRatingMean(userItemRating, user_u, max1, min1)
    # 用户v的平均评分
    user_v_rating = UserRatingMean(userItemRating, user_v, max1, min1)
    # 用户v对商品i的评分
    r_v_i = Normalized(itemRatingValue, max1, min1)
    # 求得预测评分
    min2 = min(1, user_u_rating + (r_v_i - user_v_rating))
    p = max(0, min2)
    return p


# 信任度计算
def Trust(userItemRating,userID,candidateNeighborID,result,max,min):     # candidateNeighbor:一个候选邻居ID。userID：需要预测的用户。result：目标用户所在的簇(n X 28)
    # 共同评分商品的数目
    coRatingNum = 0
    # 共同评分商品集合的ID
    co_ratingItem = []
    # 查找目标用户和候选集中candidateNeighborID的用户共同评分的数据
    for i in range(userItemRating.shape[0]):
        if (userItemRating[i][userID] != -1) and (userItemRating[i][candidateNeighborID] != -1):
            co_ratingItem.append(i)
            coRatingNum += 1
    # 累加部分的值
    sum = 0
    # 遍历共同商品（信任度计算公式的累加部分）
    for i in range(len(co_ratingItem)):
        itemID = co_ratingItem[i]
        # 求商品对用户所在簇的重要性
        s_i_k = Significance(userItemRating, itemID, result, max, min)
        # 求用户对商品i的预测
        p = Predict(userItemRating, userID, candidateNeighborID, userItemRating[itemID][candidateNeighborID], max, min)
        # 求用户对商品i的实际评分
        r_u_i = Normalized(userItemRating[itemID][userID], max, min)
        sum += (1 - s_i_k * abs(p - r_u_i))

    # 判断是否有共同评分的数据,有时返回信任值，否则返回-999999
    if sum != 0:
        return (1/coRatingNum)*sum
    else:
        return -999999


