import numpy as py
from offline.UserCluster import *
from offline.SFLA_KMeans import *
from online.UserClusterIndex import *
from online.RatingAndMovieInformation import *
from online.SignificanceBasedTrustAware import *
from online.RatingPrediction import *
from django.shortcuts import HttpResponse

alpha = 0.6
theta = 0.92

# 推荐商品;userID:从0开始
def Recommendation(userID):
    # 获取离线的人口特征聚类结果。results:每个聚类中的用户（14 X N X 28）；clusterCenters:每个簇的质心（14 X 27）；users：全部用户（6049 X 28）
    results, clusterCenters, users = getCluster()
    user = users[userID-1, :27]
    # 获得用户所在簇的索引(一个数)
    user_cluster_index = getUserClusterIndex(userID, results)

    # 用户所在簇(n X 28)
    user_cluster = results[user_cluster_index]
    print("用户所在簇",user_cluster_index)

    # 1、获得候选邻居
    # 邻居用户(M X N X 28)
    users_neighbor_results = []
    # 候选簇的索引
    candidate_cluster_index = []
    # 计算目标用户到每个簇的质心的相似度
    for i in range(len(clusterCenters)):
        d = Dsim(user, clusterCenters[i, :27])  # 一个数
        if d > alpha:
            users_neighbor_results.append(results[i])
            candidate_cluster_index.append(i)
    # 把邻居用户转换成ndarray
    users_neighbor_results = np.array(users_neighbor_results)
    print("候选簇",candidate_cluster_index)


    # 2、在候选邻居簇中删除目标用户自己
    # 目标用户所在的簇的索引
    user_cluster_index = candidate_cluster_index.index(user_cluster_index)
    # 目标用户所在的簇（N X 28）
    target_user_cluster = users_neighbor_results[user_cluster_index]
    # 遍历用户所在的簇
    for i in range(target_user_cluster.shape[0]):
        if target_user_cluster[i][27] == userID:
            target_user_index = i
            break
    # 删除目标用户本身
    user_delete = np.delete(target_user_cluster, target_user_index, 0)
    # 当为多维时
    if len(candidate_cluster_index) != 1:
        # 把更新后的数组放回users_neighbor_results中(M X N X 28)，没有了用户本身
        # 把更新后的数组放回users_neighbor_results中(M X N X 28)，没有了用户本身
        users_neighbor_results[user_cluster_index] = user_delete
    else:   # 当为2个维度时
        # 把更新后的数组放回users_neighbor_results中(M X N X 28)，没有了用户本身
        users_neighbor_results = np.array([user_delete])
    print("初次候选用户集合", users_neighbor_results)


    #     dsim_result.append(d)
    # # 对相似度进行排序,获得索引
    # dsim_index = np.argsort(np.array(dsim_result))
    #
    # # 符合条件的质心的索引,已经拍好序（1 X M）
    # cluster_neighbor_result = []
    # # 邻居用户(M X N X 28)
    # users_neighbor_results = []
    # # 查找符合的质心和用户
    # for i in range(len(clusterCenters)):
    #     if dsim_result[dsim_index[i]] > alpha:  #dsim_result[dsim_index[i]]: dsim_index中第i个索引对应的相似度值
    #         cluster_neighbor_result.append(dsim_index[i])
    #         users_neighbor_results.append(results[dsim_index[i]])
    # # 把邻居用户转换成ndarray
    # users_neighbor_results = np.array(users_neighbor_results)

    # 2、获得评分矩阵
    userItemRating = dataImport()   #评分矩阵（3953 X 6041），其中第一行、第一列没用

    # 3、获得评分矩阵最大，最小值
    max,min = MaxAndMin(userItemRating)

    # 4、获得商品对用户所在簇的重要性,从1开始
    significance_item_cluster = []
    # 遍历每个商品
    for i in range(1, userItemRating.shape[0]):
        # 调用重要性函数
        significance = Significance(userItemRating, i, user_cluster, max, min)
        significance_item_cluster.append(significance)

    # 5、信任度计算
    # 目标用户到每个邻居的信任值（(MXN) X 2）
    trustValues = []
    # 遍历每个邻居簇
    for i in range(users_neighbor_results.shape[0]):
        users_neighbor_result = users_neighbor_results[i]
        # 遍历每个邻居簇内的用户
        for j in range(users_neighbor_result.shape[0]):
            # 用户信任值对应的ID
            trustValues_userID = []
            # 邻居用户
            candidateNeighbor = users_neighbor_result[j]
            candidateNeighborID = candidateNeighbor[27]
            # 调用信任度函数
            trustValue = Trust(userItemRating, userID, candidateNeighborID, user_cluster, max, min)
            # 保存用户信任值(0位)
            trustValues_userID.append(trustValue)
            # 保存用户ID(1位)
            trustValues_userID.append(candidateNeighborID)
            trustValues.append(trustValues_userID)
    # 转换为ndarray
    trustValues = np.array(trustValues)
    print("trustValues：",trustValues,trustValues.shape)
    # 6、获得最终的候选邻居用户
    # 邻居集合（K X 2）
    neighborhood = []
    for i in range(len(trustValues)):
        # 判断信任值大于theta的为最终邻居用户
        if trustValues[i][0] > theta:
            neighborhood.append(trustValues[i])
    # 转换为ndarray
    neighborhood = np.array(neighborhood)
    print("邻居用户", neighborhood,neighborhood.shape)


    # 7、计算目标用户对未知商品的评分（不用归一化）
    # 目标用户对未评分商品的预测评分
    predictionValues = []   # (L X 2)
    # 遍历每个商品
    for i in range(1,userItemRating.shape[0]):
        predictionValue = []
        # 判断哪个商品目标用户还没有评分的
        if userItemRating[i][userID] == -1:
            # 获得商品预测评分
            predict_value = UserRatingPrediction(userItemRating,i,userID,neighborhood)
            # 放入评分（0位）
            predictionValue.append(predict_value)
            # 放入商品ID（1位）
            predictionValue.append(i)
            # 把[评分，商品ID]放入
            predictionValues.append(predictionValue)
    # 转换为ndarray
    predictionValues = np.array(predictionValues)
    return predictionValues

def main(request):
    userID = 2
    # 调用预测函数
    predictionValues = Recommendation(userID)    # (L X 2)
    # 推荐商品的个数
    RecommendItemNumber = 10
    # 对预测出来的结果进行按第0位倒序（从大到小）排序，返回索引
    sortpredictionValues_index = np.lexsort(-predictionValues[:, ::-1].T)
    # 排序后的预测结果
    sortpredictionValues = predictionValues[sortpredictionValues_index]
    print("预测结果")
    print(sortpredictionValues[:10, :])
    return HttpResponse(sortpredictionValues[:4, :])














