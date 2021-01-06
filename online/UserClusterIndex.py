import numpy as np

# 获取用户所在的簇
def getUserClusterIndex(userID,results):
    # 遍历每个簇
    for i in range(results.shape[0]):
        result = results[i]     # （N X 28）
        # 遍历簇内的每个用户
        for j in range(result.shape[0]):
            user = result[j]    # （1 X 28）
            # 判断符合条件时，返回簇的索引
            if user[27] == userID:
                return i
