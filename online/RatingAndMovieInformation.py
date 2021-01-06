import numpy as np
import pandas as pd

def dataImport():
    pd.options.display.max_rows = 10

    # 结果列名
    unames1 = ['user_id', 'movie_id', 'rating', 'time']
    # 分割评分表
    ratingData = pd.read_table('online/ratings.dat', sep='::', header=None, names=unames1, engine='python')
    ratings = ratingData.values

    # 结果列名
    unames2 = ['movie_id', 'movie_name', 'classify']
    # 分割电影表
    movieData = pd.read_table('online/movies.dat', sep='::', header=None, names=unames2,engine='python')
    # 转换成ndarray类型
    movies = movieData.values

    # 用户数量
    userItemRating = np.full([3953, 6041], -1)
    for i in range(len(ratings)):
        userID = ratings[i][0]
        moviesID = ratings[i][1]
        rating = ratings[i][2]
        userItemRating[moviesID][userID] = rating
    # 返回评分矩阵（3953 X 6041），其中第一行、第一列没用
    return userItemRating


