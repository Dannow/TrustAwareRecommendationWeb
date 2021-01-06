import numpy as np
import pandas as pd

# 判断年龄
def judgeAge(users,user,i):
    if users[i][2] <= 18:
        user[0] = 1
    elif 18 < users[i][2] <= 29:
        user[1] = 1
    elif 29 < users[i][2] <= 49:
        user[2] = 1
    else:
        user[3] = 1
    return user

# 判断性别
def judgeGender(users,user,i):
    if users[i][1] == 'M':
        user[4] = 1
    else:
        user[5] = 1
    return user

def dataProcessing():
    pd.options.display.max_rows = 10
    # 结果列名
    unames = ['user_id','gender','age','occupation','postal_code']
    # 分割数据
    data = pd.read_table('offline/users.dat', sep='::', header=None, names=unames, engine='python')
    # 把panda的DataFrame转换为numpy的ndarray
    users = data.values

    oneHotUsers = []
    # 获取用户数量
    len = users.shape[0]

    for i in range(len):
        # 初始化
        user = np.zeros(27)
        # 判断年龄
        user = judgeAge(users, user, i)
        # 判断性别
        user = judgeGender(users, user, i)
        # 判断职位
        occupation = users[i][3]
        user[occupation+6] = 1
        oneHotUsers.append(user)
    return oneHotUsers

# 这个获取的数据（6040 X 28），最后一维为ID
def dataProcessing2():
    pd.options.display.max_rows = 10
    # 结果列名
    unames = ['user_id','gender','age','occupation','postal_code']
    # 分割数据
    data = pd.read_table('offline/users.dat', sep='::', header=None, names=unames, engine='python')
    # 把panda的DataFrame转换为numpy的ndarray
    users = data.values

    oneHotUsers = []
    # 获取用户数量
    len = users.shape[0]

    for i in range(len):
        # 初始化
        user = np.zeros(28, dtype=np.int32)
        # 判断年龄
        user = judgeAge(users, user, i)
        # 判断性别
        user = judgeGender(users, user, i)
        # 判断职位
        occupation = users[i][3]
        user[occupation+6] = 1
        user[27] = users[i][0]
        oneHotUsers.append(user)
    return oneHotUsers