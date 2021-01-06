import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from offline.DemographicInformation import dataProcessing
from offline.KMeans import *

# 设置主题
sns.set_style("darkgrid")
sns.set_context("notebook")

# 获取用户特征
users = np.array(dataProcessing())    # 6040 X 27
eta = 0.45

# 对一维数据求2范数
def Norm(x):
    return np.linalg.norm(x, 2)   # 求2范数

# 相似度计算
def Dsim(user,clusterCenter):   # user:一个用户（1 X 27）；clusterCenter：聚类中心（1 X 27）
    molecular = np.sum(user * clusterCenter)  # 一个数
    Denominator = Norm(user) * Norm(clusterCenter)    # 一个数
    c = molecular/Denominator
    return c    # 一个数


# 适应度函数
def opt_func(value):  # value:一只青蛙（14 X 27）
    output = 0
    # 调用K-Means
    results,clusterCenters = K_Means(value, users)   # value：初始质心，既青蛙（14 X 27）；users：全部用户数据（6040 X 27）；results：已经归到每个簇的结果集（14 X n X 27）；clusterCenters：每个簇的质心（14 X 27）
    # 遍历每个簇
    for i in range(len(clusterCenters)):
        # 第i个簇的质心(1 X 27)
        clusterCenter = clusterCenters[i]
        # 第i个簇里的用户数据（n X 27）
        result = results[i]
        # 遍历每个簇内的用户
        for j in range(len(result)):
            user = result[j]    # 一个用户（1 X 27）
            # 结果累加
            output = output + (1 - Dsim(user, clusterCenter))
    return output


# 用高斯正态分布随机生成青蛙
def gen_frogs(frogs, dimension, sigma):  # frogs:青蛙数（int）; dimension:维度（K维）; sigma：高斯分布的Sigma（默认1）
    # frogs = sigma * (np.random.randn(frogs, dimension))  # 生成一个符合正态分布的（frogs X dimension）维的矩阵
    frogs_index = sigma * (np.random.randint(0, 6040, (frogs, dimension)))  # 生成一个符合正态分布的（300 X 14）维的矩阵在[0-6040]之间,都是对应users里的索引
    frogs = users[frogs_index]
    return frogs    # 返回青蛙矩阵（300 X 14 X 27）维矩阵


# 对青蛙进行降序排序
def sort_frogs(frogs, mplx_no, opt_func):  # forgs:要排序的青蛙（300 X 14）；mplx_no:族群数量；opt_func:适应度函数
    # 计算每只青蛙的适应值
    fitness = np.array(list(map(opt_func, frogs)))  # fitness:(1 X 300); frogs:(300 X 14 X 27)
    # 按适应度降序对索引进行排序
    sorted_fitness = np.argsort(fitness)    # argsort返回的是排序后的索引值
    # 初始化族群
    memeplexes = np.zeros((mplx_no, int(frogs.shape[0] / mplx_no)))     # 20 X (300/20)[既每个族群中每个青蛙数量]
    # 把青蛙分给不同的族群，一列一列的将青蛙放进memeplexes中
    for j in range(memeplexes.shape[1]):    #shape[1]是列数
        for i in range(mplx_no):
            memeplexes[i, j] = sorted_fitness[i + (mplx_no * j)]
    return memeplexes   # 20 X 15，[0，0]将是最大的青蛙


# 局部搜索
def local_search(frogs, memeplex, opt_func, sigma, dimension):
    # 选择族群内最坏的青蛙
    frog_w = frogs[int(memeplex[-1])]   # 因为sort_frogs，最差青蛙都是在族群最后一个(14 X 27)
    # 选择族群内最好的青蛙
    frog_b = frogs[int(memeplex[0])]    # 因为sort_frogs，最好青蛙都是在族群最后一个(14 X 27)
    # 全局最好的青蛙
    frog_g = frogs[0]
    # 跳跃步长
    rand = np.random.rand()
    delta = rand * (frog_b - frog_w)  # users[frog_b]:(14 X 27)
    t = 1/(1+np.exp(-delta))    # t:(14 X 27)
    # 移动最坏的青蛙
    temp = np.zeros((14, 27))   #初始化(14 X 27)
    # 根据条件设置frog_w_new
    temp[np.where(t <= eta)] = 0
    temp[np.where((t > eta) & (t < 1/2 * (1 + eta)))] = frog_w[np.where((t > eta) & (t < 1/2 * (1 + eta)))]
    temp[np.where(t >= 1/2 * (1 + eta))] = 1
    # 用全局最优替换局部最优
    if opt_func(temp) < opt_func(frog_w):
        frog_w_new = temp
    # 随机移动青蛙
    else:
        delta = rand * (frog_g - frog_w)
        t = 1 / (1 + np.exp(-delta))  # t:(14 X 27)
        # 移动最坏的青蛙
        temp = np.zeros((14, 27))  # 初始化(14 X 27)
        # 根据条件设置frog_w_new
        temp[np.where(t <= eta)] = 0
        temp[np.where((t > eta) & (t < 1 / 2 * (1 + eta)))] = frog_w[np.where((t > eta) & (t < 1 / 2 * (1 + eta)))]
        temp[np.where(t >= 1 / 2 * (1 + eta))] = 1
        if opt_func(temp) < opt_func(frog_w):
            frog_w_new = temp
        else:
            frogs_index = sigma * (np.random.randint(0, 6040, (1, dimension)))
            frog = users[frogs_index]
            frog_w_new = frog
    # 覆盖原本适应值最差的青蛙
    frogs[int(memeplex[-1])] = frog_w_new   # 当上面的条件都不满足时，既为向族群中最好的青蛙跳。frog_w_new:(1 X 14)可能里面的值是小数; frogs:(300 X 14)，会自动把小数向前取整
    return frogs    # frogs:(300 X 14)


# 混合族群，而不对其进行排序。
def shuffle_memeplexes(memeplexes):
    # 展平数组
    temp = memeplexes.flatten()
    # 随机排列
    np.random.shuffle(temp)
    # 重新组合成原来的维度（mplx_no X 每个族群中每个青蛙数量）
    temp = temp.reshape((memeplexes.shape[0], memeplexes.shape[1]))
    return temp     # 20 X 15


# 蛙跳算法
def sfla(opt_func, frogs, dimension, sigma, mplx_no, mplx_iters, solun_iters):   #mplx_iters：族群内局部搜索次数；solun_iters:族群重组次数；
    # 产生青蛙
    frogs = gen_frogs(frogs, dimension, sigma)  # (300 X 14 X 27)
    # 整理并把青蛙分为不同的族群进行排序
    memeplexes = sort_frogs(frogs, mplx_no, opt_func)   # (20 X 15)
    # 全局最优的解
    best_solun = frogs[int(memeplexes[0, 0])]   # (14 X 27)

    # 将族群进行重组（全局搜索）
    for i in range(solun_iters):
        # 混合族群
        if i != 0:
            # 对族群内的青蛙进行排序、分组
            memeplexes = sort_frogs(frogs, mplx_no, opt_func)   #(20 X 15)里面是每只青蛙的索引
            # 选出全局最优的青蛙
            new_best_solun = frogs[int(memeplexes[0, 0])]   # (14 X 27)
            # 选出全局最优解
            if opt_func(new_best_solun) < opt_func(best_solun):
                best_solun = new_best_solun
        # 遍历族群（局部搜索）
        for mplx_idx, memeplex in enumerate(memeplexes):
            # 每个族群内青蛙跳的次数
            for j in range(mplx_iters):
                # 进行族群内的局部搜索
                frogs = local_search(frogs, memeplex, opt_func, sigma, dimension)
        memeplexes = shuffle_memeplexes(memeplexes)
    return best_solun, frogs, memeplexes.astype(int)

# 主函数
def main():
    # 调用随机蛙跳算法
    solun, frogs, memeplexes = sfla(opt_func, 300, 14, 1, 20, 15, 100)

    # 输出结果
    print("Optimal Solution (closest to zero): {}".format(solun))
    print(opt_func(solun))

    # 画出全部青蛙（全部青蛙都会被标记出来）
    for idx, memeplex in enumerate(memeplexes):
        plt.scatter(frogs[memeplex, 0], frogs[memeplex, 1], marker='x', label="memeplex {}".format(idx))
    plt.scatter(solun[0], solun[1], marker='o', label="Optimal Solution")
    plt.scatter(0, 0, marker='*', label='Actual Solution')
    # 画图
    plt.legend()
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Shuffled Frogs")
    # 展示
    plt.show()

# 直接运行该文件时会执行里面的代码；当被其他py文件import时，不会执行里面的代码
if __name__ == '__main__':
    main()