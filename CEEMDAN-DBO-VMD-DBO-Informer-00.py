#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os                           # 导入os模块用于操作系统相关函数
import copy                         # 导入copy模块用于创建对象的副本
import numpy                        # 导入numpy库进行数值操作
import random                       # 导入random模块用于生成随机数
import datetime                     # 导入datetime模块用于处理日期和时间
import math                         # 导入math模块进行数学运算
import pandas as pd                 # 导入pandas库进行数据操作和分析
import numpy as np                  # 导入numpy并使用别名"np"以方便使用
import matplotlib.pyplot as plt    # 导入matplotlib库进行数据可视化操作
from PyEMD import CEEMDAN           # 从PyEMD库导入CEEMDAN模块
from pyroapi import pyro
from torch import optim
from sampen import sampen2          # 从sampen库导入sampen2模块
from vmdpy import VMD               # 从vmdpy库导入VMD模块
import tensorflow as tf             # 导入tensorflow库进行机器学习和深度学习
from sklearn.cluster import KMeans  # 从sklearn.cluster库导入KMeans模块
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # 导入评估指标函数
from sklearn.preprocessing import MinMaxScaler    # 从sklearn.preprocessing库导入MinMaxScaler模块用于特征缩放
import warnings                    # 导入warnings库用于忽略警告
from scipy.fftpack import hilbert, fft, ifft  # 导入傅里叶变换相关函数
from math import log                # 导入log函数
from typing import List             # 导入List类型用于类型提示
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data.dataset import MyDataset
from model.informer import Informer
from utils.setseed import set_seed

warnings.filterwarnings("ignore")    # 忽略警告消息


# In[3]:


# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False


# In[4]:


df_raw_data = pd.read_csv('/kaggle/working/dbo-inf/data/ETT/ETTh1.csv', usecols=[0, 7])  # 从名为'ETTh1.csv'的CSV文件中读取数据，只使用第一列和第二列的数据创建DataFrame对象
X='OT'
# df_raw_data = pd.read_csv("/kaggle/working/dbo-inf/data/ETT/ETTh1.csv")
# X = 'OT'  # 将字符串'OT'赋值给变量X，表示使用该列作为特征
#
series_close = pd.Series(df_raw_data[X].values, index=df_raw_data['date'])  # 使用列名为X的数据创建Series对象，使用'Date'列作为索引
#
test = df_raw_data[X].values[int(len(df_raw_data[X].values)*0.7):]  # 从X列的数据中提取后80%部分，并将结果存储在test变量中
#

# In[5]:


timestep = 30  # 定义时间步数，用于创建训练集和测试集
tau = 0.  # 设置VMD分解的参数tau，用于控制模态函数的带宽
DC = 0  # 设置VMD分解的参数DC，用于控制是否提取直流分量
init = 1  # 设置VMD分解的参数init，用于设置初始模态函数数量
tol = 1e-7  # 设置VMD分解的参数tol，用于控制停止迭代的阈值


# In[6]:


''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])  # 创建一个形状为 (pop, dim) 的全零数组 X，用于存储种群的初始位置
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]  # 生成一个位于 lb[j] 和 ub[j] 之间的随机数，赋值给 X[i, j]
    return X, lb, ub

'''边界检查函数'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:  # 如果 X[i, j] 大于上界 ub[j]，将其设置为上界
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:  # 如果 X[i, j] 小于下界 lb[j]，将其设置为下界
                X[i, j] = lb[j]
    return X

'''计算适应度函数'''
def CaculateFitness(X, fun):
    pop = X.shape[0]  # 种群大小
    fitness = np.zeros([pop, 1])  # 创建一个形状为 (pop, 1) 的全零数组 fitness，用于存储适应度值
    for i in range(pop):
        fitness[i] = fun(X[i, :])  # 计算第 i 个个体的适应度值，将结果赋值给 fitness[i]
    return fitness

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)  # 按列排序适应度值数组 Fit，将结果赋值给 fitness
    index = np.argsort(Fit, axis=0)  # 返回按列排序后的索引数组，将结果赋值给 index
    return fitness, index

'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)  # 创建一个与位置数组 X 相同形状的全零数组 Xnew
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]  # 根据索引数组 index 对位置数组 X 进行排序，并将结果赋值给 Xnew
    return Xnew


# In[7]:


def DBO(pop, dim, lb, ub, MaxIter, fun):
    # 参数设置
    PballRolling = 0.2 # 滚球蜣螂比例
    PbroodBall = 0.4 #产卵蜣螂比例
    PSmall = 0.2 # 小蜣螂比例
    Pthief = 0.2 # 偷窃蜣螂比例
    BallRollingNum = int(pop*PballRolling) #滚球蜣螂数量
    BroodBallNum = int(pop*PbroodBall) #产卵蜣螂数量
    SmallNum = int(pop*PSmall) #小蜣螂数量
    ThiefNum = int(pop*Pthief) #偷窃蜣螂数量
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    # 记录全局最优
    minIndex = np.argmin(fitness)  # 找到适应度最小值的索引
    GbestScore = copy.copy(fitness[minIndex])  # 复制最小适应度值作为全局最优分数
    GbestPositon = np.zeros([1, dim])  # 创建一个全0矩阵来存储全局最优位置
    GbestPositon[0, :] = copy.copy(X[minIndex, :])  # 复制最小适应度对应的位置为全局最优位置
    Curve = np.zeros([MaxIter, 1])  # 创建一个全0数组用于记录迭代过程中的最优分数
    Xl = copy.deepcopy(X)  # 用于记录上一代的种群位置

    # 记录当前代种群
    cX = copy.deepcopy(X)  # 复制当前种群位置
    cFit = copy.deepcopy(fitness)  # 复制当前种群适应度值
    for t in range(MaxIter):   # 迭代次数循环，从0到MaxIter-1
        print("第" + str(t) + "次迭代")
        # 蜣螂滚动 文献中式（1），（2）更新位置
        # 获取种群最差值
        maxIndex = np.argmax(fitness)  # 找到适应度最大值的索引
        Wort = copy.copy(X[maxIndex, :])  # 复制最大适应度对应的位置为Wort
        r2 = np.random.random()  # 生成一个随机数r2
        for i in range(0,BallRollingNum): 
            # 循环迭代变量 i 在从 0 到 BallRollingNum（不包括）的范围内
            if r2<0.9: # 如果 r2 的值小于 0.9
                if np.random.random()>0.5:  # 如果 np.random.random() 生成的随机数大于 0.5
                    alpha=1  # 设置alpha为1
                else:  # 如果 np.random.random() 生成的随机数不大于 0.5
                    alpha=-1 # 设置alpha为-1
                b = 0.3  # 设置b为0.3
                k = 0.1  # 设置k为0.1
                X[i,:]=cX[i,:]+b*np.abs(cX[i,:]-Wort)+alpha*k*Xl[i,:]
                # 对 X 数组中的第 i 行进行赋值计算，计算结果由 cX[i, :], b, np.abs(cX[i, :] - Wort), alpha 和 k * Xl[i, :] 组合得到
            else:  # 如果 r2 的值不小于 0.9
                theta = np.random.randint(180)# 生成一个0到179之间的随机整数
                if theta==0 or theta == 90 or theta == 180: # 如果theta的值为0、90或180度
                    X[i,:]=copy.copy(cX[i,:])# 将当前位置复制给新位置
                else:
                    theta = theta*np.pi/180 # 将theta转换为弧度制
                    X[i,:]=cX[i,:]+np.tan(theta)*np.abs(cX[i,:]-Xl[i,:])
                    # 对 X 数组中的第 i 行进行赋值计算，计算结果由 cX[i, :], np.tan(theta), np.abs(cX[i, :] - Xl[i, :]) 组合得到
            for j in range(dim): # 循环迭代变量 j 在从 0 到 dim 的范围内
                if X[i,j]>ub[j]:  # 如果 X 数组中的第 i 行、第 j 列的值大于上界 ub[j]
                    X[i,j]=ub[j]  # 将 X 数组中的第 i 行、第 j 列的值设置为上界 ub[j]
                if X[i,j]<lb[j]:  # 如果 X 数组中的第 i 行、第 j 列的值小于下界 lb[j]
                    X[i,j]=lb[j]  # 将 X 数组中的第 i 行、第 j 列的值设置为下界 lb[j]
            fitness[i]=fun(X[i,:]) # 计算第 i 行的适应度值，计算结果由 fun 函数根据 X[i, :] 得到
            if fitness[i]<GbestScore:  # 如果第 i 行的适应度值小于全局最优适应度值 GbestScore
                GbestScore=copy.copy(fitness[i]) # 将全局最优适应度值 GbestScore 更新为第 i 行的适应度值的副本
                GbestPositon[0,:]=copy.copy(X[i,:])  # 将全局最优位置 GbestPositon 的第一行设置为 X 数组中第 i 行的副本
        # 当前迭代最优
        minIndex=np.argmin(fitness)  # 使用 np.argmin 函数找到适应度值数组 fitness 中的最小值的索引
        GbestB = copy.copy(X[minIndex,:])  # 将 GbestB 设置为 X 数组中索引为 minIndex 的行的副本
        # 蜣螂产卵 ，文献中式（3）
        R=1-t/MaxIter # 根据当前迭代次数 t 和最大迭代次数 MaxIter 计算 R 的值
        X1=GbestB*(1-R)  # 根据全局最优解 GbestB 和 R 计算 X1 的值
        X2=GbestB*(1+R)  # 根据全局最优解 GbestB 和 R 计算 X2 的值
        Lb = np.zeros(dim)  # 创建长度为 dim 的零数组 Lb 和 Ub
        Ub = np.zeros(dim)
        for j in range(dim):  # 循环迭代变量 j 在从 0 到 dim 的范围内
            Lb[j]=max(X1[j],lb[j])  # Lb[j] 的值为 X1[j] 和 lb[j] 中的较大值
            Ub[j]=min(X2[j],ub[j])  # Ub[j] 的值为 X2[j] 和 ub[j] 中的较小值
        for i in range(BallRollingNum,BallRollingNum+BroodBallNum):
            # 循环迭代变量 i 在从 BallRollingNum 到 BallRollingNum + BroodBallNum 的范围内
            b1=np.random.random()  # 生成一个随机数 b1
            b2=np.random.random()  # 生成一个随机数 b2
            X[i,:]=GbestB+b1*(cX[i,:]-Lb)+b2*(cX[i,:]-Ub)# 根据公式更新 X 数组中的第 i 行的值
            for j in range(dim): # 循环迭代变量 j 在从 0 到 dim 的范围内
                if X[i,j]>ub[j]:  # 如果 X 数组中的第 i 行、第 j 列的值大于上界 ub[j]
                    X[i,j]=ub[j]  # 将 X 数组中的第 i 行、第 j 列的值设置为上界 ub[j]
                if X[i,j]<lb[j]:  # 如果 X 数组中的第 i 行、第 j 列的值小于下界 lb[j]
                    X[i,j]=lb[j]  # 将 X 数组中的第 i 行、第 j 列的值设置为下界 lb[j]
            fitness[i]=fun(X[i,:])  # 计算第 i 行的适应度值，计算结果由 fun 函数根据 X[i, :] 得到
            if fitness[i]<GbestScore:  # 如果第 i 行的适应度值小于全局最优适应度值 GbestScore
                GbestScore=copy.copy(fitness[i]) # 将全局最优适应度值 GbestScore 更新为第 i 行的适应度值的副本
                GbestPositon[0,:]=copy.copy(X[i,:]) # 将全局最优位置 GbestPositon 的第一行设置为 X 数组中第 i 行的副本
        # 小蜣螂更新
        #文献中(5),(6)
        R=1-t/MaxIter # 根据当前迭代次数 t 和最大迭代次数 MaxIter 计算 R 的值
        X1=GbestPositon[0,:]*(1-R) # 根据全局最优位置 GbestPositon 和 R 计算 X1 的值
        X2=GbestPositon[0,:]*(1+R) # 根据全局最优位置 GbestPositon 和 R 计算 X2 的值
        Lb = np.zeros(dim) # 创建长度为 dim 的零数组 Lb 和 Ub
        Ub = np.zeros(dim)
        for j in range(dim):  # 循环迭代变量 j 在从 0 到 dim 的范围内
            Lb[j]=max(X1[j],lb[j])  # Lb[j] 的值为 X1[j] 和 lb[j] 中的较大值
            Ub[j]=min(X2[j],ub[j])  # Ub[j] 的值为 X2[j] 和 ub[j] 中的较小值
        for i in range(BallRollingNum+BroodBallNum,BallRollingNum+BroodBallNum+SmallNum):  
            # 循环迭代变量 i 在从 BallRollingNum + BroodBallNum 到 BallRollingNum + BroodBallNum + SmallNum 的范围内
            C1 = np.random.random([1,dim]) # 生成一个随机数组 C1，形状为 [1, dim]
            C2 = np.random.random([1,dim]) # 生成一个随机数组 C2，形状为 [1, dim]
            X[i,:]=GbestPositon[0,:]+C1*(cX[i,:]-Lb)+C2*(cX[i,:]-Ub) # 根据公式更新 X 数组中的第 i 行的值
            for j in range(dim): # 循环迭代变量 j 在从 0 到 dim 的范围内
                if X[i,j]>ub[j]: # 如果 X 数组中的第 i 行、第 j 列的值大于上界 ub[j]
                    X[i,j]=ub[j] # 将 X 数组中的第 i 行、第 j 列的值设置为上界 ub[j]
                if X[i,j]<lb[j]: # 如果 X 数组中的第 i 行、第 j 列的值小于下界 lb[j]
                    X[i,j]=lb[j] # 将 X 数组中的第 i 行、第 j 列的值设置为下界 lb[j]
            fitness[i]=fun(X[i,:]) # 计算第 i 行的适应度值，计算结果由 fun 函数根据 X[i, :] 得到
            if fitness[i]<GbestScore: # 如果第 i 行的适应度值小于全局最优适应度值 GbestScore
                GbestScore=copy.copy(fitness[i]) # 将全局最优适应度值 GbestScore 更新为第 i 行的适应度值的副本
                GbestPositon[0,:]=copy.copy(X[i,:]) # 将全局最优位置 GbestPositon 的第一行设置为 X 数组中第 i 行的副本
        # 当前迭代最优
        minIndex=np.argmin(fitness) # 使用 np.argmin 函数找到适应度值数组 fitness 中的最小值的索引
        GbestB = copy.copy(X[minIndex,:]) # 将 GbestB 设置为 X 数组中索引为 minIndex 的行的副本
        # 偷窃蜣螂更新 
        # 文献中式（7）
        for i in range(pop-ThiefNum,pop): # 循环迭代变量 i 在从 pop - ThiefNum 到 pop 的范围内
            g=np.random.randn() # 生成一个符合标准正态分布的随机数 g
            S=0.5 # 设置 S 的值为 0.5
            X[i,:]=GbestPositon[0,:]+g*S*(np.abs(cX[i,:]-GbestB)+np.abs(cX[i,:]-GbestPositon[0,:])) # 根据公式更新 X 数组中的第 i 行的值
            for j in range(dim): # 循环迭代变量 j 在从 0 到 dim 的范围内
                if X[i,j]>ub[j]: # 如果 X 数组中的第 i 行、第 j 列的值大于上界 ub[j]
                    X[i,j]=ub[j] # 将 X 数组中的第 i 行、第 j 列的值设置为上界 ub[j]
                if X[i,j]<lb[j]: # 如果 X 数组中的第 i 行、第 j 列的值小于下界 lb[j]
                    X[i,j]=lb[j] # 将 X 数组中的第 i 行、第 j 列的值设置为下界 lb[j]
            fitness[i]=fun(X[i,:]) # 计算第 i 行的适应度值，计算结果由 fun 函数根据 X[i, :] 得到
            if fitness[i]<GbestScore: # 如果第 i 行的适应度值小于全局最优适应度值 GbestScore
                GbestScore=copy.copy(fitness[i]) # 将全局最优适应度值 GbestScore 更新为第 i 行的适应度值的副本
                GbestPositon[0,:]=copy.copy(X[i,:])  # 将全局最优位置 GbestPositon 的第一行设置为 X 数组中第 i 行的副本
        # 记录t代种群
        Xl= copy.deepcopy(cX) # 将 Xl 设置为 cX 的深拷贝，即创建一个与 cX 一样的副本
        #更新当前代种群
        for i in range(pop):  # 循环迭代变量 i 在从 0 到 pop 的范围内
            if fitness[i]<cFit[i]:  # 如果第 i 个个体的适应度值小于当前代个体的适应度值 cFit[i]
                cFit[i]=copy.copy(fitness[i]) # 将当前代个体的适应度值 cFit[i] 更新为第 i 个个体的适应度值的副本
                cX[i,:]=copy.copy(X[i,:])  # 将当前代个体 cX 的第 i 行设置为 X 数组中第 i 行的副本
        
        Curve[t] = GbestScore # 将 Curve 数组的第 t 个元素设置为全局最优适应度值 GbestScore

    return GbestScore, GbestPositon, Curve # 返回全局最优适应度值 GbestScore、全局最优位置 GbestPositon 和适应度曲线 Curve


# In[8]:


def ceemdan_decompose(series=None, trials=10, num_clusters=3): 
    decom = CEEMDAN()  # 创建CEEMDAN对象
    decom.trials = trials  # 设置分解的试验次数
    df_ceemdan = pd.DataFrame(decom(series.values).T)  # 对数据进行CEEMDAN分解并转换为数据框
    df_ceemdan.columns = ['imf'+str(i) for i in range(len(df_ceemdan.columns))]  # 为每一列设置列名为'imf' + 对应的索引号
    return df_ceemdan  # 返回分解后的数据框


# In[9]:


def sample_entropy(df_ceemdan=None, mm=1, r=0.1):
    np_sampen = []  # 存储样本熵的列表
    for i in range(len(df_ceemdan.columns)):
        sample_entropy = sampen2(list(df_ceemdan['imf'+str(i)].values), mm=mm, r=r, normalize=True)  # 计算样本熵
        np_sampen.append(sample_entropy[1][1])  # 将样本熵的值添加到列表中
    df_sampen = pd.DataFrame(np_sampen, index=['imf'+str(i) for i in range(len(df_ceemdan.columns))])  # 创建样本熵的数据框，设置行索引为'imf' + 对应的索引号
    return df_sampen  # 返回样本熵的数据框


# In[10]:


def kmeans_cluster(df_sampen=None, num_clusters=3): 
    np_integrate_form = KMeans(n_clusters=num_clusters, random_state=9).fit_predict(df_sampen)  # 使用K均值聚类进行聚类操作
    df_integrate_form = pd.DataFrame(np_integrate_form, index=['imf'+str(i) for i in range(len(df_sampen.index))], columns=['OT'])  # 创建聚类结果的数据框，设置行索引为'imf' + 对应的索引号，列名为'OT'
    return df_integrate_form  # 返回聚类结果的数据框


# In[11]:


def integrate_imfs(df_integrate_form=None, df_ceemdan=None): 
    df_tmp = pd.DataFrame()  # 创建一个空的数据框用于存储临时结果
    for i in range(df_integrate_form.values.max()+1):
        df_tmp['imf'+str(i)] = df_ceemdan[df_integrate_form[(df_integrate_form['OT']==i)].index].sum(axis=1)  # 对每个聚类簇内的IMF分量进行求和，得到综合的IMF分量
        
    df_integrate_result = df_tmp.T  # 对临时结果进行转置
    df_integrate_result['sampen'] = sample_entropy(df_tmp).values  # 计算综合的IMF分量的样本熵，并将其作为新的列添加到结果数据框中
    df_integrate_result.sort_values(by=['sampen'], ascending=False, inplace=True)  # 根据样本熵降序排列综合的IMF分量
    df_integrate_result.index = ['co-imf'+str(i) for i in range(df_integrate_form.values.max()+1)]  # 为综合的IMF分量设置新的行索引，命名规则为'co-imf' + 对应的索引号
    df_integrate_result = df_integrate_result.drop('sampen', axis=1, inplace=False)  # 移除样本熵这一列
    return df_integrate_result.T  # 返回结果数据框的转置


# In[12]:


def vmd_decompose(series=None, draw=True):
    def training(X):
        alpha = int(X[0])  # 将X的第一个元素转换为整数，表示alpha参数
        K = int(X[1])  # 将X的第二个元素转换为整数，表示K参数
        print(X)  # 输出X的值
        u, u_hat, omega = VMD(series, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)  # 使用VMD函数对series进行信号分解
        vmd = pd.DataFrame(u.T)  # 将分解后的结果转换为DataFrame对象
        vmd.columns = ['imf'+str(i) for i in range(K)]  # 设置DataFrame的列名为'imf0', 'imf1', ...

        np_sampen = []
        for i in range(len(vmd.columns)):
            SE = sampen2(list(vmd['imf'+str(i)].values), mm=1, r=0.1, normalize=True)  # 计算每个IMF的样本熵
            np_sampen.append(SE[1][1])
        df_sampen = pd.DataFrame(np_sampen, index=['imf'+str(i) for i in range(len(vmd.columns))])  # 将样本熵存储为DataFrame
        sampen = df_sampen[0].mean()  # 计算平均样本熵
        print(sampen)  # 输出平均样本熵
        return sampen
    
    ub = np.array([4000, 15])  # 设置参数的上界
    lb = np.array([100, 3])  # 设置参数的下界
    pop = 1  # 种群大小
    MaxIter = 1  # 最大迭代次数
    dim = 2  # 参数维度
    
    # 主函数
    GbestScore, GbestPositon, Curve = DBO(pop, dim, lb, ub, MaxIter, training)  # 使用DBO算法进行参数优化
    print('最优适应度值：', GbestScore)  # 输出最优适应度值
    print('最优解：', GbestPositon)  # 输出最优解
    
    # 绘制适应度曲线
    plt.figure(1)  # 创建图形窗口
    plt.semilogy(Curve, 'r-', linewidth=2)  # 绘制适应度曲线
    plt.xlabel('Iteration', fontsize='medium')  # 设置x轴标签
    plt.ylabel("Fitness", fontsize='medium')  # 设置y轴标签
    plt.grid()  # 添加网格线
    plt.title('DBO', fontsize='large')  # 设置标题
    plt.show()  # 显示图形
    
    imfs_vmd, imfs_hat, omega = VMD(series, alpha=int(GbestPositon[0]), tau=tau, K=int(GbestPositon[1]), DC=DC, init=init, tol=tol)  # 使用优化后的参数进行信号分解
    df_vmd = pd.DataFrame(imfs_vmd.T)  # 将分解后的结果转换为DataFrame对象
    df_vmd.columns = ['imf'+str(i) for i in range(int(GbestPositon[1]))]  # 设置DataFrame的列名为'imf0', 'imf1', ...
    
    return df_vmd  # 返回分解结果的DataFrame对象


# In[13]:


def create_train_test_set(data=None, timestep=timestep, co_imf_predict_for_fitting=None):
    if isinstance(data, pd.DataFrame):
        dataY = data['sum'].values.reshape(-1, 1)  # 提取DataFrame中的目标变量列，并将其转换为二维数组
        dataX = data.drop('sum', axis=1, inplace=False)  # 去除DataFrame中的目标变量列，得到特征变量列
    else:
        dataY = data.values.reshape(-1, 1)  # 将一维数组转换为二维数组作为目标变量
        dataX = dataY  # 特征变量和目标变量相同

    scalarX = MinMaxScaler(feature_range=(0, 1))  # 创建MinMaxScaler对象，用于特征变量归一化
    dataX = scalarX.fit_transform(dataX)  # 对特征变量进行归一化
    if co_imf_predict_for_fitting is not None:
        co_imf_predict_for_fitting = scalarX.transform(co_imf_predict_for_fitting)  # 对要预测的特征变量进行归一化

    scalarY = MinMaxScaler(feature_range=(0, 1))  # 创建MinMaxScaler对象，用于目标变量归一化
    dataY = scalarY.fit_transform(dataY)  # 对目标变量进行归一化

    trainX, trainY = [], []
    for i in range(len(dataY) - timestep):
        trainX.append(np.array(dataX[i:(i + timestep)]))  # 构建训练样本的特征序列
        trainY.append(np.array(dataY[i + timestep]))  # 构建训练样本的目标值
        if co_imf_predict_for_fitting is not None:
            if i < (len(dataY) - timestep - len(co_imf_predict_for_fitting)):
                trainX[i] = np.insert(trainX[i], timestep, dataX[i + timestep], 0)
            else:
                trainX[i] = np.insert(trainX[i], timestep, co_imf_predict_for_fitting[i - (len(dataY) - timestep - len(co_imf_predict_for_fitting))], 0)
                # 在训练样本的特征序列末尾插入要预测的特征变量值

    return np.array(trainX), np.array(trainY), scalarY  # 返回训练样本的特征序列、目标值和目标值的归一化器对象


# In[14]:


def evaluation_model(y_test, y_pred):
    y_test, y_pred = np.array(y_test).ravel(), np.array(y_pred).ravel()  # 将y_test和y_pred转换为一维数组
    r2 = r2_score(y_test, y_pred)  # 计算R^2分数
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # 计算均方根误差（RMSE）
    mae = mean_absolute_error(y_test, y_pred)  # 计算平均绝对误差（MAE）
    mape = mean_absolute_percentage_error(y_test, y_pred)  # 计算平均绝对百分比误差（MAPE）
    df_evaluation = pd.DataFrame({'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}, index=range(1))  # 创建评估结果的DataFrame
    return df_evaluation  # 返回评估结果的DataFrame对象


# In[15]:


def informer_predict(data=None, predict_duration=len(test), fitting=None):
    lr = 0.0001
    epochs = 4
    batch_size = 32
    seq_len = 96
    label_len = 48
    pred_len = 24
    rootpath = "/kaggle/working/dbo-inf/"
    trainrate = 0.7

    def training(X):
        lr=X[0]
        epochs=int(X[1])
        batch_size=int(X[2])
        print("lr:",lr,"  epochs:",epochs,"  batch_size:",batch_size)
        # writer = SummaryWriter(rootpath + "log/tensorboard/")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(0)
        print(data)
        df = pd.read_csv(rootpath + "data/ETT/ETTh1.csv")
        df['OT'] = data
        print(df)
        train = df.iloc[: int(trainrate * len(df)), :]
        test = df.iloc[int(trainrate * len(df)):, :]

        scaler = StandardScaler()
        scaler.fit(train.iloc[:, 1:].values)

        trainset = MyDataset(train, scaler, seq_len=96, label_len=48, pred_len=24)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = MyDataset(test, scaler, seq_len=96, label_len=48, pred_len=24)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        model = Informer().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)

        # train
        print("train...")
        model.train()
        for e in range(epochs):
            losses = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(trainloader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

                pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = pred[:, -pred_len:, :].to(device)
                true = batch_y[:, -pred_len:, :].to(device)

                loss = criterion(pred, true)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            print("Epochs:", e, " || train loss: %.4f" % np.mean(losses))

        torch.save(model, rootpath + "log/informer.pkl")

        # test
        print("test...")
        # model = torch.load("./Informer/log/informer.pkl").to(device)

        model.eval()
        losses = []
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(testloader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

            pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            preds.extend(pred.detach().cpu().numpy())
            trues.extend(batch_y.detach().cpu().numpy())

            pred = pred[:, -pred_len:, :].to(device)
            true = batch_y[:, -pred_len:, :].to(device)

            loss = criterion(pred, true)
            losses.append(loss.item())
        print("test loss: %.4f" % np.mean(losses))

        temp_mse = mean_squared_error(pred.cpu().detach().numpy().reshape(-1, 1), true.cpu().detach().numpy().reshape(-1, 1))  # 计算均方误差
        print("均方误差:", temp_mse)
        return temp_mse


    #优化参数
    # lr = 0.0001
    # epochs = 4
    # batch_size = 32

    def round_lr(lr):
        count=0
        while(lr*10<1):
            count+=1
            lr=lr*10
        if(lr*10>4):
            return (10**(-(count)))
        else:
            return (10**(-(count+1)))

    ub = np.array([0.001, 2, 2])  # 优化算法上界
    lb = np.array([0.0001, 1, 1])  # 优化算法下界
    pop = 5  # 种群大小
    MaxIter = 1  # 最大迭代次数
    dim = 3  # 优化变量维度
    GbestScore, GbestPositon, Curve = DBO(pop, dim, lb, ub, MaxIter, training)  # 使用Differential Evolution进行优化
    print('最优适应度值：', GbestScore)
    print('最优解：', GbestPositon)

    GbestPositon = GbestPositon[0]
    lr = GbestPositon[0]
    epochs = int(GbestPositon[1])
    batch_size = int(GbestPositon[2])
    lr = round_lr(lr)
    print("lr:",lr,"  epochs:",epochs,"  batch_size:",batch_size)
    seq_len = 96
    label_len = 48
    pred_len = 24
    rootpath = "/kaggle/working/dbo-inf/"
    trainrate = 0.7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    df = pd.read_csv(rootpath + "data/ETT/ETTh1.csv")
    df['OT'] = data
    print(df)
    train = df.iloc[: int(trainrate * len(df)), :]
    test = df.iloc[int(trainrate * len(df)):, :]

    scaler = StandardScaler()
    scaler.fit(train.iloc[:, 1:].values)

    trainset = MyDataset(train, scaler, seq_len=96, label_len=48, pred_len=24)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = MyDataset(test, scaler, seq_len=96, label_len=48, pred_len=24)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = Informer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)


    # train
    print("final train...")
    model.train()
    for e in range(epochs):
        train_losses = []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(trainloader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

            pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred = pred[:, -pred_len:, :].to(device)
            true = batch_y[:, -pred_len:, :].to(device)

            loss = criterion(pred, true)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print("Epochs:", e, " ||final  train loss: %.4f" % np.mean(train_losses))

    torch.save(model, rootpath + "log/informer.pkl")

    # test
    print("final test...")
    # model = torch.load("./Informer/log/informer.pkl").to(device)

    model.eval()
    test_losses = []
    trues, preds = [], []
    for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(testloader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

        pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        preds.extend(pred.detach().cpu().numpy())
        trues.extend(batch_y.detach().cpu().numpy())

        pred = pred[:, -pred_len:, :].to(device)
        true = batch_y[:, -pred_len:, :].to(device)

        loss = criterion(pred, true)
        test_losses.append(loss.item())
    print("final test loss: %.4f" % np.mean(test_losses))

    temp_mse = mean_squared_error(pred.cpu().detach().numpy().reshape(-1, 1), true.cpu().detach().numpy().reshape(-1, 1))  # 计算均方误差
    print("优化后的均方误差：",temp_mse)

    np.save(rootpath + "log/preds", np.array(preds))
    np.save(rootpath + "log/tures", np.array(trues))

    # show
    pred = np.load(rootpath + "log/preds.npy")
    true = np.load(rootpath + "log/tures.npy")

    print(pred.shape, true.shape)
    plt.plot(pred[0, -24:, -1], label="pred")
    plt.plot(true[0, -24:, -1], label="true")
    plt.legend()
    plt.savefig(rootpath + "img/show.png")
    plt.show()


    df_gru_evaluation = evaluation_model(true, pred)  # 评估模型性能
    y_test_predict = pred.ravel().reshape(-1, 1)
    scalarY = MinMaxScaler(feature_range=(0, 1))  # 创建MinMaxScaler对象，用于目标变量归一化
    scalarY.fit(df)
    y_test_predict_result = scalarY.inverse_transform(y_test_predict)  # 将预测结果反归一化
    y_test_raw = scalarY.inverse_transform(true)  # 将测试集目标值反归一化
    df_predict_raw = pd.DataFrame({'raw': y_test_raw.ravel(), 'predict': y_test_predict_result.ravel()},
                                  index=range(len(y_test_raw)))  # 创建预测结果的DataFrame
    df_train_loss = pd.DataFrame({'loss': train_losses, 'val_loss': test_losses},
                                 index=range(len(test_losses)))  # 创建训练损失的DataFrame

    return df_predict_raw, df_gru_evaluation, df_train_loss


# In[16]:


# CEEMDAN 分解
df_ceemdan = ceemdan_decompose(series_close)  # 对 series_close 应用 CEEMDAN 分解，得到分解后的数据框 df_ceemdan
df_ceemdan.plot(title='CEEMDAN 分解', subplots=True, figsize=(8, 8))  # 绘制 CEEMDAN 分解结果的子图，设置标题和图像大小


# In[17]:


# 样本熵的计算
df_sampen = sample_entropy(df_ceemdan)  # 计算 CEEMDAN 分解结果的样本熵，保存在数据框 df_sampen 中
df_sampen.plot(title='SE')  # 绘制样本熵的图像，设置标题为 'SE'


# In[18]:


# 通过样本熵的K-Means聚类
df_integrate_form = kmeans_cluster(df_sampen)  # 对样本熵数据框 df_sampen 进行 K-Means 聚类，得到聚类结果保存在数据框 df_integrate_form 中
print(df_integrate_form)  # 打印聚类结果 df_integrate_form


# In[19]:


# 将IMF和残留物整合为3个共同IMF
df_integrate_result = integrate_imfs(df_integrate_form, df_ceemdan)  # 将聚类结果 df_integrate_form 和 CEEMDAN 分解结果 df_ceemdan 整合为3个共同IMF，保存在数据框 df_integrate_result 中
df_integrate_result.plot(title='Co-IMFs', subplots=True)  # 绘制共同IMFs的子图，设置标题为 'Co-IMFs'


# In[20]:


# # 通过VMD分解高频的Co-IMF0
# df_vmd_co_imf0 = vmd_decompose(df_integrate_result['co-imf0'])  # 使用 VMD 对高频的 Co-IMF0 进行分解，得到分解结果保存在数据框 df_vmd_co_imf0 中


# # In[ ]:


# df_vmd_co_imf0.plot(title='VMD 分解', subplots=True, figsize=(10, 8))  # 绘制 VMD 分解结果 df_vmd_co_imf0 的子图，设置标题为 'VMD 分解'，图形大小为 (10, 8)


# # In[21]:


# df_vmd_co_imf0['sum'] = df_integrate_result['co-imf0']  # 将 df_integrate_result['co-imf0'] 列赋值给 df_vmd_co_imf0 的 'sum' 列

co_imf0_predict_raw, co_imf0_gru_evaluation, co_imf0_train_loss = informer_predict(df_integrate_result['co-imf0'])  # 使用 informer 进行预测并得到预测结果、评估结果和训练损失

print('======Co-IMF0 最终预测======\n', co_imf0_gru_evaluation)  # 打印 Co-IMF0 的最终预测评估结果

co_imf0_predict_raw.plot(title='Co-IMF0 预测结果')  # 绘制 Co-IMF0 的预测结果图，设置标题为 'Co-IMF0 预测结果'

co_imf0_train_loss.plot(title='Co-IMF0 训练损失')  # 绘制 Co-IMF0 的训练损失图，设置标题为 'Co-IMF0 训练损失'


# In[ ]:


co_imf1_predict_raw, co_imf1_gru_evaluation, co_imf1_train_loss = informer_predict(df_integrate_result['co-imf1'])  # 使用 LSTM 进行预测并得到预测结果、评估结果和训练损失

print('======Co-IMF1 最终预测======\n', co_imf1_gru_evaluation)  # 打印 Co-IMF1 的最终预测评估结果

co_imf1_predict_raw.plot(title='Co-IMF1 预测结果')  # 绘制 Co-IMF1 的预测结果图，设置标题为 'Co-IMF1 预测结果'

co_imf1_train_loss.plot(title='Co-IMF1 训练损失')  # 绘制 Co-IMF1 的训练损失图，设置标题为 'Co-IMF1 训练损失'


# In[ ]:


co_imf2_predict_raw, co_imf2_gru_evaluation, co_imf2_train_loss = informer_predict(df_integrate_result['co-imf2'])  # 使用 LSTM 进行预测并得到预测结果、评估结果和训练损失

print('======Co-IMF2 最终预测======\n', co_imf2_gru_evaluation)  # 打印 Co-IMF2 的最终预测评估结果

co_imf2_predict_raw.plot(title='Co-IMF2 预测结果')  # 绘制 Co-IMF2 的预测结果图，设置标题为 'Co-IMF2 预测结果'

co_imf2_train_loss.plot(title='Co-IMF2 训练损失')  # 绘制 Co-IMF2 的训练损失图，设置标题为 'Co-IMF2 训练损失'


# In[ ]:


result = co_imf0_predict_raw['predict'] + co_imf1_predict_raw['predict'] + co_imf2_predict_raw['predict']  # 将 Co-IMF0、Co-IMF1 和 Co-IMF2 的预测结果相加得到最终预测结果

df_add_evaluation = evaluation_model(test, result)  # 对最终预测结果和真实值进行评估，得到评估结果

print('======最终预测======\n', df_add_evaluation)  # 打印最终预测的评估结果


# In[ ]:


# 创建一个图形窗口
plt.figure(figsize=(12, 3))

# 设置图形标题和字体大小
plt.title('CEEMDAN-DBO-VMD-DBO-LSTM', size=15)

# 绘制真实值曲线
plt.plot(test, color='r', linewidth=2.5, linestyle="-", label='Actual')

# 绘制预测值曲线
plt.plot(result, color='yellow', linewidth=2, linestyle="--", label='Prediction')

# 显示图例
plt.legend()

# 设置y轴标签和字体大小
plt.ylabel('O3', size=15)

# 设置x轴标签和字体大小
plt.xlabel('time/day', size=15)

# 显示图形
plt.show()


# In[ ]:




