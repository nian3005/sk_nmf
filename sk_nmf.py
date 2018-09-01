# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 23:09:37 2018

@author: Default
"""
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
import os
import math


# In[]
def semi_train(V, H0,components, iternum, e,R):
    '''
    非负矩阵分解函数
    :param V:  原始矩阵
    :param h:
    :param components:  要提取多少个特征
    :param iternum: 迭代次数
    :param e: 误差阈值
    m像素数目，n通道数目
    :return:
    '''
    m,n = V.shape
    # 随机初始化两个矩阵
    W1 = np.random.random((m, components))
    H1 = np.random.random((components, n))
    W0 = np.random.random((m, components))
    
    
    # 迭代计算过程，循环中使用了numpy的切片操作，可以避免直接使用Python的多重循环，从而提高了运行速度
    for iter in range(iternum):
        
        V_pre = np.dot(W1, H1)+np.dot(W0, H0)
        E = V - V_pre
        
        err = np.sum(E * E)
        print(iter," error: ",err)
        if err < e:
            break
        # 对照更新公式
            
        
        a1 = np.dot(V, H0.T)
        b1 = np.dot(W0, np.dot(H0, H0.T))+np.dot(W1, np.dot(H1, H0.T))
        W0 = W0 * (a1 / b1)
        
        a2 = np.dot(V, H1.T)
        b2 = np.dot(W0, np.dot(H0, H1.T))+np.dot(W1, np.dot(H1, H1.T))
        W1 = W1 * (a2 / b2)  
        
        a3 = np.dot(W1.T, V)
        b3 = np.dot(W1.T, np.dot(W0, H0))+np.dot(W1.T, np.dot(W1, H1))+R
        H1 = H1 * (a3 / b3)        
        
        
        
        a4 = np.dot(V, H0.T)
        b4 = np.dot(W0, np.dot(H0, H0.T))+np.dot(W1, np.dot(H1, H0.T))
        W0 = W0 * (a1 / b1)
    return W1, H1, W0

# In[]
def get_avr_spec(img,T):
    percent=0.1
    pica_120 = img[:,:,120].reshape(-1)
    num= int(percent*(len(pica_120)))
    if T==1:
        index = pica_120.argsort()[:num]
    elif T==0:
        index = pica_120.argsort()[-num:][::-1]

    pica=[]
    avr=[]
    for i in range(160):
        pica.append(img[:,:,i].reshape(-1))
        count=0
        for j in range(1633):
            if j in index:
                count+=pica[i][j]
        avr.append(count/len(index))
    return avr
def get_ab(transmission):
    return (math.log10(1/transmission))

# In[]
def unmixing(img):
    percent=0.01
    pica_120 = img[:,:,120].reshape(-1)
    num= int(percent*(len(pica_120)))
    
    index = pica_120.argsort()[:num]
    
    V=[]
    
    for i in range(1633):
        if i in index:
            Abvs=list(map(get_ab,img[i//71,i%71]))
            V.append(Abvs)
    V=np.array(V)
    print(V.shape)
    
 
    H_0=np.array([list(map(get_ab,get_avr_spec(img,0)))])
    #print(H_0.shape)
    
    W_1, H_1, W_0 = semi_train(V, H_0, 1, 100000, 1e-4, 0)
    print('OK') 
    return W_1, H_1, W_0 

# In[]
def no_nan(img):
    img_no_NaN =np.zeros((160,23,71))
    for i in range(0,160):
        img_no_NaN[i]=img[:,:,i].reshape(23,71)
        where_are_nan = np.isnan(img_no_NaN[i])  
        where_are_inf = np.isinf(img_no_NaN[i])  
        img_no_NaN[i][where_are_nan] = 0.3  
        img_no_NaN[i][where_are_inf] = 0.3
    img_no_NaN = img_no_NaN.transpose(1,2,0)

    return img_no_NaN

# In[]
def run_unmixing(hdrfile,datfile):
    img = envi.open(hdrfile,datfile)
    img=no_nan(img)
    Ws0 = get_avr_spec(img,0)    
    Abs0 = list(map(get_ab,Ws0))    
    W_1, H_1, W_0=unmixing(img)
    
    w3=W_1
    h3=H_1
    v3=np.dot(w3,h3)
    result=[]
    for i in range(160):
        avr = np.average(v3[:,i])
        result.append(avr)
    return np.array(result).reshape(-1)

# In[]
#result0=run_unmixing('C:/sub/0.hdr','C:/sub/0.dat')
#result1=run_unmixing('C:/sub/1.hdr','C:/sub/1.dat')
#result2=run_unmixing('C:/sub/2.hdr','C:/sub/2.dat')
#result3=run_unmixing('C:/sub/3.hdr','C:/sub/3.dat')
result4=run_unmixing('C:/sub/4.hdr','C:/sub/4.dat')

# In[]

# In[]
wavelength = [
 859.500000, 860.349976, 861.200012, 862.049988, 862.890015, 863.739990,
 864.590027, 865.440002, 866.289978, 867.140015, 867.989990, 868.840027,
 869.690002, 870.530029, 871.380005, 872.229980, 873.080017, 873.929993,
 874.770020, 875.619995, 876.469971, 877.320007, 878.159973, 879.010010,
 879.859985, 880.710022, 881.549988, 882.400024, 883.250000, 884.090027,
 884.940002, 885.780029, 886.630005, 887.479980, 888.320007, 889.169983,
 890.010010, 890.859985, 891.700012, 892.549988, 893.390015, 894.239990,
 895.080017, 895.929993, 896.770020, 897.609985, 898.460022, 899.299988,
 900.150024, 900.989990, 901.830017, 902.679993, 903.520020, 904.359985,
 905.210022, 906.049988, 906.890015, 907.729980, 908.580017, 909.419983,
 910.260010, 911.099976, 911.940002, 912.780029, 913.630005, 914.469971,
 915.309998, 916.150024, 916.989990, 917.830017, 918.669983, 919.510010,
 920.349976, 921.190002, 922.030029, 922.869995, 923.710022, 924.549988,
 925.390015, 926.229980, 927.059998, 927.900024, 928.739990, 929.580017,
 930.419983, 931.260010, 932.090027, 932.929993, 933.770020, 934.609985,
 935.440002, 936.280029, 937.119995, 937.950012, 938.789978, 939.630005,
 940.460022, 941.299988, 942.130005, 942.969971, 943.799988, 944.640015,
 945.469971, 946.309998, 947.140015, 947.979980, 948.809998, 949.650024,
 950.479980, 951.309998, 952.150024, 952.979980, 953.809998, 954.650024,
 955.479980, 956.309998, 957.140015, 957.979980, 958.809998, 959.640015,
 960.469971, 961.299988, 962.130005, 962.969971, 963.799988, 964.630005,
 965.460022, 966.289978, 967.119995, 967.950012, 968.780029, 969.609985,
 970.440002, 971.270020, 972.090027, 972.919983, 973.750000, 974.580017,
 975.409973, 976.239990, 977.059998, 977.890015, 978.719971, 979.549988,
 980.369995, 981.200012, 982.030029, 982.849976, 983.679993, 984.500000,
 985.330017, 986.150024, 986.979980, 987.799988, 988.630005, 989.450012,
 990.280029, 991.099976, 991.929993, 992.750000]
plt.plot(wavelength,result0,'r',label='0')
plt.plot(wavelength,result1,'b',label='1')
plt.plot(wavelength,result2,'c',label='2')
plt.plot(wavelength,result3,'g',label='3')
plt.plot(wavelength,result4,label='4')

plt.legend(loc='upper right')
plt.xlabel('Wavelength[nm]')  # 横轴标题
plt.ylabel('Absorbance')  # 横轴标题
fig = plt.gcf()
fig.set_size_inches(15, 10)

plt.show()

# In[]
def train(V, components, iternum, e):
    '''
    非负矩阵分解函数
    :param V:  原始矩阵
    :param components:  要提取多少个特征
    :param iternum: 迭代次数
    :param e: 误差阈值
    m像素数目，n通道数目
    :return:
    '''
    m,n = V.shape
    # 随机初始化两个矩阵
    W = np.random.random((m, components))
    H = np.random.random((components, n))
    # 迭代计算过程，循环中使用了numpy的切片操作，可以避免直接使用Python的多重循环，从而提高了运行速度
    for iter in range(iternum):
        V_pre = np.dot(W, H)
        E = V - V_pre

        err = np.sum(E * E)
        print(err)
        if err < e:
            break
        # 对照更新公式
        a = np.dot(W.T, V)
        b = np.dot(W.T, np.dot(W, H))
        H[b != 0] = (H * a / b)[b != 0]

        c = np.dot(V, H.T)
        d = np.dot(W, np.dot(H, H.T))

        W[d != 0] = (W * c / d)[d != 0]
    return W, H