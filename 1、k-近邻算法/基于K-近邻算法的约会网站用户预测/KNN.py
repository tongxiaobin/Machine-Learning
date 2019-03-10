import numpy as np
#文本文件转换为计算机认识的数据格式
def file2matrix(fileName):
    f = open(fileName)
    arrayLines = f.readlines()#读出所有行
    lenLines = len(arrayLines)#有多少行
    dataSet = np.zeros((lenLines,3))
    labels = []
    index = 0
    for line in arrayLines:
        line = line.strip()#去掉首尾字符，一般去掉首尾的空格换行等字符
        line = line.split('\t')#字符分割，这里以水平制表符‘\t’分割
        dataSet[index,:] = line[0:3]#字符切片
        labels.append(int(line[-1]))#将切片后的最后一片放到标签里
        index += 1
    return dataSet,labels

#数据归一化
def normlize(dataSet):
    maxVal = dataSet.max(axis=0)#按行最大值
    minVal = dataSet.min(axis=0)#按行最小值
    ranges = maxVal - minVal
    row = dataSet.shape[0]#数据行数，即有多少个样本数据
    norm_dataSet = (dataSet - np.tile(minVal,(row,1)))/np.tile((ranges),(row,1))#归一化
    return norm_dataSet,ranges,minVal

#K-nearst-neighbor
def classifier(inX,dataSet,labels,K):
    row = dataSet.shape[0]
    subtract = np.tile(inX,(row,1)) - dataSet
    square = subtract**2
    sqr_sum = (np.sum(square,axis=1))**0.5
    sort_index = sqr_sum.argsort()
    classCount = {}
    for i in range(K):
        key = labels[sort_index[i]]
        classCount[key] = classCount.get(key,0)+1
        result = sorted(classCount.items(),key = lambda x:x[1],reverse=True)
        return result[0][0]