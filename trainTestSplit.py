import numpy as np
from scipy.io import loadmat
import os
import random
from scipy import io
import h5py

# get train test split
keys = {'PaviaU':'paviaU_gt',#9
        'Salinas':'salinas_gt',#17个类别
        'KSC':'KSC_gt',#100
        'Houston':'Houston2018_gt',
        'gf5': 'gf5_gt',
        'Indian_pines':'indian_pines_gt',#10
        'WHU_Hi_LongKou':'WHU_Hi_LongKou_gt',
        'Xiongan': 'xiongan_gt'}#传的是地面真值
TRAIN_SIZE = [20]#
RUN = 1

def sample_gt(gt, train_size, mode='fixed_withone'):#gt:地面真值，（）里面的是参数
    indices = np.nonzero(gt)#gt是一个二维数组(610,340),indices是二维的元组，可分为indices[0] 或 indices_rows表行索引，indices[1] 或 indices_cols表列索引
    # print(gt.shape)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes,将索引展平成一维数组 y=(42776,)
    # print(y.shape)
    train_gt = np.zeros_like(gt)#创建一个与gt图像相同形状的全零数组，用于存储训练集的地面真值。
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)
        if mode == 'random':
            train_size = float(train_size) / 100  # dengbin:20181011

    if mode == 'fixed_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices =np.nonzero(gt == c)#c是光谱维度，c=204
            X = list(zip(*indices))  # x,y features
            train_indices += random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

# 总结：random_withone是按比例随机抽样，保证每类至少抽到一个样本；而fixed_withone则是固定数量的随机抽样，同样确保每个类别都有样本被纳入训练集。

# 保存样本
def save_sample(train_gt, test_gt, dataset_name, sample_size, run):
    sample_dir = './trainTestSplit/' + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt':train_gt, 'test_gt':test_gt})



def load(dname):
    # path = os.path.join(dname,'{}_gt.mat'.format(dname))
    path = os.path.join("data",dname,'{}_gt.mat'.format(dname))
    dataset = loadmat(path)
    key = keys[dname]
    gt = dataset[key]
    # # 采样背景像素点
    # gt += 1
    return gt

def TrainTestSplit(datasetName):
    gt = load(datasetName)
    for size in TRAIN_SIZE:
        for r in range(RUN):
            train_gt, test_gt = sample_gt(gt, size)
            save_sample(train_gt, test_gt, datasetName, size, r)
    print('Finish split {}'.format(datasetName))


if __name__ == '__main__':
    dataseteName = ['PaviaU']#Salinas/PaviaU
    for name in dataseteName:
        TrainTestSplit(name)
    print('*'*8 + 'FINISH' + '*'*8)

