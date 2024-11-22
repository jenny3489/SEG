from scipy import io
import argparse
import configparser
import torch
from torch import nn
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import scale, minmax_scale
import os
from PIL import Image
from utils import get_graph_list, split, get_edge_index
import math
from Model.module import SubGcnFeature, GraphNet
from Trainer import JointTrainer
from Monitor import GradMonitor
from visdom import Visdom
from tqdm import tqdm
from scipy.io import loadmat, savemat
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='Salinas', help='DATASET NAME')
    parser.add_argument('--block', type=int, default=100, help='BLOCK SIZE')
    parser.add_argument('--epoch', type=int, default=100, help='ITERATION')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--comp', type=int, default=10, help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=64, help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=1, help='EXPERIMENT AMOUNT')
    parser.add_argument('--spc', type=int, default=20, help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=128, help='HIDDEN SIZE')
    parser.add_argument('--lr', type=float, default=5e-4, help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0., help='WEIGHT DECAY')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    viz = Visdom(port=8097)



    # Reading hyperspectral image
    data_path = 'data/{0}/{0}.mat'.format(arg.name)
    m = loadmat(data_path)
    data = m[config.get(arg.name, 'data_key')]
    gt_path = 'data/{0}/{0}_gt.mat'.format(arg.name)
    m = loadmat(gt_path)
    gt = m[config.get(arg.name, 'gt_key')]

    # Normalizing data
    h, w, c = data.shape
    data = data.reshape((h * w, c))#把数据展平，利于归一化处理
    data = data.astype(float)#浮点型能提供更高的精度
    if arg.name == 'Xiongan':
        minmax_scale(data, copy=False)
    data_normalization = scale(data).reshape((h, w, c))#归一化后再把数据变成三维的

    # Superpixel segmentation
    seg_root = 'data/rgb'
    seg_path = os.path.join(seg_root, '{}_seg_{}.npy'.format(arg.name, arg.block))
    if os.path.exists(seg_path):#若分割路径存在，则直接加载分割好的放入到seg中
        seg = np.load(seg_path)
    else:#若不存在就利用下面路径，重新分好放进去seg中
        rgb_path = os.path.join(seg_root, '{}_rgb.jpg'.format(arg.name))#这个路径应该是自定义的
        img = Image.open(rgb_path)
        img_array = np.array(img)
        # The number of superpixel
        n_superpixel = int(math.ceil((h * w) / arg.block))
        seg = slic(img_array, n_superpixel, arg.comp)#超像素分割
        # Saving
        np.save(seg_path, seg)

    # Constructing graphs
    graph_path = 'data/{}/{}_graph.pkl'.format(arg.name, arg.block)
    if os.path.exists(graph_path):
        graph_list = torch.load(graph_path)
    else:
        graph_list = get_graph_list(data_normalization, seg)#若路径不存在，会调用get_graph_list（在utils中定义）函数来生成图数据列表。这个函数接收两个参数
        torch.save(graph_list, graph_path)
    subGraph = Batch.from_data_list(graph_list)

    # Constructing full graphs
    full_edge_index_path = 'data/{}/{}_edge_index.npy'.format(arg.name, arg.block)
    if os.path.exists(full_edge_index_path):
        edge_index = np.load(full_edge_index_path)
    else:
        edge_index, _ = get_edge_index(seg)#会调用get_edge_index（在utils中定义）函数
        np.save(full_edge_index_path, edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())
    fullGraph = Data(None,edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                     seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)

#训练
    for r in range(arg.run):
        print('——'*15 + 'Run {}'.format(r) + '——'*15)
        # Reading the training data set and testing data set
        m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(arg.name, arg.spc, r))
        tr_gt, te_gt = m['train_gt'], m['test_gt']#从加载的数据中分离出训练和测试的标签，转换为PyTorch的张量，并保存到fullGraph对象中。
        tr_gt_torch, te_gt_torch = torch.from_numpy(tr_gt).long(), torch.from_numpy(te_gt).long()
        fullGraph.tr_gt, fullGraph.te_gt = tr_gt_torch, te_gt_torch

#实例化两个图卷积网络模型
        gcn1 = SubGcnFeature(config.getint(arg.name, 'band'), arg.hsz)
        gcn2 = GraphNet(arg.hsz, arg.hsz, config.getint(arg.name, 'nc'))
#定义优化器、损失函数、训练器和梯度监控器
        optimizer = torch.optim.Adam([{'params': gcn1.parameters()}, {'params': gcn2.parameters()}], lr=arg.lr,weight_decay=arg.wd)
        criterion = nn.CrossEntropyLoss()#交叉熵损失函数来衡量分类预测与真实标签之间的差距
        trainer = JointTrainer([gcn1, gcn2])#管理模型的训练过程
        monitor = GradMonitor()#监控训练过程中的梯度信息

        device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
        max_acc = 0
        # save_root = 'models/{}/{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, arg.block)
        save_root = 'models/{}/{}/{}_overall_skip_2_SGAT_l1_clip'.format(arg.name, arg.spc, arg.block)
        pbar = tqdm(range(arg.epoch))#设置进度条
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # 训练循环
        for epoch in pbar:
            pbar.set_description_str('Epoch: {}'.format(epoch))
            tr_loss = trainer.train(subGraph, fullGraph, optimizer, criterion, device, monitor.clear(), is_l1=True,is_clip=True)
            te_loss, acc, AA, kappa = trainer.evaluate(subGraph, fullGraph, criterion, device)#在每个epoch中，先训练模型，然后评估模型在测试集上的性能。
            pbar.set_postfix_str('train loss: {} test loss:{} acc:{} AA:{} kappa:{}'.format(tr_loss, te_loss, acc, AA, kappa))

            # 可视化训练和测试损失
            viz.line([[tr_loss, te_loss]], [epoch], win='{}_train_test_loss{}'.format(arg.name, r),
                     opts=dict(title='{}_Training and Testing Loss_{}'.format(arg.name, r), legend=['train_loss', 'test_loss']),
                     update='append')

            # 可视化准确率、平均准确率和Kappa
            viz.line([[acc, AA, kappa]], [epoch], win='{}_metrics_{}'.format(arg.name, r),
                     opts=dict(title='{}_Accuracy, AA, and Kappa_{}'.format(arg.name, r), legend=['accuracy', 'AA', 'kappa']),
                     update='append')

            # viz.line([monitor.get()], [epoch], win='{}_grad_{}'.format(arg.name, r), update='append')

            if acc > max_acc:
                max_acc = acc
                no_improve_epochs = 0
                # if not os.path.exists(save_root):
                #     os.makedirs(save_root)
                trainer.save([os.path.join(save_root, 'intNet_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'extNet_best_{}_{}.pkl'.format(arg.spc, r))])
    print('——' * 15 + ' Training Finish' + '——' * 15)

#预测
    for r in range(arg.run):

        # predicting
        preds = trainer.predict(subGraph, fullGraph, device)
        seg_torch = torch.from_numpy(seg)
        map = preds[seg_torch]
        save_root = 'prediction/{}/{}/{}_overall_skip_2_SGAT_l1_clip'.format(arg.name, arg.spc, arg.block)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, '{}.mat'.format(r))
        savemat(save_path, {'pred': map.cpu().numpy()})
    print('——'*15 + 'Predict Finishing' + '——'*15)


# 评估结果
prediction_path = 'prediction/{}/{}/{}_overall_skip_2_SGAT_l1_clip/0.mat'.format(arg.name, arg.spc, arg.block)
prediction_data = loadmat(prediction_path)
pred = prediction_data['pred']
# 加载GT数据
gt_path = 'data/{0}/{0}_gt.mat'.format(arg.name)
gt_data = loadmat(gt_path)
gt = gt_data[config.get(arg.name, 'gt_key')]
# 对预测结果进行偏移调整
pred += 1
# 将背景区域的标签设置为0
pred[gt == 0] = 0
# 保存处理后的预测结果
save_path = os.path.join('prediction/{}/{}/{}_overall_skip_2_SGAT_l1_clip'.format(arg.name, arg.spc, arg.block),'0.mat')
savemat(save_path, {'pred': pred})
# 计算评价指标
def calculate_metrics(gt, pred):
    # 过滤掉背景区域
    mask = gt > 0
    gt_filtered = gt[mask]
    pred_filtered = pred[mask]
    # 计算总体精度（OA）
    oa = accuracy_score(gt_filtered, pred_filtered)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(gt_filtered, pred_filtered)
    # 计算每个类别的精度
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    # 计算平均精度（AA）
    aa = np.mean(class_acc)
    # 计算Kappa系数.
    kappa = cohen_kappa_score(gt_filtered, pred_filtered)
    return oa, aa, kappa, class_acc
# 计算评价指标
oa, aa, kappa, class_acc = calculate_metrics(gt, pred)
print(f"Overall Accuracy (OA): {oa}")
print(f"Average Accuracy (AA): {aa}")
print(f"Kappa Coefficient: {kappa}")
print(f"Class Accuracy: {class_acc}")


#可视化预测结果t
plt.figure(figsize=(20, 20))
# 将背景（标签为0的部分）设置为黑色
map_visualization = map.cpu().numpy()
map_visualization[gt == 0] = 0  # 将背景部分设置为 -1，以便在 colormap 中显示为黑色
# 使用自定义的 colormap
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['white', 'red', 'green', 'blue', 'purple', 'cyan', 'yellow', 'magenta', 'orange', 'brown', 'pink', 'grey', 'lime', 'indigo', 'violet', 'teal', 'maroon', 'olive', 'navy', 'gold'])
plt.imshow(map_visualization, cmap=cmap, interpolation='nearest')
plt.axis('off')
plt.title('Prediction Visualization - Run {}'.format(r))
plt.show()


