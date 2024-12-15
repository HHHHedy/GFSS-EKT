import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from networks.pspnet_ekt import GFSS_Model
from engine import Engine
import dataset
import argparse
import torch.nn.functional as F
import umap
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Few-shot Segmentation Framework")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="choose the number of workers.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    return parser

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    avgpool = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    # maxpooling
    maxpool = F.max_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:])
    return avgpool, maxpool

def extract_features(model, dataloader):
    # 提取特征向量
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, targets, label_id = data
            # mask = (targets == pad_label_list).float().unsqueeze(1)
            # print("mask:", mask.shape)
            # pad_label_list = torch.masked_select(pad_label_list, pad_label_list != 0)
            # print('pad_label_list:', pad_label_list)
            '''repeat_num = len(pad_label_list)
            print('repeat_num:', repeat_num)
            assert repeat_num > 0'''
            _, outputs = model(inputs)
            # outputs = outputs.repeat(repeat_num, 1, 1, 1)
            mask = (targets == label_id).float().unsqueeze(1)
            mask = F.interpolate(mask, size=outputs.shape[-2:], mode='bilinear', align_corners=True)
            masked_features = outputs * mask
            sum_masked_features = torch.sum(masked_features, dim=(-2, -1))
            valid_counts = torch.sum(mask, dim=(-2, -1))
            valid_counts = valid_counts + 1e-6
            features_pool = sum_masked_features / valid_counts
            features_pool = features_pool.reshape(features_pool.shape[0], -1)  # [n, d]
            features.append(features_pool)
            labels.append(torch.tensor([label_id], device=outputs.device))

            # for i in label_list:
            #     i = int(i)
            #     mask = (targets == i).float().unsqueeze(1)
            #     mask = F.interpolate(mask, size=outputs.shape[-2:], mode='bilinear', align_corners=True)
            #     # mask_list.append(mask)
            #     # feature = outputs[i-1, :, :, :].unsqueeze(0)
            #     # Apply the mask to the feature map
            #     masked_features = outputs * mask
    
            #     # Compute the sum of masked features
            #     sum_masked_features = torch.sum(masked_features, dim=(-2, -1))  # Sum over H and W
    
            #     # Compute the count of non-zero elements in the mask
            #     valid_counts = torch.sum(mask, dim=(-2, -1))  # Count over H and W
    
            #     # Avoid division by zero by adding a small value to valid_counts
            #     valid_counts = valid_counts + 1e-6
    
            #     # Compute the average
            #     features_pool = sum_masked_features / valid_counts
            #     # feature_list.append(feature)
            #     # features_pool = F.avg_pool2d(input=feature, kernel_size=feature.shape[-2:])
            #     # print('features_pool:', features_pool.shape)
            #     features_pool = features_pool.reshape(features_pool.shape[0], -1)  # [n, d]
            #     features.append(features_pool)
            #     labels.append(torch.tensor([i], device=outputs.device))
            # mask = torch.cat(mask_list, dim=0)
            # feature = torch.cat(feature_list, dim=0)
            # mask = F.interpolate(mask, size=outputs.shape[-2:], mode='bilinear', align_corners=True)
            # features_pool, _ = Weighted_GAP(feature, mask)
            # features_pool = F.avg_pool2d(input=feature, kernel_size=feature.shape[-2:])
            # print('features_pool:', features_pool.shape)
            # features_pool = features_pool.reshape(features_pool.shape[0], -1)  # [n, d]
            # features.append(features_pool)
            #b, c, h, w = outputs.shape
            #features.append(outputs.permute(1,0,2,3).reshape(c,-1))
            # pad_label_list = torch.tensor(pad_label_list)
            # label = np.arange(1, 21)  # [1, 2, 3, ..., 20]
            # label = torch.tensor(label)
            # labels.extend(label_list)
            # print('labels:', labels)
    return torch.cat(features), torch.cat(labels)

def visualize_tsne(features, labels, save_path='tsne_ours_shot10_bfdecom_test.png'):
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(features.numpy())

    colors = [
    'gray', 'orange', 'green', 'purple', 'cyan', 'brown', 'pink', 'olive', 'darkblue', 'teal',
    'gold', 'lime', 'darkred', 'navy', 'darkgreen', 'magenta', 'tan', 'coral', 'turquoise', 'khaki'
    ]
    base_color = 'blue'
    novel_color = 'red'
    plt.figure(figsize=(12, 8))

    # other labels (1-20)
    for i in range(1, 21):
        label_indices = np.where(labels == i)[0]
        plt.scatter(embedded[label_indices, 0], embedded[label_indices, 1], 
                c=colors[i-1], label=f'Class {i}', alpha=0.7, zorder=5)
    
    base_indices = np.where(labels == 21)[0]
    plt.scatter(embedded[base_indices, 0], embedded[base_indices, 1], 
            c=base_color, marker='p', label='BP', alpha=0.7, s=100, zorder=10)

    # novel_embs (label = 22)
    novel_indices = np.where(labels == 22)[0]
    plt.scatter(embedded[novel_indices, 0], embedded[novel_indices, 1], 
            c=novel_color, marker='^', label='NP', alpha=0.7, s=100, zorder=10)

    
    # # plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='tab20', alpha=0.6)
    # labels = labels.numpy()
    # unique_labels = np.unique(labels)
    # # print('unique_labels:', unique_labels)
    # legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), markersize=10) for i in unique_labels]
    # legend_labels = [str(label) for label in unique_labels]
    # plt.legend(handles=legend_handles, labels=legend_labels, title="Classes")
    # # plt.legend(*scatter.legend_elements(), title="Classes")
    plt.legend(loc='best', fontsize=8)
    plt.title('t-SNE Visualization of Feature Vectors')
    plt.grid(True)
    plt.savefig(save_path)

def visualize_umap(features, labels, save_path='umap_ours_shot10_bfdecom_test.png'):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
    embedded = reducer.fit_transform(features.numpy())

    colors = [
    'gray', 'orange', 'green', 'purple', 'cyan', 'brown', 'pink', 'olive', 'darkblue', 'teal',
    'gold', 'lime', 'darkred', 'navy', 'darkgreen', 'magenta', 'tan', 'coral', 'turquoise', 'khaki'
    ]
    base_color = 'blue'
    novel_color = 'red'
    plt.figure(figsize=(10, 8))

    base_indices = np.where(labels == 21)[0]
    plt.scatter(embedded[base_indices, 0], embedded[base_indices, 1], 
            c=base_color, marker='p', label='Base Embeddings', alpha=0.7)

    # novel_embs (label = 22)
    novel_indices = np.where(labels == 22)[0]
    plt.scatter(embedded[novel_indices, 0], embedded[novel_indices, 1], 
            c=novel_color, marker='^', label='Novel Embeddings', alpha=0.7)

    # other labels (1-20)
    for i in range(1, 21):
        label_indices = np.where(labels == i)[0]
        plt.scatter(embedded[label_indices, 0], embedded[label_indices, 1], 
                c=colors[i-1], marker='o', label=f'Class {i-1}', alpha=0.7)
    # plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='tab20', alpha=0.6)
    # handles, labels = scatter.legend_elements()  # get legend handles and labels automatically
    # Create legend handles and labels for each class
    # labels = labels.numpy()
    # unique_labels = np.unique(labels)
    # print('unique_labels:', unique_labels)
    # legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i), markersize=10) for i in unique_labels]
    # legend_labels = [str(label) for label in unique_labels]
    # plt.legend(handles=legend_handles, labels=legend_labels, title="Classes")
    # plt.title('UMAP Visualization of Feature Vectors')
    plt.legend(loc='best', fontsize=8)
    plt.title('UMAP Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.savefig(save_path)

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def visualize_pca(model, save_path='pca_5novel.png'):
    # vectors = []
    for name, param in model.named_parameters():
        print(f'{name}: {param.shape}')
        if name == 'novel_emb':
            vectors = param
        # elif name == 'novel_emb':
            # vectors.append(param)
    # vectors = torch.cat(vectors, dim=0)  # [n, d]

    vectors = vectors.detach().cpu().numpy()
    # 创建 PCA 模型
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(vectors)

    # 创建颜色数组，每个向量对应一个颜色
    colors = np.arange(20)

    # 创建 3D 图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制向量
    for i in range(features_pca.shape[0]):
        ax.quiver(0, 0, 0, features_pca[i, 0], features_pca[i, 1], features_pca[i, 2], color=plt.cm.tab10(i), label=f'Vector {i+1}')

    # 设置坐标轴范围
    ax.set_xlim([min(features_pca[:, 0]), max(features_pca[:, 0])])
    ax.set_ylim([min(features_pca[:, 1]), max(features_pca[:, 1])])
    ax.set_zlim([min(features_pca[:, 2]), max(features_pca[:, 2])])

    # 设置图的标签
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    # 添加图例
    ax.legend()
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = get_parser()
    data_dir = '/users/data/GFSS/VOCdevkit2012/VOC2012'
    # train_list = './dataset/list/voc/trainaug.txt'
    # val_list = './dataset/list/voc/val.txt'
    novel_lists = [
    './dataset/list/voc/fold2/train_novel_class11.txt',
    './dataset/list/voc/fold2/train_novel_class12.txt',
    './dataset/list/voc/fold2/train_novel_class13.txt',
    './dataset/list/voc/fold2/train_novel_class14.txt',
    './dataset/list/voc/fold2/train_novel_class15.txt'
    ]
    base_lists = [
        f'./dataset/list/voc/fold0/train_base_class{i}.txt' for i in list(range(1, 11)) + list(range(16, 21))
    ]

    def get_samples_with_labels(file_list, label_list, num_samples=10):
        """
        获取样本及其对应的标签
        :param file_list: 文件路径列表
        :param label_list: 与文件路径对应的标签列表
        :param num_samples: 每个文件的采样数
        :return: 样本列表和标签列表
        """
        samples = []
        labels = []
        for file_path, label in zip(file_list, label_list):
            if os.path.exists(file_path):  # 检查文件是否存在
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # 取前 num_samples 行并分配对应的标签
                    samples.extend(lines[:num_samples])
                    labels.extend([label] * min(len(lines), num_samples))
            else:
                print(f"File not found: {file_path}")
        return samples, labels

    # 定义标签范围
    base_labels_list = list(range(1, 11)) + list(range(16, 21))  # Base labels: 1-10, 16-20
    novel_labels_list = list(range(11, 16))                     # Novel labels: 11-15

    # 获取 Base 和 Novel 样本及其对应标签
    base_samples, base_labels = get_samples_with_labels(base_lists, base_labels_list, num_samples=1)
    novel_samples, novel_labels = get_samples_with_labels(novel_lists, novel_labels_list, num_samples=1)


    # 合并所有样本和标签
    all_samples = novel_samples + base_samples
    all_labels = novel_labels + base_labels

    # 去掉换行符，并存入最终的列表
    all_samples = [sample.strip() for sample in all_samples]
    fold = 2
    shot = 10
    trainset = eval('dataset.' + 'tsne_data' + '.GFSSegTrain')(
                data_dir, all_samples, all_labels, fold, shot,
                crop_size=[473,473], base_size=[473,473], 
                mode='train')
    engine = Engine(custom_parser=parser)
    train_loader, train_sampler = engine.get_train_loader(trainset)
    
    model = GFSS_Model(n_base=15, fold=0, backbone='resnet50v2', dilated=True, os=8, n_novel=5, is_ft=True)
    model_path = '/users/exp/voc/pspnet_ekt/2/resnet50v2/Novel/1/ours/best_123.pth'
    checkpoint = torch.load(model_path)
    if 'module.' in list(checkpoint.keys())[0]:
        # 去除键的前缀
        new_state_dict = {k[7:]: v for k, v in checkpoint.items()}
        # 加载去除前缀的状态字典
        model.load_state_dict(new_state_dict)
    else:
        # 如果没有前缀，直接加载状态字典
        model.load_state_dict(checkpoint)
    
    features, labels = extract_features(model, train_loader)
    # print('features:', features.shape)
    base_embs = model.base_emb.detach()
    # print('base_embs:', base_embs.shape)
    novel_embs = model.novel_emb.detach()
    embs = torch.cat((base_embs, novel_embs), dim=0)
    features = torch.cat((features, embs), dim=0)
    print('features:', features.shape)
    # labels = torch.tensor([i for i in range(1, 21)], device=embs.device)
    labels_embs = torch.tensor([21] * 15 + [22] * 5, device=embs.device)
    labels = torch.cat((labels, labels_embs), dim=0)
    plt.hist(base_embs[0].flatten(), bins=50, alpha=0.5, label='BP')
    plt.hist(novel_embs[0].flatten(), bins=50, alpha=0.5, label='NP')
    plt.hist(features[0].flatten(), bins=50, alpha=0.5, label='Features')
    plt.legend()
    plt.show()

    # visualize_tsne(features, labels)
    # visualize_umap(features, labels)