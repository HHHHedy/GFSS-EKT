import argparse
import cv2
from datetime import datetime
import numpy as np
import sys
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import networks
import dataset
import os
from math import ceil
from utils.pyt_utils import load_model, get_confusion_matrix, set_seed, get_logger, intersectionAndUnionGPU
import torchvision.transforms as transforms
from PIL import Image

from engine import Engine

DATA_DIRECTORY = '/data/wzx99/pascal-context'
VAL_LIST_PATH = './dataset/list/context/val.txt'
COLOR_PATH = './dataset/list/context/context_colors.txt'
BATCH_SIZE = 1
INPUT_SIZE = '512,512'
BASE_SIZE = '2048,512'
RESTORE_FROM = '/model/lsa1997/deeplabv3_20200106/snapshots/CS_scenes_40000.pth'
SAVE_PATH = '/output/predict'
RANDOM_SEED = '123,234,345'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='Dataset for training')
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--train-list", type=str, default='',
                        help="Path to the file listing the images in the training set.")
    parser.add_argument("--val-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--test-batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument('--backbone', type=str, default='resnet101',
                    help='backbone model, can be: resnet101 (default), resnet50')
    parser.add_argument("--base-size", type=str, default=BASE_SIZE,
                        help="Base size of images for resize.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--save", type=str2bool, default='False',
                        help="save predicted image.")
    parser.add_argument("--os", type=int, default=8, help="output stride")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="path to save results")
    parser.add_argument("--random-seed", type=str, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument('--fold', type=int, default=1, choices=[0, 1, 2, 3], help='validation fold')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    parser.add_argument('--clip_pretrain', type=str, default='pretrain/RN50.pt',
                        help='backbone')
    parser.add_argument('--word_len', type=int, default=77, help='word length')
    return parser

def save_segmentation_result(image, seg_pred, save_path, name, color_map):
    """Save the predicted segmentation result."""
    seg_color = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        seg_color[seg_pred == label] = color
    seg_color = cv2.addWeighted(image, 1, seg_color, 0.7, 0)
    cv2.imwrite(os.path.join(save_path, name), seg_color)

def label2rgb(ind_im, color_map):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb

    return rgb_im

def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()
    num_classes = 20
    colors = {
        0: [0, 0, 0],
        1: [128, 0, 0],
        2: [0, 128, 0],
        3: [128, 128, 0],
        4: [0, 0, 128],
        5: [128, 0, 128],
        6: [0, 128, 128],
        7: [128, 128, 128],
        8: [64, 0, 0],
        9: [192, 0, 0],
        10: [64, 128, 0],
        11: [192, 128, 0],
        12: [64, 0, 128],
        13: [192, 0, 128],
        14: [64, 128, 128],
        15: [192, 128, 128],
        16: [0, 64, 0],
        17: [128, 64, 0],
        18: [0, 192, 0],
        19: [128, 192, 0],
        20: [0, 64, 128]
    }
    # colors = np.array([
    #     [0, 0, 0],         # 背景 - 黑色
    #     [128, 0, 0],       # 类别1 - 颜色可以根据需要调整
    #     [0, 128, 0],
    #     [128, 128, 0],
    #     [0, 0, 128],
    #     [128, 0, 128],
    #     [0, 128, 128],
    #     [128, 128, 128],
    #     [64, 0, 0],
    #     [192, 0, 0],
    #     [64, 128, 0],
    #     [192, 128, 0],
    #     [64, 0, 128],
    #     [192, 0, 128],
    #     [64, 128, 128],
    #     [192, 128, 128],
    #     [0, 64, 0],
    #     [128, 64, 0],
    #     [0, 192, 0],
    #     [128, 192, 0],
    #     [0, 64, 128]
    # ], dtype=np.uint8)
    
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            save_path = args.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            logger = get_logger('', save_path, date_str)

        cudnn.benchmark = False
        cudnn.deterministic = True

        h, w = map(int, args.base_size.split(','))
        args.base_size = (h, w)

        testset = eval('dataset.' + args.dataset + '.GFSSegVal')(
            args.data_dir, args.val_list, args.fold, 
            base_size=args.base_size, resize_label=False, use_novel=True, use_base=True)

        test_loader, test_sampler = engine.get_test_loader(testset)
        args.ignore_label = testset.ignore_label
        args.base_classes = len(testset.base_classes)
        args.novel_classes = len(testset.novel_classes)
        args.num_classes = testset.num_classes + 1 # consider background as a class

        # base_classes = sorted(list(testset.base_classes))
        # novel_classes = sorted(list(testset.novel_classes))
        base_classes = list(testset.base_classes)
        novel_classes = list(testset.novel_classes)
        
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger.info('[Testset] base_cls_list: ' + str(base_classes))
            logger.info('[Testset] novel_cls_list: ' + str(novel_classes))
            logger.info('[Testset] {} valid images are loaded!'.format(len(testset)))

        if engine.distributed:
            test_sampler.set_epoch(0)

        stride = args.os
        assert stride in [8, 16, 32]
        dilated = False if stride == 32 else True 
        seg_model = eval('networks.' + args.model + '.GFSS_Model')(
            n_base=args.base_classes, fold=args.fold, shot=args.shot, backbone=args.backbone,
                dilated=dilated, os=stride, is_ft=True, n_novel=args.novel_classes)
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger.info(seg_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)
        model = engine.data_parallel(seg_model)

        seed_list = args.random_seed
        seeds = list(map(int, seed_list.split(',')))
        for seed in seeds:
            restore_model = args.restore_from[:-4] + '.pth' # '_' + str(seed) + '.pth'
            load_model(model, restore_model)

            model.eval()

            # generalized few-shot segmentation evaluation (base + novel)
            confusion_matrix = np.zeros((args.num_classes, args.num_classes))
            miou_array = np.zeros((args.num_classes))
            for idx, data in enumerate(test_loader):
                image, label, id = data
                image = image.cuda()
                # text = text.cuda(non_blocking=True)

                with torch.no_grad():
                    output = model(img=image)
                    h, w = label.size(1), label.size(2)
                    longside = max(h, w)
                    output = F.interpolate(input=output, size=(image.shape[-2], image.shape[-1]), mode='bilinear', align_corners=True)
                    output = output[:, :, :h, :355]

                seg_pred = np.asarray(np.argmax(output.cpu().numpy(), axis=1), dtype=np.uint8)
                
                seg_pred = label2rgb(seg_pred[0], colors)
                pred = transforms.ToPILImage()(seg_pred.astype(np.uint8))
                # image = image.cpu()
                # image = image*std + mean
                # img = transforms.ToPILImage()(image[0].cpu())
                # pred = pred.convert('RGBA')
                # img = img.convert('RGBA')
                # pred.putalpha(128)
                # pred = Image.alpha_composite(img, pred)
            

                # new_im = Image.new('RGB', (pred.size[0], pred.size[1]), "white")
                # new_im.paste(pred, (0, 0))
                # new_im.save(os.path.join(save_path, '{}_{}.png'.format(id, seed)))
                # new_im.close()

                # 将输入图像转换为可保存的格式
                image = image.cpu()
                image = image * std + mean  # 假设已经有标准化参数 std 和 mean
                input_image = transforms.ToPILImage()(image[0].cpu())

                # 保存预测图像
                pred_path = os.path.join(save_path, '{}_{}_pred.png'.format(id, seed))
                pred.save(pred_path)

                # 保存输入图像
                input_path = os.path.join(save_path, '{}_{}_input.png'.format(id, seed))
                input_image.save(input_path)

                # 可选：同时保存合成图像（输入和预测叠加）
                pred = pred.convert('RGBA')
                input_image = input_image.convert('RGBA')
                pred.putalpha(128)  # 设置预测图像透明度
                overlay = Image.alpha_composite(input_image, pred)
                overlay_path = os.path.join(save_path, '{}_{}_overlay.png'.format(id, seed))
                overlay.save(overlay_path)

                # 清理资源
                pred.close()
                input_image.close()
                overlay.close()

                # print(np.unique(seg_pred))
                seg_gt = np.asarray(label.numpy(), dtype=np.int_)
                pad_gt = np.ones((seg_gt.shape[0], longside, longside), dtype=np.int_) * args.ignore_label
                pad_gt[:, :h, :w] = seg_gt
                seg_gt = pad_gt
                #print(image.shape, seg_pred.shape)
                # image = image[0].permute(1, 2, 0).cpu().numpy()
                # image = (image * 255).astype(np.uint8)
                #save_segmentation_result(image, seg_pred[0], save_path, '{}_{}.png'.format(idx, seed), colors)
                
                ignore_index = seg_gt != args.ignore_label
                seg_gt = seg_gt[ignore_index]
                seg_pred = seg_pred[ignore_index]
                # print(seg_gt.shape, seg_pred.shape)
                confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)
                

            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            miou_array = (tp / (pos + res - tp))
            base_miou = np.nanmean(miou_array[:args.base_classes+1])
            novel_miou = np.nanmean(miou_array[args.base_classes+1:])
            total_miou = np.nanmean(miou_array)

            # np.save(os.path.join(save_path, 'cmatrix_{}.npy'.format(seed)), confusion_matrix)

            if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                logger.info('>>>>>>> Current Seed {}: <<<<<<<'.format(seed))
                logger.info('meanIoU---base: mIoU {:.4f}.'.format(base_miou))
                logger.info('meanIoU---novel: mIoU {:.4f}.'.format(novel_miou))
                logger.info('meanIoU---total: mIoU {:.4f}.'.format(total_miou))

if __name__ == '__main__':
    main()
