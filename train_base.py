import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import networks
import dataset
import cv2

import random
import time
import logging
import utils.pyt_utils as my_utils
from loss import get_loss
from engine import Engine
from dataset.base_dataset import BaseDataset
from networks.backbones import get_backbone

BATCH_SIZE = 8
DATA_DIRECTORY = '/data/pascal-context'
DATA_LIST_PATH = './dataset/list/context/train.txt'
VAL_LIST_PATH = './dataset/list/context/val.txt'
INPUT_SIZE = '512,512'
TEST_SIZE = '512,512'
BASE_SIZE = '2048,512'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_STEPS = 100
POWER = 0.9
RANDOM_SEED = 321
RESTORE_FROM = '/home/model/resnet_backbone/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5
SNAPSHOT_DIR = '/home/output'

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
    parser = argparse.ArgumentParser(description="Few-shot Segmentation Framework")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='Dataset for training')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--train-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the training set.")
    parser.add_argument("--base-size", type=str, default=BASE_SIZE,
                        help="Base size of images for resize.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Which epoch to start.")
    parser.add_argument("--num-epoch", type=int, default=NUM_STEPS,
                        help="Number of training epochs.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="choose the number of workers.")
    parser.add_argument('--backbone', type=str, default='resnet50',
                    help='backbone model, can be: resnet101, resnet50 (default)')
    parser.add_argument("--os", type=int, default=8, help="output stride")
    parser.add_argument("--print-frequency", type=int, default=100,
                        help="Number of training steps.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument('--fold', type=int, default=0, choices=[-1, 0, 1, 2, 3], help='validation fold')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    parser.add_argument("--val-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the file listing the images in the val set.")
    parser.add_argument("--test-batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--fix-bn", action="store_true", default=False,
                        help="whether to fix batchnorm during training.")
    parser.add_argument("--filter-novel", action="store_true", default=False,
                        help="whether to filter images containing novel classes during training.")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="whether to freeze the backbone during training.")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="whether to use mix precision training.")
    parser.add_argument("--resnet", type=str, default='./initmodel/backbones/resnet50_v2.pth')
    
    return parser

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, lr, index_split=-1, scale_lr=10.):
    for index in range(len(optimizer.param_groups)):
        if index <= index_split:
            optimizer.param_groups[index]['lr'] = lr
        else:
            optimizer.param_groups[index]['lr'] = lr * scale_lr
    return lr

def adjust_learning_rate_poly(optimizer, learning_rate, i_iter, max_iter, power, freeze_backbone=False):
    split = -1 if freeze_backbone else 0
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    lr = adjust_learning_rate(optimizer, lr, index_split=split)
    return lr

def Weighted_GAP(supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2], supp_feat.shape[-1]
        area = F.avg_pool2d(mask, (feat_h, feat_w)) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        supp_feat = supp_feat.squeeze(-1).squeeze(-1)  # [1, 2048]
        return supp_feat

def generate_protos(base_id_list, args, norm_layer):
    # backbone = get_backbone(args.backbone, args.resnet, norm_layer)
    dataset = BaseDataset(crop_size=args.input_size)
    interval = args.num_classes // 4
    base_classes = set(range(1, args.num_classes + 1)) - set(range(interval * args.fold + 1, interval * (args.fold + 1) + 1))
    novel_classes = set(range(interval * args.fold + 1, interval * (args.fold + 1) + 1))
    base_classes = list(base_classes)
    novel_classes = list(novel_classes)
    img_dir = 'JPEGImages'
    lbl_dir = 'SegmentationClassAug'
    base_images = []
    base_labels = []
    for cls in base_id_list.keys():
        # print('base_id:', cls, len(base_id_list[cls]))
        if len(base_id_list[cls]) >= args.shot:
            ids = random.sample(base_id_list[cls], args.shot)
        else:
            ids = random.choices(base_id_list[cls], k=args.shot)
        for id in ids:
            # print('id:', id, 'ids', ids)
            image = cv2.imread(osp.join(args.data_dir, img_dir, '%s.jpg'%id), cv2.IMREAD_COLOR)
            # print('image:', image)
            label = cv2.imread(osp.join(args.data_dir, lbl_dir, '%s.png'%id), cv2.IMREAD_GRAYSCALE)
            image, label = dataset.resize(image, label, random_scale=True)
            image, label = dataset.random_flip(image, label)
            image, label = dataset.crop(image, label)
            image = dataset.normalize(image)
            image, label = dataset.pad(args.base_size, image, label)
            image, label = dataset.totensor(image, label)
            # print(image.shape, label.shape)
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            base_images.append(image)
            base_labels.append(label)
    base_images = torch.cat(base_images, dim=0)  # [len(base_classes)*args.shot, 3, 473, 473]
    base_labels = torch.cat(base_labels, dim=0)  # [len(base_classes)*args.shot, 473, 473]
    '''proto_avg = []
    for i in range(len(base_classes)):
        per_base_imgs = base_images[i*args.shot:(i+1)*args.shot]
        per_base_labels = base_labels[i*args.shot:(i+1)*args.shot]
        protos = []
        for j in range(args.shot):
            img = per_base_imgs[j]
            feature = backbone.base_forward(img)
            lbl = per_base_labels[j]
            mask = (lbl == base_classes[i]).float()
            mask = F.interpolate(mask.unsqueeze(0), size=feature.size()[-2:], mode='nearest').squeeze(0)
            proto = Weighted_GAP(feature, mask)
            protos.append(proto)
        proto_avg.append(torch.mean(torch.cat(protos, dim=0), dim=0))
    proto_avg = torch.stack(proto_avg, dim=0) # [len(novel_classes), 1024]'''

    return base_images, base_labels

def print_gpu_memory_usage():
    rank = dist.get_rank()  # 获取当前进程的全局排名
    allocated_memory = torch.cuda.memory_allocated(rank) / (1024 ** 3)  # 转换为GB
    reserved_memory = torch.cuda.memory_reserved(rank) / (1024 ** 3)  # 转换为GB
    print(f"Rank {rank} - Allocated Memory: {allocated_memory:.2f} GB, Reserved Memory: {reserved_memory:.2f} GB")

def main():
    """Create the model and start the training."""
    global logger
    parser = get_parser()
    torch_ver = torch.__version__[:3]
    use_val = True
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger = my_utils.prep_experiment(args, need_writer=False)
        cudnn.benchmark = True
        seed = args.random_seed
        if seed > 0:
            my_utils.set_seed(seed)

        # data loader
        h1, w1 = map(int, args.input_size.split(','))
        args.input_size = (h1, w1)
        h2, w2 = map(int, args.base_size.split(','))
        args.base_size = (h2, w2)

        trainset = eval('dataset.' + args.dataset + '.GFSSegTrain')(
            args.data_dir, args.train_list, args.fold, args.shot,
            crop_size=args.input_size, base_size=args.base_size, 
            mode='train', filter=args.filter_novel)
        train_loader, train_sampler = engine.get_train_loader(trainset)
        args.ignore_label = trainset.ignore_label
        args.num_classes = trainset.num_classes
        args.base_classes = len(trainset.base_classes)

        testset = eval('dataset.' + args.dataset + '.GFSSegVal')(
            args.data_dir, args.val_list, args.fold, 
            base_size=args.base_size, resize_label=True, use_novel=False)            
        test_loader, test_sampler = engine.get_test_loader(testset)
        args.novel_classes = len(testset.novel_classes)
        if engine.distributed:
            test_sampler.set_epoch(0)

        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger.info('[Trainset] base_cls_list: ' + str(trainset.base_classes))
            logger.info('[Trainset] {} valid images are loaded!'.format(len(trainset.data_list)))
            logger.info('[Testset] base_cls_list: ' + str(testset.base_classes))
            logger.info('[Testset] {} valid images are loaded!'.format(len(testset.ids)))

        criterion = get_loss(args)
        if engine.distributed:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm2d

        stride = args.os
        assert stride in [8, 16, 32]
        dilated = False if stride == 32 else True 
        if args.start_epoch > 0:
            seg_model = eval('networks.' + args.model + '.GFSS_Model')(
                n_base=args.base_classes, fold=args.fold, shot=args.shot,
                criterion=criterion, backbone=args.backbone, norm_layer=BatchNorm,
                dilated=dilated, os=stride, n_novel=args.novel_classes
                )
        else:
            seg_model = eval('networks.' + args.model + '.GFSS_Model')(
                n_base=args.base_classes, fold=args.fold, shot=args.shot,
                criterion=criterion, backbone=args.backbone, pretrained_model=args.restore_from, 
                norm_layer=BatchNorm, dilated=dilated, os=stride, n_novel=args.novel_classes
                )

        if args.start_epoch > 0:
            my_utils.load_model(seg_model, args.restore_from, is_restore=True)

        params = my_utils.get_parameters(seg_model, lr=args.learning_rate, freeze_backbone=args.freeze_backbone)  # freeze backbone. no module
        if 'pspnet' in args.model or 'fpn' in args.model:
            optimizer = optim.SGD(params, lr=args.learning_rate, 
                momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = optim.AdamW(params, lr=args.learning_rate, 
                weight_decay=args.weight_decay)
        
        optimizer.zero_grad()

        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger.info(seg_model)

        model = engine.data_parallel(seg_model.cuda())  # have module
        loss_scaler = my_utils.NativeScalerWithGradNormCount()

        print('freeze_backbone', args.freeze_backbone)
        for name, param in model.named_parameters():
            print(name, ':', param.requires_grad)

        if not os.path.exists(args.snapshot_dir) and engine.local_rank == 0:
            os.makedirs(args.snapshot_dir)

        global_iteration = args.start_epoch * len(train_loader)
        max_iteration = args.num_epoch * len(train_loader)
        loss_dict_memory = {}
        lr = args.learning_rate
        best_miou = 0
        best_epoch = 0
        for epoch in range(args.start_epoch, args.num_epoch):
            if seed > 0:
                my_utils.set_seed(seed+epoch)
            
            epoch_log = epoch + 1
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            
            if args.freeze_backbone:
                model.module.train_mode() # freeze backbone and BN
            else:
                model.train()

            for i, data in enumerate(train_loader):
                global_iteration += 1
                img, mask, img_w, img_s = data
                img, mask, img_w, img_s = img.cuda(non_blocking=True), mask.cuda(non_blocking=True), img_w.cuda(non_blocking=True), img_s.cuda(non_blocking=True)
                # support_imgs, support_labels = generate_protos(trainset.train_dict, args, BatchNorm)
                # support_imgs = support_imgs.cuda(non_blocking=True)
                # support_labels = support_labels.cuda(non_blocking=True)

                lr = adjust_learning_rate_poly(optimizer, args.learning_rate, global_iteration-1, max_iteration, args.power, args.freeze_backbone)

                with torch.cuda.amp.autocast(enabled=args.fp16):
                    loss_dict = model(img=img, mask=mask, img_w=img_w, img_s=img_s)

                total_loss = loss_dict['total_loss']
                grad_norm = loss_scaler(total_loss, optimizer, clip_grad=5.0,
                                parameters=model.parameters())
                
                optimizer.zero_grad()

                for loss_name, loss in loss_dict.items():
                    loss_dict_memory[loss_name] = engine.all_reduce_tensor(loss).item()
                # loss_dict_memory['assist_loss'] = engine.all_reduce_tensor(assist_loss).item()

                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if i % args.print_frequency == 0:
                        print_str = 'Epoch{}/Iters{}'.format(epoch_log, global_iteration) \
                            + ' Iter{}/{}:'.format(i + 1, len(train_loader)) \
                            + ' lr=%.2e' % lr \
                            + ' grad_norm=%.4f' % grad_norm 
                        for loss_name, loss_value in loss_dict_memory.items():
                            print_str += ' %s=%.4f' % (loss_name, loss_value)
                        logger.info(print_str)
                # dist.barrier()
                # if torch.cuda.is_available() and dist.is_initialized():
                #     print_gpu_memory_usage()

            if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                if epoch_log % 10 == 0 or epoch_log >= args.num_epoch:
                    print('taking snapshot ...')
                    if torch_ver < '1.6':
                        torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'epoch_'+str(epoch_log)+'.pth'))
                    else:
                        torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'epoch_'+str(epoch_log)+'.pth'), _use_new_zipfile_serialization=False)

            if use_val and epoch > 1 and (epoch % 2 == 0 or epoch == args.num_epoch-1):
                # support_imgs, support_labels = generate_protos(trainset.train_dict, args, BatchNorm)
                # support_imgs = support_imgs.cuda(non_blocking=True)
                # support_labels = support_labels.cuda(non_blocking=True)
                inter, union = validate(model, test_loader, args)
                inter =  engine.all_reduce_tensor(inter, norm=False)
                union =  engine.all_reduce_tensor(union, norm=False)
                miou_array = inter / union
                # miou = np.nanmean(miou_array[1:]) # exclude background when calculating mean IoU
                
                miou_array = miou_array.cpu().numpy()
                miou = np.nanmean(miou_array)
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if miou >= best_miou:
                        print('taking snapshot ...')
                        if torch_ver < '1.6':
                            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'best.pth'))
                        else:
                            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'best.pth'), _use_new_zipfile_serialization=False)
                        best_miou = miou
                        best_epoch = epoch_log
                    logger.info('>>>>>>> Evaluation Results: <<<<<<<')
                    logger.info('meanIU: {:.2%}, best_IU: {:.2%}, best_epoch: {}'.format(miou,best_miou,best_epoch))
                    logger.info('>>>>>>> ------------------- <<<<<<<')

def validate(model, dataloader, args):
    '''
        Validation on base classes (only for training)
    '''
    model.eval()
    num_classes = args.base_classes + 1 # 0 for background
    inter_meter = torch.zeros(num_classes).cuda()
    union_meter = torch.zeros(num_classes).cuda()

    for idx, data in enumerate(dataloader):
        img, mask, _ = data
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16):
                output = model(img=img)
            h, w = mask.size(1), mask.size(2)
            output = F.interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)

        output = output.max(1)[1]
        # print('output:', output.shape, 'mask:', mask.shape)
        intersection, union, _ = my_utils.intersectionAndUnionGPU(output, mask, num_classes, args.ignore_label)
        inter_meter += intersection
        union_meter += union

    return inter_meter, union_meter
    
if __name__ == '__main__':
    main()