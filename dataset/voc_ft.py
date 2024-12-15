import os
import os.path as osp
import numpy as np
import random
import cv2
from PIL import Image
from collections import defaultdict
from typing import List, Union
import torch
from torchvision.transforms import RandomErasing

from .base_dataset import BaseDataset
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .augmentations import RandAugment
from .randaugment import RandAugmentMC
import matplotlib.pyplot as plt

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class GFSSegTrain(BaseDataset):
    num_classes = 20
    def __init__(self, root, list_path, fold, shot=1, mode='train', crop_size=(512, 512),
             ignore_label=255, base_size=(2048,512), resize_label=False, seed=123, filter=False, use_base=True, word_len=7):
        super(GFSSegTrain, self).__init__(mode, crop_size, ignore_label, base_size=base_size)
        assert mode in ['train', 'val_supp']
        self.root = root
        self.list_path = list_path
        self.fold = fold
        self.shot = shot
        self.mode = mode
        self.resize_label = resize_label
        self.use_base = use_base
        self.img_dir = 'JPEGImages'
        self.lbl_dir = 'SegmentationClassAug'
        self.word_len = word_len
        self.RE = RandomErasing(p=1, scale=(0.01, 0.1), ratio=(0.3, 3.3), value='random')
        self.strong_aug = RandAugmentMC(n=2, m=10)

        self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat',
                       'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'potted-plant',
                       'sheep', 'sofa', 'train', 'tv/monitor']
        interval = self.num_classes // 4
        # base classes = all classes - novel classes
        self.base_classes = set(range(1, self.num_classes + 1)) - set(range(interval * fold + 1, interval * (fold + 1) + 1))
        self.base_classnames = [self.classnames[i-1] for i in self.base_classes]
        # novel classes
        self.novel_classes = set(range(interval * fold + 1, interval * (fold + 1) + 1))
        self.novel_classnames = [self.classnames[i-1] for i in self.novel_classes]
        self.all_classes = self.base_classes | self.novel_classes

        filter_flag = True if (self.mode == 'train' and filter) else False

        list_dir = os.path.dirname(self.list_path)
        list_dir = list_dir + '/fold%s'%fold
        if filter_flag:
            list_dir = list_dir + '_filter'
        list_saved = os.path.exists(os.path.join(list_dir, 'train_base_class%s.txt'%(list(self.base_classes)[0])))
        if list_saved:
            print('id files exist...')
            self.base_cls_to_ids = defaultdict(list)
            for cls in self.base_classes:
                with open(os.path.join(list_dir, 'train_base_class%s.txt'%cls), 'r') as f:
                    self.base_cls_to_ids[cls] = f.read().splitlines()
        else:
            '''
            fold0/train_fold0.txt: training images containing base classes (novel classes will be ignored during training)
            fold0/train_base_class[6-20].txt: training images containing base class [6-20]
            fold0/train_novel_class[1-5].txt: training images containing novel class [1-5]
            '''
            with open(os.path.join(self.list_path), 'r') as f:
                self.ids = f.read().splitlines()
            print('checking ids...')
            
            self.base_cls_to_ids, self.novel_cls_to_ids = self._filter_and_map_ids(filter_intersection=filter_flag)
            for cls in self.base_classes:
                with open(os.path.join(list_dir, 'train_base_class%s.txt'%cls), 'w') as f:
                    for id in self.base_cls_to_ids[cls]:
                        f.write(id+"\n")

        with open(os.path.join(list_dir, 'fold%s_%sshot_seed%s.txt'%(fold, shot, seed)), 'r') as f:
            self.novel_id_list = f.read().splitlines()
        self.novel_cls_to_ids = defaultdict(list)
        for cls in self.novel_classes:
            with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'r') as f:
                self.novel_cls_to_ids[cls] = f.read().splitlines()
        self.base_unlabel_id_list = [id for sublist in self.base_cls_to_ids.values() for id in sublist]
        self.novel_id_set = set(self.novel_id_list)
        self.filtered_base_unlabel_id_list = [id for id in self.base_unlabel_id_list if id not in self.novel_id_set]
        # with open(os.path.join(list_dir, 'fold%s_%sshot_seed%s_base.txt'%(fold, shot, seed)), 'r') as f:
            # self.base_ids = f.read().splitlines()
        if self.use_base:
            self.supp_cls_id_list, self.base_id_list, self.base_classname_list, self.novel_classname_list = self._get_supp_list()
        else:
            self.supp_cls_id_list = self.novel_id_list

    def __len__(self):
        if self.mode == 'val_supp':
            return len(self.novel_classes) + len(self.base_classes) if self.use_base else len(self.novel_classes)
        else:
            return len(self.base_id_list)

    def _convert_label(self, label, from_base=False, from_novel=False):
        new_label = label.copy()
        label_class = np.unique(label).tolist()
        
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)
        for c in label_class:
            if c in base_list:
                if from_novel:
                    new_label[label == c] = 0
                else:
                    new_label[label == c] = (base_list.index(c) + 1)    # 0 as background
                
            elif c in novel_list:
                if from_base:
                    new_label[label == c] = 0 # as background
                else:
                    new_label[label == c] = (novel_list.index(c) + len(base_list) + 1)
                    
            else:
                raise ValueError("unexpected class label")

        return new_label

    def update_base_list(self):  # useless?
        base_list = list(self.base_classes)
        base_id_list = []
        id_s_list = []
        base_with_novel = 0
        for target_cls in base_list:
            file_class_chosen = self.base_cls_to_ids[target_cls]  # list of image ids for each base class. eg.331 images for class 6
            num_file = len(file_class_chosen)
            assert num_file >= self.shot

            for k in range(self.shot):
                id_s = ' '
                while((id_s == ' ') or id_s in id_s_list or id_s in self.novel_id_list):
                    support_idx = random.randint(1, num_file) - 1
                    id_s = file_class_chosen[support_idx]                
                id_s_list.append(id_s)
                base_id_list.append(id_s)

                label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id_s), cv2.IMREAD_GRAYSCALE)
                label_class = np.unique(label).tolist()
                if 0 in label_class:
                    label_class.remove(0)
                if 255 in label_class:
                    label_class.remove(255)
                if set(label_class).issubset(self.base_classes):
                    pass
                else:
                    base_with_novel += 1
        print('%s base images contain novel classes'%base_with_novel)
        
        self.supp_cls_id_list = self.novel_id_list + base_id_list
        self.base_id_list = base_id_list

    def _get_supp_list(self):  # get subset of base classes (with the same number of each novel class)
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)
        base_id_list = []
        base_classname_list = []
        novel_classname_list = []
        novel_id_list = self.novel_id_list
        id_s_list = []
        for id in novel_id_list:  # list of image ids for novel samples (len=class_num*shot)
            id_s_list.append(id)  # len=25

        # assign novel class name
        for target_cls in novel_list:  # novel class
            novel_class_label = self.classnames[target_cls-1]
            novel_classname_list.append(novel_class_label)

        base_with_novel = 0
        for target_cls in base_list:  # base class
            base_class_label = self.classnames[target_cls-1]  # base class name
            file_class_chosen = self.base_cls_to_ids[target_cls]  # list of image ids for each base class. eg.331 images for class 6
            num_file = len(file_class_chosen)
            assert num_file >= self.shot

            for k in range(self.shot):
                id_s = ' '
                while((id_s == ' ') or id_s in id_s_list):
                    support_idx = random.randint(1, num_file) - 1
                    id_s = file_class_chosen[support_idx]               
                id_s_list.append(id_s)  # avoid repeated image ids
                base_id_list.append(id_s)  # list of image ids for base samples (len=class_num*shot)
                base_classname_list.append(base_class_label)
                
                label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id_s), cv2.IMREAD_GRAYSCALE)
                label_class = np.unique(label).tolist()
                if 0 in label_class:
                    label_class.remove(0)
                if 255 in label_class:
                    label_class.remove(255)
                if set(label_class).issubset(self.base_classes):
                    pass
                else:
                    base_with_novel += 1
        print('%s base images contain novel classes'%base_with_novel)
        supp_cls_id_list = novel_id_list + base_id_list
        return supp_cls_id_list, base_id_list, base_classname_list, novel_classname_list

    def __getitem__(self, index):
        if self.mode == 'val_supp':
            return self._get_val_support(index)
        else:
            return self._get_train_sample(index)

    def _get_train_sample(self, index):
        # id_b = self.base_id_list[index]  # index of base_id_list
        unlabel_id_list = self.filtered_base_unlabel_id_list # + self.novel_id_list
        id_b = random.choice(unlabel_id_list)  # unlabelled novel samples
        # extend_novel_id_list = [self.novel_id_list[i % len(self.novel_id_list)] for i in range(len(self.base_id_list))]
        id = random.choice(self.novel_id_list)  # random choice from novel_id_list
        image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id), cv2.IMREAD_COLOR)
        label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id), cv2.IMREAD_GRAYSCALE)

        image_b = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id_b), cv2.IMREAD_COLOR)
        label_b = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id_b), cv2.IMREAD_GRAYSCALE)

        label = self._convert_label(label, from_base=False, from_novel=False)
        # date augmentation & preprocess  weak augmentation
        image, label = self.resize(image, label, random_scale=True)
        image, label = self.random_flip(image, label)
        image, label = self.crop(image, label)
        
        # image = self.cutout(image, label, n_holes=1, length=16)
        image = self.normalize(image)
        image, label = self.pad(self.crop_size, image, label)
        image, label = self.totensor(image, label)

        label_b = self._convert_label(label_b, from_base=False, from_novel=False)  # convert label to novel label
        # date augmentation & preprocess  strong augmentation
        image_b, label_b = self.resize(image_b, label_b, random_scale=True)
        image_b, label_b = self.random_flip(image_b, label_b)
        image_b, label_b = self.crop_context(image_b, label_b)
        
        '''image_b_copy = image_b.copy()
        image_b_rgb = cv2.cvtColor(image_b_copy, cv2.COLOR_BGR2RGB)
        image_b_pil = Image.fromarray(image_b_rgb)
        image_s = self.strong_aug(image_b_pil)
        image_s = np.array(image_s)
        image_s = self.normalize(image_s)
        image_s = self.pad(self.crop_size, image_s)
        image_s = self.totensor(image_s)'''
        
        image_s = image_b.copy()
        image_s = self.cutout(image_s, n_holes=1, length=16)
        '''image_bgr = image_s / 255.0
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.savefig('image'+str(index)+'.png')'''
        # plt.imshow(image_s)
        # plt.savefig('image'+str(index)+'.png')
        image_s = self.normalize(image_s)
        image_s = self.pad(self.crop_size, image_s)
        image_s = self.totensor(image_s)
        image_w = image_b.copy()
        image_w = self.normalize(image_w)
        image_w = self.pad(self.crop_size, image_w)
        image_w = self.totensor(image_w)
         
        
        return image, label, image_w, image_s

    def _get_val_support(self, index):
        if self.use_base:
            if index < len(self.base_classes):
                cls_id_list = self.base_id_list
                cls_idx = index
                target_cls = list(self.base_classes)[cls_idx]
            else:
                cls_id_list = self.novel_id_list
                cls_idx = index - len(self.base_classes)
                target_cls = list(self.novel_classes)[cls_idx]
        else:
            cls_id_list = self.novel_id_list
            cls_idx = index
            target_cls = list(self.novel_classes)[cls_idx]

        id_s_list, image_s_list, label_s_list = [], [], []

        for k in range(self.shot):
            id_s = cls_id_list[cls_idx*self.shot+k]              
            image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id_s), cv2.IMREAD_COLOR)
            label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id_s), cv2.IMREAD_GRAYSCALE)

            new_label = label.copy()
            new_label[(label != target_cls) & (label != self.ignore_label)] = 0
            new_label[label == target_cls] = 1
            label = new_label.copy()
            # date augmentation & preprocess
            image, label = self.resize(image, label)
            image = self.normalize(image)
            image, label = self.pad(self.crop_size, image, label)
            image, label = self.totensor(image, label)
            id_s_list.append(id_s)
            image_s_list.append(image)
            label_s_list.append(label)
        return image_s_list, label_s_list, id_s_list, target_cls

    def _filter_and_map_ids(self, filter_intersection=False):
        base_cls_to_ids = defaultdict(list)
        novel_cls_to_ids = defaultdict(list)
        for i in range(len(self.ids)):
            mask = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%self.ids[i]), cv2.IMREAD_GRAYSCALE)
            label_class = np.unique(mask).tolist()
            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255)
            valid_base_classes = set(np.unique(mask)) & self.base_classes
            valid_novel_classes = set(np.unique(mask)) & self.novel_classes

            new_label_class = []
            if valid_base_classes:
                if filter_intersection:
                    if set(label_class).issubset(self.base_classes):
                        
                        for cls in valid_base_classes:
                            if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                                new_label_class.append(cls)
                else:
                    for cls in valid_base_classes:
                        if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                            new_label_class.append(cls)

                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        base_cls_to_ids[cls].append(self.ids[i])

            if valid_novel_classes:
            # remove images whose valid objects are all small (according to PFENet)
                new_label_class = []
                for cls in valid_novel_classes:
                    if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                        new_label_class.append(cls)

                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        novel_cls_to_ids[cls].append(self.ids[i])

        return base_cls_to_ids, novel_cls_to_ids
