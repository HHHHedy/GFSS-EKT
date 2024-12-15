import numpy as np
import random
import torch
import cv2
from scipy.ndimage import label, find_objects
from torch.utils import data
import matplotlib.pyplot as plt

class BaseDataset(data.Dataset):
    def __init__(self, mode='train', crop_size=(512, 512), 
            ignore_label=255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], base_size=(512, 512)):
        self.crop_size = crop_size

        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.padding = [val*255.0 for val in self.mean]
        
        self.ids = []
        self.mode = mode

        self.base_size = base_size
        self.ratio_range = (0.9, 1.1)
        self.blur_radius = 5
        self.rotate_range = (-10, 10)
        self.fg_remain_ratio = 0.85

    def __len__(self):
        return len(self.ids)

    def normalize(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def totensor(self, image, label=None):
        image = image.transpose((2, 0, 1)) # [H, W, C] -> [C, H, W]
        image = torch.from_numpy(image.copy()).float()
        if label is not None:
            label = torch.from_numpy(label.copy()).long()
            return image, label
        return image

    def resize(self, image, label=None, random_scale=False):
        if random_scale:
            min_ratio, max_ratio = self.ratio_range
            f_scale = random.random() * (max_ratio - min_ratio) + min_ratio
            dsize = int(image.shape[1] * f_scale + 0.5), int(image.shape[0] * f_scale + 0.5)
            image_scale = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            label_scale = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
            return image_scale, label_scale
        else:
            output_size = self.base_size[0], self.base_size[1]
            scale_factor = min(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            new_w = int(image.shape[1] * scale_factor + 0.5)
            new_h = int(image.shape[0] * scale_factor + 0.5)
            dsize = new_w, new_h
            image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            if label is not None:
                label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
                return image, label
            else:
                return image

    def square_resize(self, image, label=None, random_scale=False):
        if random_scale:
            min_ratio, max_ratio = self.ratio_range
            f_scale = random.random() * (max_ratio - min_ratio) + min_ratio
            dsize = int(self.base_size[1] * f_scale + 0.5), int(self.base_size[0] * f_scale + 0.5)
        else:
            dsize = self.base_size[1], self.base_size[0]
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
            return image, label
        else:
            return image

    def pad(self, output_size, image, label=None):
        pad_h = max(output_size[0] - image.shape[0],0)
        pad_w = max(output_size[1] - image.shape[1],0)
        if pad_h > 0 or pad_w > 0:
            image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            if label is not None:
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(self.ignore_label,))
        else:
            image_pad, label_pad = image, label
        if label is not None:
            return image_pad, label_pad
        else:
            return image_pad

    def random_flip(self, image, label, p=0.5):
        if random.random() < p:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        return image, label

    def random_gaussian(self, image, p=0.5):
        if random.random() < p:
            image = cv2.GaussianBlur(image, (self.blur_radius, self.blur_radius), 0)
        return image

    def random_rotate(self, image, label, p=0.5):
        if random.random() < p:
            rotate_cnt = 0
            while (rotate_cnt < 5):
                angle = self.rotate_range[0] + (self.rotate_range[1] - self.rotate_range[0]) * random.random()
                h, w = label.shape
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                image_tmp = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
                label_tmp = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
                if np.sum(label_tmp == 1) > 0:
                    break
                rotate_cnt += 1
            if rotate_cnt < 5:
                image = image_tmp
                label = label_tmp
        return image, label

    def crop(self, image, label, target_cls=1):
        img_h, img_w = label.shape
        crop_h, crop_w = self.crop_size
        margin_h = max(img_h - crop_h, 0)
        margin_w = max(img_w - crop_w, 0)
        if self.mode == 'train':
            h_off = np.random.randint(0, margin_h + 1)
            w_off = np.random.randint(0, margin_w + 1)
            # if self.fg_remain_ratio > 0:
            #     raw_pos_num = np.sum(label == target_cls)
            #     label_temp = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
            #     pos_num = np.sum(label_temp == target_cls)
            #     crop_cnt = 0
            #     while (pos_num < self.fg_remain_ratio * raw_pos_num and crop_cnt <= 30):
            #         h_off = np.random.randint(0, margin_h + 1)
            #         w_off = np.random.randint(0, margin_w + 1)
            #         label_temp = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
            #         pos_num = np.sum(label_temp == 1)
            #         crop_cnt += 1
            #     if pos_num < self.fg_remain_ratio * raw_pos_num:
            #         image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            #         label = cv2.resize(label, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)            
            #         return image, label
        else:
            h_off = int(round(margin_h / 2.))
            w_off = int(round(margin_w / 2.))
        image = image[h_off : h_off+crop_h, w_off : w_off+crop_w]
        label = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
        return image, label
    
    def gaussian_distribution(self, center, sigma, shape):
        x = np.arange(0, shape[1], 1, float)
        y = np.arange(0, shape[0], 1, float)[:, np.newaxis]
        x0 = center[1]
        y0 = center[0]
        return np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
    
    def crop_context(self, image, mask):
        ## step 1: extract the bounding box of the object from the label
        img_h, img_w = mask.shape
        crop_h, crop_w = self.crop_size
        margin_h = max(img_h - crop_h, 0)
        margin_w = max(img_w - crop_w, 0)
        labels = np.unique(mask)
        labels = labels[labels != self.ignore_label]
        labels = labels[labels != 0]
        labeled_objects = {}
        for label_id in labels:
            binary_mask = (mask == label_id).astype(np.uint8)
            labeled_array, num_features = label(binary_mask)
            bboxes = find_objects(labeled_array)
            labeled_objects[label_id] = bboxes
        all_bboxes = []
        for bboxes in labeled_objects.values():
            all_bboxes.extend(bboxes)
        gaussian_map_list = []
        for bbox in all_bboxes:
            center = [(bbox[0].start + bbox[0].stop) / 2, (bbox[1].start + bbox[1].stop) / 2]
            sigma = np.mean([(bbox[0].stop - bbox[0].start) / 2, (bbox[1].stop - bbox[1].start) / 2])
            gaussian_map = self.gaussian_distribution(center, sigma, mask.shape)
            gaussian_map_list.append(gaussian_map)
        ## step 2: crop the image and label
        # get a mixture of gaussian maps
        mixture_gaussian_map = np.zeros(mask.shape)
        for gaussian_map in gaussian_map_list:
            mixture_gaussian_map += gaussian_map
        # calculate the weight of each possible cropping area
        min_weight = float('inf')
        best_top_left = None
        # max_weight = -float('inf')
        # crop_weights = []
        # crop_positions = []
        for top in range(margin_h+1):
            for left in range(margin_w+1):
                weight = np.sum(mixture_gaussian_map[top:top+crop_h, left:left+crop_w])
                if weight < min_weight:
                    min_weight = weight
                    best_top_left = (top, left)
                # crop_weights.append(weight)
                # crop_positions.append((top, left))
        
        # # normalize the weights to get the probabilities
        # crop_weights = np.array(crop_weights)
        # probabilities = crop_weights / np.sum(crop_weights)

        # invert the weights to make higher weights less probable
        # max_weight = np.max(crop_weights)
        # inverted_weights = max_weight - crop_weights
        
        # # normalize the inverted weights to get probabilities
        # inverted_weights = np.array(inverted_weights)
        # probabilities = inverted_weights / np.sum(inverted_weights)

        # select a crop position based on the probabilities
        # selected_idx = np.random.choice(len(crop_positions), p=probabilities)
        # best_top_left = crop_positions[selected_idx]

        # crop the image and mask
        image = image[best_top_left[0]:best_top_left[0]+crop_h, best_top_left[1]:best_top_left[1]+crop_w]
        mask = mask[best_top_left[0]:best_top_left[0]+crop_h, best_top_left[1]:best_top_left[1]+crop_w]
                
        return image, mask

    def cutout(self, img, n_holes, length):
        """
        Args:
            img (tensor): numpy
            n_holes (int): 要剪切的洞的数量
            length (int): 每个洞的边长
        Returns:
            img: 增强后的图像。
        """
        h, w, C = img.shape
        mask = np.ones((h, w), np.float32)
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        # mask = torch.from_numpy(mask)
        # mask_img = mask.expand_as(img)
        # mask_label = mask.expand_as(label)
        mask = np.expand_dims(mask, axis=-1)
        mask_img = np.repeat(mask, C, axis=-1)
        # if random.random() < 0.5:
        img = img * mask_img

        return img #, label.long()

    def cutout_context(self, img, mask, n_holes, length):
        
        img_h, img_w = mask.shape
        crop_h, crop_w = length, length
        margin_h = max(img_h - crop_h, 0)
        margin_w = max(img_w - crop_w, 0)
        labels = np.unique(mask)
        labels = labels[labels != self.ignore_label]
        labels = labels[labels != 0]
        labeled_objects = {}
        for label_id in labels:
            binary_mask = (mask == label_id).astype(np.uint8) #.float()
            labeled_array, num_features = label(binary_mask)
            bboxes = find_objects(labeled_array)
            labeled_objects[label_id] = bboxes
        all_bboxes = []
        for bboxes in labeled_objects.values():
            all_bboxes.extend(bboxes)

        # Collect all centers of bounding boxes
        centers = []
        for bbox in all_bboxes:
            if bbox is not None:
                y_slice, x_slice = bbox
                center_y = (y_slice.start + y_slice.stop) // 2
                center_x = (x_slice.start + x_slice.stop) // 2
                centers.append((center_y, center_x))
        
        if not centers:
            raise ValueError("No valid bounding boxes found.")
        
        # Randomly select a center from the list
        center_y, center_x = random.choice(centers)
        
        # Compute the cutout region boundaries
        start_y = max(center_y - crop_h // 2, 0)
        start_x = max(center_x - crop_w // 2, 0)
        end_y = min(start_y + crop_h, img_h)
        end_x = min(start_x + crop_w, img_w)
        
        # Adjust start positions if the end positions exceed image boundaries
        if end_y - start_y < crop_h:
            start_y = max(end_y - crop_h, 0)
        if end_x - start_x < crop_w:
            start_x = max(end_x - crop_w, 0)
        
        '''if not all_bboxes:
            raise ValueError("No valid bounding boxes found.")
        
        # Randomly select one bounding box
        bbox = random.choice(all_bboxes)
        if bbox is None:
            raise ValueError("Selected bounding box is invalid.")
        
        y_slice, x_slice = bbox
        bbox_height = y_slice.stop - y_slice.start
        bbox_width = x_slice.stop - x_slice.start
        
        # Ensure the cutout size fits within the bounding box
        cutout_h = min(crop_h, bbox_height)
        cutout_w = min(crop_w, bbox_width)
        
        # Randomly choose the center of the cutout within the bounding box
        center_y = random.randint(y_slice.start + cutout_h // 2, y_slice.stop - cutout_h // 2 - 1)
        center_x = random.randint(x_slice.start + cutout_w // 2, x_slice.stop - cutout_w // 2 - 1)
        
        # Compute the cutout region boundaries
        start_y = max(center_y - cutout_h // 2, 0)
        start_x = max(center_x - cutout_w // 2, 0)
        end_y = min(start_y + cutout_h, img_h)
        end_x = min(start_x + cutout_w, img_w)
        
        # Adjust start positions if the end positions exceed image boundaries
        if end_y - start_y < cutout_h:
            start_y = max(end_y - cutout_h, 0)
        if end_x - start_x < cutout_w:
            start_x = max(end_x - cutout_w, 0)'''
        
        '''if not all_bboxes:
            raise ValueError("No valid bounding boxes found.")
        
        # Calculate the merged bounding box that covers all the bboxes
        min_y = min([bbox[0].start for bbox in all_bboxes if bbox is not None])
        max_y = max([bbox[0].stop for bbox in all_bboxes if bbox is not None])
        min_x = min([bbox[1].start for bbox in all_bboxes if bbox is not None])
        max_x = max([bbox[1].stop for bbox in all_bboxes if bbox is not None])
        
        # Ensure the bounding box does not exceed image boundaries
        min_y = max(min_y - crop_h, 0)
        max_y = min(max_y + crop_h, img_h)
        min_x = max(min_x - crop_w, 0)
        max_x = min(max_x + crop_w, img_w)
        
        # Define the valid region by excluding the area covered by the bboxes
        valid_regions = []
        if min_y > 0:
            valid_regions.append((0, min_y, img_w, min_y + crop_h))
        if max_y < img_h:
            valid_regions.append((0, max_y, img_w, max_y + crop_h))
        if min_x > 0:
            valid_regions.append((min_x, 0, min_x + crop_w, img_h))
        if max_x < img_w:
            valid_regions.append((max_x, 0, max_x + crop_w, img_h))
        
        # If no valid region is found, raise an error
        if not valid_regions:
            raise ValueError("No valid regions found outside the bounding boxes.")
        
        # Randomly select a valid region
        selected_region = random.choice(valid_regions)
        start_x, start_y, end_x, end_y = selected_region
        
        # Ensure the selected region is within image boundaries
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        end_x = min(end_x, img_w)
        end_y = min(end_y, img_h)
        
        # Ensure the cutout size is correct
        cutout_h = min(crop_h, end_y - start_y)
        cutout_w = min(crop_w, end_x - start_x)
        
        # Randomly choose the center of the cutout within the valid region
        center_y = random.randint(start_y + cutout_h // 2, end_y - cutout_h // 2 - 1)
        center_x = random.randint(start_x + cutout_w // 2, end_x - cutout_w // 2 - 1)
        
        # Compute the cutout region boundaries
        start_y = max(center_y - cutout_h // 2, 0)
        start_x = max(center_x - cutout_w // 2, 0)
        end_y = min(start_y + cutout_h, img_h)
        end_x = min(start_x + cutout_w, img_w)'''
        
        
        # Cutout the region from the image
        img[start_y:end_y, start_x:end_x] = 0.
        
        return img
        
        '''gaussian_map_list = []
        for bbox in all_bboxes:
            center = [(bbox[0].start + bbox[0].stop) / 2, (bbox[1].start + bbox[1].stop) / 2]
            sigma = np.mean([(bbox[0].stop - bbox[0].start) / 2, (bbox[1].stop - bbox[1].start) / 2])
            gaussian_map = self.gaussian_distribution(center, sigma, mask.shape)
            gaussian_map_list.append(gaussian_map)
        
        # get a mixture of gaussian maps
        mixture_gaussian_map = np.zeros(mask.shape)
        for gaussian_map in gaussian_map_list:
            mixture_gaussian_map += gaussian_map

        # Normalize the gaussian map to [0, 1]
        mixture_gaussian_map = (mixture_gaussian_map - mixture_gaussian_map.min()) / (mixture_gaussian_map.max() - mixture_gaussian_map.min())

        # Define the weight range for edge areas, e.g., between 0.3 and 0.7
        lower_bound = 0.3
        upper_bound = 0.7

        # find the area of croppping (min area)
        min_weight = float('inf')
        max_weight = -float('inf')
        min_diff = float('inf')
        best_max = None
        best_min = None
        # cutout_weights = []
        # cutout_positions = []
        for top in range(margin_h+1):
            for left in range(margin_w+1):
                # weight = np.sum(mixture_gaussian_map[top:top+crop_h, left:left+crop_w])
                crop_region = mixture_gaussian_map[top:top + crop_h, left:left + crop_w]
                mean_weight = np.mean(crop_region)
                if lower_bound <= mean_weight <= upper_bound:
                    diff = abs(mean_weight - (lower_bound + upper_bound) / 2)
                    if diff < min_diff:
                        min_diff = diff
                        best_max = (top, left)
                # if weight > max_weight:
                #     max_weight = weight
                #     best_max = (top, left)
                # if weight < min_weight:
                #     min_weight = weight
                #     best_min = (top, left)
        #         cutout_weights.append(weight)
        #         cutout_positions.append((top, left))
        # cutout_weights = np.array(cutout_weights)
        # probabilities = cutout_weights / np.sum(cutout_weights)
        # selected_idx = np.random.choice(len(cutout_positions), p=probabilities)
        # best_top_left = cutout_positions[selected_idx]
        if best_max is not None:
            img[best_max[0]:best_max[0]+crop_h, best_max[1]:best_max[1]+crop_w] = 0.
        else:
            for n in range(n_holes):
                y = np.random.randint(img_h)
                x = np.random.randint(img_w)

                y1 = np.clip(y - length // 2, 0, img_h)
                y2 = np.clip(y + length // 2, 0, img_h)
                x1 = np.clip(x - length // 2, 0, img_w)
                x2 = np.clip(x + length // 2, 0, img_w)

                img[y1: y2, x1: x2] = 0.'''

                
        # return img