# Copyright (c) SenseTime. All Rights Reserved.

import cv2
import json
import logging
import numpy as np
import torch
import numbers
import math
import random
import pdb
from torch.utils.data import Dataset
from os.path import join
import torchvision.transforms as transforms
from config.config import cfg, cfg_from_file

logger = logging.getLogger("global")

class SubDataset(object):
    def __init__(self, config):
        for string in ['root', 'anno']:
            if string not in config:
                raise Exception('SubDataSet need "{}"'.format(string))
        logger.info("loading " + config['mark'])
        with open(config['anno']) as fin:
            meta_data = json.load(fin)
            self.labels = self._filter_zero(meta_data, config['mark'])
        def isint(x):
            try:
                int(x)
                return True
            except:
                return False

        # add frames args into labels
        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, filter(lambda x: isint(x), frames.keys())))
                frames.sort()
                valids = [self.labels[video][track]["{:06d}".format(f)]['valid'] for f in frames]
                self.labels[video][track]['frames'] = frames
                self.labels[video][track]['valids'] = valids
                if len(frames) <= 0:
                    to_del.append((video, track))

        # delete tracks with no frames
        for video, track in to_del:
            del self.labels[video][track]

        # delete videos with no valid track
        to_del = []
        for video in self.labels:
            if len(self.labels[video]) <= 0:
                to_del.append(video)

        for video in to_del:
            del self.labels[video]

        self.videos = list(self.labels.keys())
        
        self.root = "/"
        self.start = 0
        self.num = len(self.labels)
        self.num_use = self.num
        self.frame_range = 100
        self.mark = "vid"
        self.path_format = "{}.{}.{}.jpg"
        self.pick = []
        self.__dict__.update(config)
        self.num_use = int(self.num_use)

        self.shuffle()

        logger.info("{} loaded".format(self.mark))


    def _filter_zero(self, anno, name):
        out = {}
     
        for video, tracks in anno.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                valid_num = 0
                for frm, info in frames.items():
                    new_info = {}
                    bbox = info
                    valid = 1
                    if type(info) is dict:
                        bbox = info['bbox']
                        valid = info['valid']
                    new_info['bbox'] = bbox
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 -y1
                    else:
                        w, h= bbox
                    if w <= 0 or h <= 0:
                        valid = 0
                    else:
                        valid = 1
                        valid_num += 1
                    new_info['valid'] = valid
                    new_frames[frm] = new_info

                if valid_num > 0:
                    new_tracks[trk] = new_frames

            if len(new_tracks) > 0:
                out[video] = new_tracks

        return out

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.mark, self.start, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start, self.start + self.num))

        m = 0
        pick = []
        while m < self.num_use:
            np.random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = join(self.root, video, self.path_format.format(frame, track, 'x'))
        try:
            image_anno = self.labels[video][track][frame]['bbox']
        except:
            print(video + ' ' + track + ' ' + frame)
            raise Exception

        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = np.array(track_info['frames'])
        valids = np.array(track_info['valids'])
        candidate = np.where(valids == 1)
        template_frame = random.choice(candidate[0])

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        valid_range = valids[left:right]
        search_range = search_range[valid_range == 1]
        search_frame = random.choice(search_range)
        template_frame = frames[template_frame]
        
        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = random.randint(0, self.num-1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = np.array(track_info['frames'])
        valids = np.array(track_info['valids'])
        candidate = np.where(valids == 1)
        frame = random.choice(candidate[0])
        frame = frames[frame]
        return self.get_image_anno(video_name, track, frame)


class Augmentation(object):
    def __init__(self, config):
        self.shift = 0
        self.scale = 0
        self.blur = 0
        self.color = 0
        self.__dict__.update(config)
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    def crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def shift_scale_aug(self, image, bbox, size):
        shape = image.shape[:2]
        cy, cx = shape[0] // 2, shape[1] // 2
        scale_h = 1.0 + random.uniform(-self.scale, self.scale)
        scale_w = 1.0 + random.uniform(-self.scale, self.scale)
        shift_x = random.randint(-self.shift, self.shift)
        shift_y = random.randint(-self.shift, self.shift)
        crop_box = [cx + shift_x, cy + shift_y, size * scale_w, size * scale_h]
        w, h = crop_box[2], crop_box[3]
        crop_bbox = [crop_box[0]-w*0.5, crop_box[1]-h*0.5, crop_box[0]+w*0.5, crop_box[1]+h*0.5]
        image = self.crop_roi(image, crop_bbox, size)
        x, y = crop_bbox[0], crop_bbox[1]
        bbox = [bbox[0] - x, bbox[1] - y, bbox[2] - x, bbox[3] - y]
        bbox = [bbox[0] / scale_w, bbox[1] / scale_h, bbox[2] / scale_w, bbox[3] / scale_h]
        return image, bbox


    def blur_image(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel

        kernel = rand_kernel()

        image = cv2.filter2D(image, -1, kernel)
        return image
    
    def __call__(self, image, bbox, size, gray=False):
        if gray:
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)

        image, bbox  = self.shift_scale_aug(image, bbox, size)

        if self.color > random.random():
            offset = np.dot(self.rgbVar, np.random.randn(3, 1))
            offset = offset[::-1]  # bgr 2 rgb
            offset = offset.reshape(3)
            image = image - offset

        if self.blur > random.random():
            image = self.blur_image(image)

        return image, bbox
        

class Datasets(Dataset):
    def __init__(self, is_train=True):
        self.template_size = cfg.train.template_size
        self.search_size = cfg.train.search_size
        self.response_size = cfg.train.response_size
        self.stride = cfg.train.stride
        self.label_sigma1 = cfg.train.label_sigma1
        self.label_sigma2 = cfg.train.label_sigma2
        self.all_data = []
        start = 0
        self.num = 0
        if is_train:
            dataset_dict = cfg.train.train_dataset
        else:
            dataset_dict = cfg.train.val_dataset
        
        for name in dataset_dict.names:
            dataset = dataset_dict[name]
            dataset['mark'] = name
            dataset['start'] = start
            dataset = SubDataset(dataset)
            self.all_data.append(dataset)

            start += dataset.num # real video number
            self.num += dataset.num_use # the number used for subset shuffle

            dataset.log()
        
        aug_cfg = dataset_dict.augmentation
        self.template_aug = Augmentation(aug_cfg['template'])
        self.search_aug = Augmentation(aug_cfg['search'])
        self.gray = aug_cfg['gray']
        self.neg = aug_cfg['neg']
        self.norm = 0 if 'norm' not in aug_cfg else aug_cfg['norm']
        if self.norm:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        self.inner_neg = 0 if 'inner_neg' not in aug_cfg else aug_cfg['inner_neg']

        self.pick = None  # list to save id for each img
        self.pick_epoch = None # list to each epoch
        self.last_epoch = 0
        if 'video_per_epoch' in dataset_dict:  # number used in training for all dataset
            self.num = int(dataset_dict['video_per_epoch'])
        self.num_per_epoch = self.num
        #self.num *= num_epoch
        self.shuffle()

    def net_epoch(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        self.pick_epoch = self.pick[self.last_epoch * self.num_per_epoch: (self.last_epoch+1)*self.num_per_epoch]
        self.last_epoch += 1

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.all_data:
                sub_p = subset.shuffle()
                p += sub_p

            np.random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick[:self.num]
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))

    def __len__(self):
        return self.num_per_epoch

    def find_dataset(self, index):
        for dataset in self.all_data:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self.find_dataset(index)
        neg = self.neg and self.neg > random.random()
        gray = self.gray and self.gray > random.random()
        if neg:
            template = dataset.get_random_target(index)
            if self.inner_neg and self.inner_neg > random.random():
                search = dataset.get_random_target()
            else:
                search = random.choice(self.all_data).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)
        #import pdb
        #pdb.set_trace()
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        def toBBox(image, shape):
            imh, imw = image.shape[:2]
            if len(shape) == 4:
                w, h = shape[2]-shape[0], shape[3]-shape[1]
            else:
                w, h = shape
            context_amount = 0.5
            exemplar_size = self.template_size  # 127
            wc_z = w + context_amount * (w+h)
            hc_z = h + context_amount * (w+h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w, h = w * scale_z, h * scale_z
            cx, cy = imw//2, imh//2
            bbox = [cx - w*0.5, cy - h*0.5, cx + w*0.5, cy + h*0.5]
            return bbox
        try:
            template_bbox = toBBox(template_image, template[1])
            search_bbox = toBBox(search_image, search[1])
        except:
            print(template[0])
            print(search[0])
            raise Exception

        template_image, template_target_box = self.template_aug(template_image, template_bbox, self.template_size, gray=gray)
        search_image, target_box = self.search_aug(search_image, search_bbox, self.search_size, gray=gray)
        target_box, template_target_box = np.array(target_box), np.array(template_target_box)
        cls_label, bbox_label = self.construct_label(target_box - self.search_size//2, neg)

        template_image, search_image = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template_image, search_image])
        if self.norm:
            template_image = self.normalize(template_image)
            search_image = self.normalize(search_image)
        return {
                    'template': template_image,
                    'search': search_image,
                    'cls_label': cls_label,
                    'bbox_label': bbox_label,
                    'bbox_xyxy': target_box.astype(np.float32),
                    'template_bbox': template_target_box.astype(np.float32)
                }

    def construct_label(self, target_box, is_negative):
        sigma1 = self.label_sigma1
        sigma2 = self.label_sigma2
        response_sz = self.response_size
        ratio = self.stride
        xp = target_box / ratio
        w, h = xp[2] - xp[0], xp[3] - xp[1]
        center = [xp[0] + 0.5 * w, xp[1] + 0.5 * h]
        Rpos = np.array([center[0] - 0.5 * w * sigma1, 
                center[1] - 0.5 * h * sigma1, 
                center[0] + 0.5 * w * sigma1, 
                center[1] + 0.5 * h * sigma1
                ]).astype(np.int) + response_sz // 2
        
        Npos = np.array([center[0] - 0.5 * w * sigma2, 
                center[1] - 0.5 * h * sigma2, 
                center[0] + 0.5 * w * sigma2, 
                center[1] + 0.5 * h * sigma2
                ]).astype(np.int) + response_sz // 2
        
        labels = np.zeros((response_sz, response_sz))
        if not is_negative:
            labels[max(Npos[1], 0):min(Npos[3]+1, response_sz), max(Npos[0], 0):min(Npos[2]+1, response_sz)] = -1
            labels[max(Rpos[1], 0):min(Rpos[3]+1, response_sz), max(Rpos[0], 0):min(Rpos[2]+1, response_sz)] = 1
        pos_map = (labels == 1)
        neg_map = (labels == 0)
        pos_pixels = np.sum(pos_map, axis=(0,1))
        neg_pixels = np.sum(neg_map, axis=(0,1))
        cls_label = np.zeros([response_sz, response_sz, 2],np.float32)
        cls_label[:, :, 0] = labels
        if is_negative:
            cls_label[:, :, 1] = 0.5 * neg_map / neg_pixels
        else:
            cls_label[:, :, 1] = 0.5 * pos_map / pos_pixels + 0.5 * neg_map / neg_pixels

        bbox_label = np.zeros((response_sz, response_sz, 4), np.float32)
        ori = - (response_sz // 2)
        xx, yy = np.meshgrid([ori + dx for dx in range(response_sz)],
                         [ori + dy for dy in range(response_sz)])

        norm_value = ratio * ratio
        bbox_label[:,:,0] = (xx * ratio - target_box[0]) / norm_value
        bbox_label[:,:,1] = (yy * ratio - target_box[1]) / norm_value
        bbox_label[:,:,2] = (target_box[2] - xx * ratio) / norm_value
        bbox_label[:,:,3] = (target_box[3] - yy * ratio) / norm_value

        return cls_label, bbox_label
