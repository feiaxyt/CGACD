# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, isdir
from os import listdir
import json
import numpy as np
import glob
import cv2
from pathlib import Path

base_path = '/home/feiji/Research/Data/GOT-10k'
sub_sets= sorted({'train', 'val'})

def check_neg(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 -y1
    if w <= 0 or h <= 0:
        return False
    return True

def check_size(frame_sz, bbox):
    #min_ratio = 0.1
    max_ratio = 0.75
    # only accept objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/float(np.prod(frame_sz)))
    ok =  (area_ratio < max_ratio) # and (area_ratio > min_ratio)
    return ok


def check_borders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2] - bbox[0] + bbox[3] - bbox[1])/2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and \
         ((frame_sz[0] - bbox[2]) > dist_from_border) and \
         ((frame_sz[1] - bbox[3]) > dist_from_border)
    return ok

def isinvalid(name):
    allname = 'GOT-10k_Train_008628' + 'GOT-10k_Train_008630' + 'GOT-10k_Train_009058' + \
        'GOT-10k_Train_009059' + 'GOT-10k_Train_008633' + 'GOT-10k_Train_008632' + \
        'GOT-10k_Train_008625' + 'GOT-10k_Train_008623' + 'GOT-10k_Train_008637' + \
        'GOT-10k_Train_008627' + 'GOT-10k_Train_008629' + 'GOT-10k_Train_008634' + \
        'GOT-10k_Train_008626' + 'GOT-10k_Train_005996' + 'GOT-10k_Train_004419'

    if allname.find(name) != -1:
        return True
    return False


snippets = dict()
n_snippets = 0
n_videos = 0
for subset in sub_sets:
    sub_set_base_path = join(base_path, subset)
    videos = sorted(listdir(sub_set_base_path))
    for video in videos:
        if not isdir(join(sub_set_base_path, video)):
            continue
        if isinvalid(video):
            continue
        n_videos += 1
        ground_truth_file = join(sub_set_base_path, video, 'groundtruth.txt')
        full_occlusion_file = join(sub_set_base_path, video, 'absence.label')
        #cover = join(sub_set_base_path, video, 'cover.label')
        gt = np.genfromtxt(ground_truth_file, delimiter=',', dtype=float).astype(np.int)
        fo = np.genfromtxt(full_occlusion_file,  dtype=int)
        subdir_paths = sorted(glob.glob(join(sub_set_base_path, video, '*.jpg')))
        snippets[join(subset, video)] = dict()
        snippet = dict()
        track_id = 0
        valid_num = 0
        img = cv2.imread(subdir_paths[0])
        frame_sz = [img.shape[1], img.shape[0]]
        for i, img in enumerate(subdir_paths):
            info = {}
            filename = Path(img).stem
            bbox = gt[i]
            fo_i = fo[i]
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]
            info['valid'] = 0
            if (not fo_i) and check_neg(bbox) and check_size(frame_sz, bbox) and check_borders(frame_sz, bbox):
                info['valid'] = 1
                valid_num += 1
            info['bbox'] = bbox
            snippet['{:06d}'.format(int(filename))] = info
            #snippet['track_category'] = 0
        if valid_num > 1:
            snippets[join(subset, video)]['{:02d}'.format(track_id)] = snippet
            n_snippets += 1
        print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))

json.dump(snippets, open('train_largeclean.json', 'w'), indent=4, sort_keys=True)
print('done!')
