# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from pycocotools.coco import COCO
from os.path import join
import numpy as np
import json

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


dataDir = '/home/feiji/Research/Data/COCO'
for data_subset in ['val2017', 'train2017']:
    dataset = dict()
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, data_subset)
    coco = COCO(annFile)
    n_imgs = len(coco.imgs)
    for n, img_id in enumerate(coco.imgs):
        print('subset: {} image id: {:04d} / {:04d}'.format(data_subset, n, n_imgs))
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        crop_base_path = join(data_subset, img['file_name'].split('/')[-1].split('.')[0])
        frame_sz = [img['width'], img['height']]

        for track_id, ann in enumerate(anns):
            info = {}
            rect = ann['bbox']
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            bbox = [rect[0], rect[1], rect[0]+rect[2]-1, rect[1]+rect[3]-1]  # x1,y1,x2,y2
            if check_neg(bbox) and check_size(frame_sz, bbox) and check_borders(frame_sz, bbox):
                if crop_base_path not in dataset:
                    dataset[crop_base_path] = dict()
                info['valid'] = 1
                info['bbox'] = bbox
                dataset[crop_base_path]['{:02d}'.format(track_id)] = {'000000': info}
                #dataset[crop_base_path]['{:02d}'.format(track_id)]['track_category'] = ann['category_id']

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open('{}_largeclean.json'.format(data_subset), 'w'), indent=4, sort_keys=True)
    print('done!')

