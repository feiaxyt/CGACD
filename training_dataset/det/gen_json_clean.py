# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, isdir
from os import mkdir
import glob
import numpy as np
import xml.etree.ElementTree as ET
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

js = {}
VID_base_path = '/home/feiji/Research/Data/ILSVRC2015'
ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
sub_sets = ('ILSVRC2013_train', 'ILSVRC2014_train_0000', 'ILSVRC2014_train_0001','ILSVRC2014_train_0002','ILSVRC2014_train_0003','ILSVRC2014_train_0004','ILSVRC2014_train_0005','ILSVRC2014_train_0006', 'val')
for sub_set in sub_sets:
    sub_set_base_path = join(ann_base_path, sub_set)

    if 'ILSVRC2013_train' == sub_set:
        xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
    else:
        xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))
    n_imgs = len(xmls)
    for f, xml in enumerate(xmls):
        print('subset: {} frame id: {:08d} / {:08d}'.format(sub_set, f, n_imgs))
        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')
        size = xmltree.find('size')
        video = join(sub_set, xml.split('/')[-1].split('.')[0])

        for id, object_iter in enumerate(objects):
            info = {}
            bndbox = object_iter.find('bndbox')
            frame_sz = [int(size.find('width').text), int(size.find('height').text)]
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            if check_neg(bbox) and check_size(frame_sz, bbox) and check_borders(frame_sz, bbox):
                info['valid'] = 1
                info['bbox'] = bbox
                frame = '%06d' % (0)
                obj = '%02d' % (id)
                if video not in js:
                    js[video] = {}
                if obj not in js[video]:
                    js[video][obj] = {}
                js[video][obj][frame] = info
                #js[video][obj]['track_category'] = str(object_iter.find('name').text)

json.dump(js, open('train_largeclean.json', 'w'), indent=4, sort_keys=True)


