# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join
from os import listdir
import json
import numpy as np

print('load json (raw vid info), please wait 20 seconds~')
vid = json.load(open('vid.json', 'r'))

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


snippets = dict()
n_snippets = 0
n_videos = 0
for subset in vid:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        id_set = []
        id_frames = [[]] * 60  # at most 60 objects
        for f, frame in enumerate(frames):
            objs = frame['objs']
            frame_sz = frame['frame_sz']
            for obj in objs:
                trackid = obj['trackid']
                occluded = obj['occ']
                bbox = obj['bbox']
                # if occluded:
                #     continue
                #
                obj['valid'] = 0
                if not(occluded) and check_neg(bbox) and check_size(frame_sz, bbox) and check_borders(frame_sz, bbox):
                    obj['valid'] = 1
                #
                # if obj['c'] in ['n01674464', 'n01726692', 'n04468005', 'n02062744']:
                #     continue

                if trackid not in id_set:
                    id_set.append(trackid)
                    id_frames[trackid] = []
                id_frames[trackid].append(f)
        if len(id_set) > 0:
            snippets[video['base_path']] = dict()
        for selected in id_set:
            frame_ids = sorted(id_frames[selected])
            sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
            sequences = [s for s in sequences if len(s) > 1]  # remove isolated frame.
            for seq in sequences:
                snippet = dict()
                valid_num = 0
                for frame_id in seq:
                    info = {}
                    frame = frames[frame_id]
                    for obj in frame['objs']:
                        if obj['trackid'] == selected:
                            o = obj
                            continue
                    info['bbox'] =  o['bbox']
                    info['valid'] = o['valid']
                    if o['valid'] == 1:
                        valid_num+=1
                    snippet[frame['img_path'].split('.')[0]] = info
                    #snippet['track_category'] = o['c']
                if valid_num > 0:
                    snippets[video['base_path']]['{:02d}'.format(selected)] = snippet
                    n_snippets += 1
        print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))
        
train = {k:v for (k,v) in snippets.items() if 'train' in k}
val = {k:v for (k,v) in snippets.items() if 'val' in k}

json.dump(train, open('train_largeclean.json', 'w'), indent=4, sort_keys=True)
json.dump(val, open('val_largeclean.json', 'w'), indent=4, sort_keys=True)
print('done!')
