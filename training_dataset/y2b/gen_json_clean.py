from os.path import join, isdir
from os import mkdir
import numpy as np
import cv2
import glob
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

data_file = 'train.json'
path_format = "{}.{}.{}.jpg"
root = "/ssd/feiji/Research/Data/y2b_crop511"
anno = json.load(open(data_file, 'r'))
wh_file = 'train_wh.json'
wh = json.load(open(wh_file, 'r'))
out = {}
n_videos = 0
for video, tracks in anno.items():
    new_tracks = {}
    video_id = video.split('/')[-1]
    if not(video_id in wh):
        continue
    frame_sz = wh[video_id]
    for track, frames in tracks.items():
        new_frames = {}
        valid_num = 0
        for frame, bbox in frames.items():
            new_info = {}
            image_path = join(root, video, path_format.format(frame, track, 'x'))
            new_info['bbox'] = bbox
            new_info['valid'] = 0
            if check_neg(bbox) and check_size(frame_sz, bbox) and check_borders(frame_sz, bbox):
                new_info['valid'] = 1
                valid_num += 1
            new_frames[frame] = new_info
            #new_frames['track_category'] = video.split('/')[0]
        
        if valid_num > 0:
            new_tracks[track] = new_frames
    if len(new_tracks) > 0:
        out[video] = new_tracks
        n_videos += 1
        print('video: {:d}'.format(n_videos))

json.dump(out, open('train_largeclean.json', 'w'), indent=4, sort_keys=True)

















                
