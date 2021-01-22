# Copyright (c) SenseTime. All Rights Reserved.

import argparse
import os

import cv2
import torch
import numpy as np

from config.config import cfg, cfg_from_file
from models.build_model import ModelBuilder
from utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, cxy_wh_2_rect1,rect1_2_cxy_wh, read_image
from track.run_CGACD import CGACD_init, CGACD_track
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='CGACD tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--model', default='', type=str,
        help='models to eval')
parser.add_argument('--save_name', metavar='PATH', default='test')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg_from_file(args.config)

    dataset_root = os.path.join('dataset', args.dataset)

    # create model
    net = ModelBuilder()
    checkpoint = torch.load(args.model)
    if 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.cuda().eval()
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.save_name
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                    state = CGACD_init(img, target_pos, target_sz, net)
                    pred_bbox = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    state = CGACD_track(state, img)
                    pred_bbox = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    pred_polygon = [pred_bbox[0], pred_bbox[1],
                                    pred_bbox[0] + pred_bbox[2], pred_bbox[1],
                                    pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3],
                                    pred_bbox[0], pred_bbox[1] + pred_bbox[3]]
                    overlap = vot_overlap(gt_bbox, pred_polygon, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    target_pos = state['target_pos']
                    target_sz = state['target_sz']
                    cv2.rectangle(img, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                                  (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                                  (0, 255, 0), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(video.name, img)
                    cv2.moveWindow(video.name, 100, 100)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('result', args.dataset, model_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    if 'OTB' in args.dataset:
                        target_pos, target_sz = rect1_2_cxy_wh(gt_bbox)
                    else:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                    state = CGACD_init(img, target_pos, target_sz, net)
                    if 'OTB' in args.dataset:
                        pred_bbox = cxy_wh_2_rect1(state['target_pos'], state['target_sz'])
                    else:
                        pred_bbox = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    pred_bboxes.append(pred_bbox)
                else:
                    state = CGACD_track(state, img)
                    pred_bbox = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    pred_bboxes.append(pred_bbox)
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    target_pos = state['target_pos']
                    target_sz = state['target_sz']
                    cv2.rectangle(img, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                                  (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                                  (0, 255, 0), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(video.name, img)
                    cv2.moveWindow(video.name, 100, 100)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
            toc /= cv2.getTickFrequency()
            if 'GOT-10k' == args.dataset:
                video_path = os.path.join('result', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('result', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
