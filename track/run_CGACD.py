import numpy as np
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
import pdb
from utils.utils import get_subwindow_tracking
from config.config import cfg

def tracker_eval(net, x_crop, target_pos, template_bbox, target_sz, window, scale_z):
    target, penalty, score, best_pscore_id = net.track(x_crop, target_sz, template_bbox, window)
    #pdb.set_trace()
    target = target / scale_z
    #import pdb
    #pdb.set_trace()
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * cfg.track.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def CGACD_init(im, target_pos, target_sz, net):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + cfg.track.contex_amount * sum(target_sz)
    hc_z = target_sz[1] + cfg.track.contex_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, cfg.track.template_size, s_z, avg_chans)

    scale_z = cfg.track.template_size / s_z
    w, h = target_sz[0] * scale_z, target_sz[1] * scale_z
    cx, cy = cfg.track.template_size//2, cfg.track.template_size//2
    template_bbox = [cx - w*0.5, cy - h*0.5, cx + w*0.5, cy + h*0.5]

    z = torch.from_numpy(np.transpose(z_crop, (2, 0, 1))).float().unsqueeze(0)
    net.template(z.cuda())

    if cfg.track.windowing == 'cosine':
        window = np.outer(np.hanning(cfg.track.response_size), np.hanning(cfg.track.response_size))
    elif cfg.track.windowing == 'uniform':
        window = np.ones((cfg.track.response_size, cfg.track.response_size))
    window = window.flatten()

    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['template_bbox'] = template_bbox
    return state


def CGACD_track(state, im):
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    template_bbox = state['template_bbox']
    wc_z = target_sz[1] + cfg.track.contex_amount * sum(target_sz)
    hc_z = target_sz[0] + cfg.track.contex_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = cfg.track.template_size / s_z
    s_x = s_z * (cfg.track.search_size / cfg.track.template_size)

    # extract scaled crops for search region x at previous target position
    x_crop = get_subwindow_tracking(im, target_pos, cfg.track.search_size, round(s_x), avg_chans)

    x_crop = torch.from_numpy(np.transpose(x_crop, (2, 0, 1))).float().unsqueeze(0)

    target_pos, target_sz, best_score = tracker_eval(net, x_crop.cuda(), target_pos, template_bbox, target_sz * scale_z, window, scale_z)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['best_score'] = best_score
    return state
