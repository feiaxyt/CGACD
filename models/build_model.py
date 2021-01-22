# Copyright (c) SenseTime. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import pdb
import numpy as np
from .neck import get_neck
from .backbone import get_backbone
from .siamese import get_siamese
from .cornerdet import get_cornerdet
from .attention import get_attention
from os.path import join
from utils.utils import smooth_l1_loss
from .PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from config.config import cfg


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.backbone = get_backbone(cfg.backbone.type, **cfg.backbone.kwargs)

        if cfg.adjust.adjust:
            self.neck = get_neck(cfg.adjust.type, **cfg.adjust.kwargs)

        self.siamese = get_siamese(cfg.siamese.type, **cfg.siamese.kwargs)
        
        if cfg.attention.attention:
            self.attention = get_attention(cfg.attention.type, **cfg.attention.kwargs)
        
        if cfg.cornerdet.cornerdet:
            self.cornerdet = get_cornerdet(cfg.cornerdet.type, **cfg.cornerdet.kwargs)
            self.prpool = PrRoIPool2D(cfg.train.search_pool_size, cfg.train.search_pool_size, 1.0/cfg.train.stride)
            self.prpoolzf = PrRoIPool2D(cfg.train.template_pool_size, cfg.train.template_pool_size, 1.0/cfg.train.stride)

    def forward(self, sample, **kwargs):
        template = sample['template'].cuda()
        search = sample['search'].cuda()
        cls_label = sample['cls_label'].cuda()
        bbox_label = sample['bbox_label'].cuda()
        perform_corner = kwargs['perform_corner']
        
        z_f = self.backbone(template)
        x_f = self.backbone(search)
        if cfg.adjust.adjust:
            z_f = self.neck(z_f)
            x_f = self.neck(x_f)
        
        regress, score = self.siamese(z_f, x_f)

        batch_sz = regress.shape[0]
        rpn_labels = cls_label[:, :, :, 0]  # [b, w, h]
        rpn_cls_weight = cls_label[:, :, :, 1]
        rpn_labels = rpn_labels.reshape(batch_sz, -1)  # [b, w*h]
        rpn_cls_weight = rpn_cls_weight.reshape(batch_sz, -1)
        rpn_bbox_targets = bbox_label.reshape(batch_sz, -1, 4)
        rpn_select = (rpn_labels != -1)
        rpn_select_bbox = (rpn_labels == 1)

        rpn_cls_score_shaped = score.reshape(batch_sz, -1)
        rpn_cls_score = rpn_cls_score_shaped[rpn_select].reshape(-1)
        rpn_labels = rpn_labels[rpn_select].reshape(-1)
        rpn_cls_weight = rpn_cls_weight[rpn_select].reshape(-1)
        rpn_cross_entropy = F.binary_cross_entropy_with_logits(rpn_cls_score, rpn_labels,
                                                               weight=rpn_cls_weight, reduction='sum') / batch_sz

        bbox_data = regress.reshape(batch_sz, 4, -1).permute(0, 2, 1)
        bbox_data_anchor = bbox_data.data.clone()
        bbox_data = bbox_data[rpn_select_bbox].reshape(-1, 4)
        rpn_bbox_targets = rpn_bbox_targets[rpn_select_bbox].reshape(-1, 4)
        rpn_loss_box = smooth_l1_loss(bbox_data, rpn_bbox_targets, cfg.train.breg_sigma)#l1_loss(bbox_data, rpn_bbox_targets)

        outputs = {}
        outputs['rpn_cross_entropy'] = rpn_cross_entropy
        outputs['rpn_loss_box'] = rpn_loss_box

        outputs['loss'] = rpn_cross_entropy * cfg.train.cls_weight + rpn_loss_box * cfg.train.breg_weight

        if perform_corner and cfg.cornerdet.cornerdet:
            bbox_xyxy = sample['bbox_xyxy'].cuda()
            template_bbox = sample['template_bbox'].cuda()
            stride = cfg.train.stride
            norm_value = stride * stride

            response_size = cfg.train.response_size
            ori = - (response_size // 2)
            xx, yy = np.meshgrid([ori + dx for dx in range(response_size)],
                                 [ori + dy for dy in range(response_size)])
            response_xx = torch.from_numpy(xx).float().reshape(-1).cuda()
            response_yy = torch.from_numpy(yy).float().reshape(-1).cuda()

            bbox_data_anchor[..., 0] = response_xx * stride - bbox_data_anchor[..., 0] * norm_value + search.shape[-1]//2
            bbox_data_anchor[..., 1] = response_yy * stride - bbox_data_anchor[..., 1] * norm_value + search.shape[-1]//2
            bbox_data_anchor[..., 2] = response_xx * stride + bbox_data_anchor[..., 2] * norm_value + search.shape[-1]//2
            bbox_data_anchor[..., 3] = response_yy * stride + bbox_data_anchor[..., 3] * norm_value + search.shape[-1]//2

            training_box = bbox_data_anchor[rpn_select_bbox]

            wp = training_box[:, 2] - training_box[:, 0]
            hp = training_box[:, 3] - training_box[:, 1]
            
            positive_select = (wp > 0) * (hp > 0)
            
            training_box = training_box[positive_select, :]
            wp = wp[positive_select]
            hp = hp[positive_select]

            if cfg.train.roi_augmentation.ratio:
                ratio = cfg.train.roi_augmentation.ratio > torch.rand_like(wp)
                scale_w = 1.0 + (torch.rand_like(wp[ratio]) * 2 - 1) * cfg.train.roi_augmentation.scale
                scale_h = 1.0 + (torch.rand_like(hp[ratio]) * 2 - 1) * cfg.train.roi_augmentation.scale
                wp[ratio] = scale_w * wp[ratio]
                hp[ratio] = scale_h * hp[ratio]

            wc_z = wp + 0.5 * (wp + hp)
            hc_z = hp + 0.5 * (wp + hp)
            s_z = torch.sqrt(wc_z * hc_z)

            center_x = (training_box[..., 0] + training_box[..., 2]) / 2
            center_y = (training_box[..., 1] + training_box[..., 3]) / 2
            
            if cfg.train.roi_augmentation.ratio:
                max_x = (s_z[ratio] - wp[ratio]) / 2
                max_y = (s_z[ratio] - hp[ratio]) / 2
                shift_x = (torch.rand_like(center_x[ratio]) * 2 - 1) * cfg.train.roi_augmentation.shift
                shift_y = (torch.rand_like(center_y[ratio]) * 2 - 1) * cfg.train.roi_augmentation.shift
                center_x[ratio] = torch.min(shift_x, max_x) + center_x[ratio]
                center_y[ratio] = torch.min(shift_y, max_y) + center_y[ratio]

            training_box[..., 0] = center_x - s_z / 2
            training_box[..., 1] = center_y - s_z / 2
            training_box[..., 2] = center_x + s_z / 2 + 1
            training_box[..., 3] = center_y + s_z / 2 + 1

            training_box_size = rpn_select_bbox.sum(1)
            index = []
            for t, ts in enumerate(training_box_size):
                for js in range(ts):
                    index.append(t)
            index = torch.tensor(index)[positive_select]
            index_cat = index.float().reshape(-1, 1).cuda()

            training_box_cat = torch.cat((index_cat, training_box), 1)

            train_template_bbox = template_bbox[index, ...]
            train_template_bbox[...,2:4] = train_template_bbox[...,2:4]+1
            template_box_cat = torch.cat((index_cat, train_template_bbox), 1)

            training_f = self.prpool(x_f, training_box_cat)
            zf_train = self.prpoolzf(z_f, template_box_cat)

            if cfg.attention.attention:
                feat_for_cornerdet = self.attention(zf_train, training_f)
            else:
                feat_for_cornerdet = (training_f, training_f)
            predict_target, heatmap_size = self.cornerdet(feat_for_cornerdet)

            predict_target = predict_target * heatmap_size

            w0 = heatmap_size
            h0 = heatmap_size

            train_bbox_xyxy = bbox_xyxy[index, :]

            target_label = torch.zeros_like(predict_target)

            training_box[..., 2] = training_box[..., 2] - 1
            training_box[..., 3] = training_box[..., 3] - 1

            wp = training_box[:, 2] - training_box[:, 0]
            hp = training_box[:, 3] - training_box[:, 1]

            target_label[..., 0] = (train_bbox_xyxy[:, 0] - training_box[:, 0]) / wp * w0
            target_label[..., 1] = (train_bbox_xyxy[:, 1] - training_box[:, 1]) / hp * h0
            target_label[..., 2] = (train_bbox_xyxy[:, 2] - training_box[:, 0]) / wp * w0
            target_label[..., 3] = (train_bbox_xyxy[:, 3] - training_box[:, 1]) / hp * h0
            loss1 = torch.mean(torch.abs(predict_target - target_label))
            loss2 = torch.mean((predict_target - target_label) ** 2)
            if math.isnan(loss1):
                pdb.set_trace()
            outputs['loss1'] = loss1
            outputs['loss2'] = loss2
            outputs['loss'] += (loss1 + loss2) * cfg.train.creg_weight

        return outputs

    def template(self, z):
        zf = self.backbone(z)
        if cfg.adjust.adjust:
            zf = self.neck(zf)
        self.zf = zf
        response_size = cfg.track.response_size
        ori = - (response_size // 2)
        xx, yy = np.meshgrid([ori + dx for dx in range(int(response_size))],
                             [ori + dy for dy in range(int(response_size))])
        self.xx = xx.reshape(-1)
        self.yy = yy.reshape(-1)
    
    def track(self, x, target_sz, template_bbox, window):
        xf = self.backbone(x)
        if cfg.adjust.adjust:
            xf = self.neck(xf)
        delta, score = self.siamese(self.zf, xf)

        delta = delta.squeeze().view(4, -1).data.cpu().numpy()
        score = torch.sigmoid(score.squeeze().view(-1)).data.cpu().numpy()

        temp_delta = np.zeros_like(delta)

        stride = cfg.track.stride
        norm_value = stride * stride

        temp_delta[0, :] = stride * self.xx - delta[0, :] * norm_value
        temp_delta[1, :] = stride * self.yy - delta[1, :] * norm_value
        temp_delta[2, :] = stride * self.xx + delta[2, :] * norm_value
        temp_delta[3, :] = stride * self.yy + delta[3, :] * norm_value

        delta[0, :] = (temp_delta[0, :] + temp_delta[2, :]) / 2
        delta[1, :] = (temp_delta[1, :] + temp_delta[3, :]) / 2
        delta[2, :] = np.maximum(temp_delta[2, :] - temp_delta[0, :], 1e-8)
        delta[3, :] = np.maximum(temp_delta[3, :] - temp_delta[1, :], 1e-8)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1.) * cfg.track.penalty_k)
        pscore = penalty * score

        # window float
        pscore = pscore * (1 - cfg.track.window_influence) + window * cfg.track.window_influence
        best_pscore_id = np.argmax(pscore)

        target = delta[:, best_pscore_id]

        if cfg.cornerdet.cornerdet:
            best_box = temp_delta[:, best_pscore_id] + cfg.track.search_size // 2
            wp = best_box[2] - best_box[0]
            hp = best_box[3] - best_box[1]
            wc_z = wp + 0.5 * (wp + hp)
            hc_z = hp + 0.5 * (wp + hp)
            s_z = np.sqrt(wc_z * hc_z)
            center_x = (best_box[0] + best_box[2]) / 2
            center_y = (best_box[1] + best_box[3]) / 2
            best_box[0] = center_x - s_z / 2
            best_box[1] = center_y - s_z / 2
            best_box[2] = center_x + s_z / 2 + 1
            best_box[3] = center_y + s_z / 2 + 1
            
            best_box_tensor = torch.from_numpy(best_box).reshape(-1, 4).cuda()
            index = torch.from_numpy(np.array([0])).reshape(-1, 1).float().cuda()
            training_box_cat = torch.cat((index, best_box_tensor), 1)
            template_bbox_tensor = torch.tensor(template_bbox).reshape(-1, 4).float().cuda()
            template_bbox_tensor[..., 2:4] =  template_bbox_tensor[..., 2:4] + 1
            template_box_cat = torch.cat((index, template_bbox_tensor), 1)

            track_f = self.prpool(xf, training_box_cat)
            zf_track = self.prpoolzf(self.zf, template_box_cat)

            if cfg.attention.attention:
                feat_for_cornerdet = self.attention(zf_track, track_f)
            else:
                feat_for_cornerdet = (track_f, track_f)
            predict_target, heatmap_size = self.cornerdet(feat_for_cornerdet)
            predict_target = predict_target.data.cpu().numpy().reshape(-1)
            predict_target = predict_target * heatmap_size
            best_box[2] = best_box[2] - 1
            best_box[3] = best_box[3] - 1
            w0 = heatmap_size
            h0 = heatmap_size
            wp = best_box[2] - best_box[0]
            hp = best_box[3] - best_box[1]
            target_xyxy = np.zeros_like(predict_target)
            target_xyxy[0] = predict_target[0] / w0 * wp + best_box[0]
            target_xyxy[1] = predict_target[1] / h0 * hp + best_box[1]
            target_xyxy[2] = predict_target[2] / w0 * wp + best_box[0]
            target_xyxy[3] = predict_target[3] / h0 * hp + best_box[1]

            target = np.zeros_like(predict_target)
            target[0] = (target_xyxy[0] + target_xyxy[2]) / 2 - cfg.track.search_size//2
            target[1] = (target_xyxy[1] + target_xyxy[3]) / 2 - cfg.track.search_size//2
            target[2] = target_xyxy[2] - target_xyxy[0]
            target[3] = target_xyxy[3] - target_xyxy[1]

        #target = delta[:, best_pscore_id]

        return target, penalty, score, best_pscore_id
        

