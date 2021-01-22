# Copyright (c) SenseTime. All Rights Reserved.

import argparse
import logging
from os.path import join, isdir, isfile, exists
from os import makedirs
import random
import numpy as np
import torch
import pdb
import os
from torch.nn.utils import clip_grad_norm_
from utils.log_helper import add_file_handler, init_log
from models.build_model import ModelBuilder
from utils.lr_helper import build_lr_scheduler
from utils.misc import describe, commit
from utils.model_load import load_pretrain, restore_from
import torch.backends.cudnn as cudnn
from dataset import Datasets
from easydict import EasyDict as edict
import torch.nn as nn
import torch.nn.functional as F
import time
import shutil
import json
import math
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config.config import cfg, cfg_from_file

logger = logging.getLogger('global')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('-c', '--config', required=True,
                        help='hyperparameter in json format')
    parser.add_argument('-r', '--resume_file', default=None,
                        help="Optional, name of file to restore from")
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help="batch size")
    parser.add_argument('-l', '--log', default=False,
                        help="write log file or not")
    parser.add_argument('-t', '--save_name', default='train',
                        help="name of save path")
    parser.add_argument('-s', '--save_path', default='checkpoint',
                        help="path to directory where checkpoint stored")
    parser.add_argument("-j", "--num_workers", dest="num_workers", type=int,
                        default=0, help="The number of workers for the dataloaders"
                                        " i.e. the number of additional"
                                        " dedicated threads to dataloading.")
    args = parser.parse_args()
    return args


def build_opt_lr(model, epoch):
    trainable_params = []
    for param in model.backbone.parameters():
        param.requires_grad = False
    for i in range(len(cfg.backbone.unfix_steps)):
        if epoch >= cfg.backbone.unfix_steps[i]:
            layer = getattr(model.backbone, cfg.backbone.unfix_layers[i])
            for param in layer.parameters():
                param.requires_grad = True
            trainable_params += [{'params': layer.parameters(),
                                  'lr': cfg.backbone.unfix_lr[i]}]

    if cfg.adjust.adjust:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': 1}]
    trainable_params += [{'params': model.siamese.parameters(),
                          'lr': 1}]
    if cfg.cornerdet.cornerdet:
        if epoch >= cfg.train.pretrain_epoch:
            if cfg.attention.attention:
                for param in model.attention.parameters():
                    param.requires_grad = True
                trainable_params += [{'params': model.attention.parameters(),
                                    'lr': 1}]
            for param in model.cornerdet.parameters():
                param.requires_grad = True
            trainable_params += [{'params': model.cornerdet.parameters(),
                                'lr': 1}]
        else:
            if cfg.attention.attention:
                for param in model.attention.parameters():
                    param.requires_grad = False
            for param in model.cornerdet.parameters():
                param.requires_grad = False
    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, cfg.train.lr, cfg.train.epoch)
    logger.info('change training parameters.')
    logger.info("model\n{}".format(describe(model)))
    return optimizer, lr_scheduler


class Meter(object):
    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}

    def update(self, batch=1, **kwargs):
        val = {}
        for k in kwargs:
            val[k] = kwargs[k] / float(batch)
        self.val.update(val)
        for k in kwargs:
            if k not in self.sum:
                self.sum[k] = 0
                self.count[k] = 0
            self.sum[k] += kwargs[k]
            self.count[k] += batch

    def __repr__(self):
        s = ''
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
            name=attr,
            val=float(self.val[attr]),
            avg=float(self.sum[attr]) / self.count[attr])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super(AverageMeter, self).__getattr__(attr)
        if attr not in self.sum:
            # logger.warn("invalid key '{}'".format(attr))
            print("invalid key '{}'".format(attr))
            return Meter(attr, 0, 0)
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr):
        return float(self.sum[attr]) / self.count[attr]


def save_checkpoint(state, epoch, save_path='checkpoint'):
    filename = join(save_path, 'checkpoint_epoch{}.pth'.format(epoch))
    torch.save(state, filename)


def train(train_loader, model, optimizer, epoch):
    avg = AverageMeter()
    model.train()
    end = time.time()
    for i, sample in enumerate(tqdm(train_loader)):
        data_time = time.time() - end
        perform_corner = epoch >= cfg.train.pretrain_epoch
        outputs = model(sample, perform_corner=perform_corner)

        for k, v in outputs.items():
            outputs[k] = torch.mean(v)

        loss = outputs['loss']
        
        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        batch_time = time.time() - end

        avg.update(**outputs)

        avg.update(data_time=data_time, batch_time=batch_time)
        end = time.time()

        if (epoch == 0 or epoch == cfg.train.pretrain_epoch) and i % cfg.train.print_freq == 0:
            message = 'Epoch: [{0}][{1}/{2}]\t'.format(
                epoch, i, len(train_loader))
            message += '{batch_time:s}\t{data_time:s}\t'.format(
                batch_time=avg.batch_time, data_time=avg.data_time)
            for k in outputs:
                message += '{loss:s}\t'.format(loss=avg.__getattr__(k))
            logger.info(message)

    record_dict = {}
    for k in outputs:
        record_dict[k] = avg.__getattr__(k).avg

    return record_dict


def validate(val_loader, model, epoch):
    avg = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_loader)):
            perform_corner = epoch >= cfg.train.pretrain_epoch
            outputs = model(sample, perform_corner=perform_corner)

            for k, v in outputs.items():
                outputs[k] = torch.mean(v)

            loss = outputs['loss']

            batch_time = time.time() - end
            avg.update(**outputs)
            avg.update(batch_time=batch_time)
            end = time.time()

    record_dict = {}
    for k in outputs:
        record_dict[k] = avg.__getattr__(k).avg
    return record_dict

def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)

def main(args):
    cfg_from_file(args.config)
    cfg.save_name = args.save_name
    cfg.save_path = args.save_path
    cfg.resume_file = args.resume_file
    cfg.config = args.config
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    save_path = join(args.save_path, args.save_name)
    if not exists(save_path):
        makedirs(save_path)
    resume_file = args.resume_file
    init_log('global', logging.INFO)
    add_file_handler('global',
                             os.path.join(save_path, 'logs.txt'),
                             logging.INFO)
    logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    start_epoch = 0

    model = ModelBuilder().cuda()
    if cfg.backbone.pretrained:
        load_pretrain(model.backbone, join(
            'pretrained_net', cfg.backbone.pretrained))

    train_dataset = Datasets()
    val_dataset = Datasets(is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=True)

    if resume_file:
        if isfile(resume_file):
            logger.info("=> loading checkpoint '{}'".format(resume_file))
            model, start_epoch = restore_from(
                model, resume_file)
            start_epoch = start_epoch + 1
            for i in range(start_epoch):
                train_loader.dataset.shuffle()
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, start_epoch-1))
        else:
            logger.info("=> no checkpoint found at '{}'".format(resume_file))

    ngpus = torch.cuda.device_count()
    is_dataparallel = False
    if ngpus > 1:
        model = torch.nn.DataParallel(model, list(range(ngpus))).cuda()
        is_dataparallel = True

    if is_dataparallel:
        optimizer, lr_scheduler = build_opt_lr(model.module, start_epoch)
    else:
        optimizer, lr_scheduler = build_opt_lr(model, start_epoch)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    if args.log:
        writer = SummaryWriter(comment=args.save_name)

    for epoch in range(start_epoch, cfg.train.epoch):
        train_loader.dataset.shuffle()
        if (epoch == np.array(cfg.backbone.unfix_steps)).sum() > 0 or epoch == cfg.train.pretrain_epoch:
            if is_dataparallel:
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
            else:
                optimizer, lr_scheduler = build_opt_lr(model, epoch)
        lr_scheduler.step(epoch)
        record_dict_train = train(train_loader, model, optimizer, epoch)
        record_dict_val = validate(val_loader, model, epoch)
        message = 'Train Epoch: [{0}]\t'.format(epoch)
        for k, v in record_dict_train.items():
            message += '{name:s} {loss:.4f}\t'.format(name=k, loss=v)
        logger.info(message)
        message = 'Val Epoch: [{0}]\t'.format(epoch)
        for k, v in record_dict_val.items():
            message += '{name:s} {loss:.4f}\t'.format(name=k, loss=v)
        logger.info(message)

        if args.log:
            for k, v in record_dict_train.items():
                writer.add_scalar('train/' + k, v, epoch)
            for k, v in record_dict_val.items():
                writer.add_scalar('val/' + k, v, epoch)
        if is_dataparallel:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg
            }, epoch, save_path)
        else:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg
            }, epoch, save_path)


if __name__ == '__main__':
    random.seed(1234)
    os.environ['PYTHONHASHSEED'] = str(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    cudnn.benchmark = False
    cudnn.deterministic = True
    args = parse_arguments()
    main(args)
