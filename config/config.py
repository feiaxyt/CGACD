# Copyright (c) SenseTime. All Rights Reserved.

from easydict import EasyDict as edict
import numpy as np

__c = edict()

cfg = __c

__c.train = edict()

__c.train.epoch = 20

__c.train.pretrain_epoch = 1
#for cls loss
__c.train.cls_weight = 1.0
#for bbox reg loss
__c.train.breg_weight = 1.0
#for corner reg loss
__c.train.creg_weight = 0.25

__c.train.response_size = 25

__c.train.stride = 8

__c.train.search_size = 255

__c.train.template_size = 127

__c.train.breg_sigma = 5.0

__c.train.label_sigma1 = 0.3

__c.train.label_sigma2 = 0.4

__c.train.template_pool_size = 5

__c.train.search_pool_size = 7

#for train config
__c.train.weight_decay = 5e-4

__c.train.momentum = 0.9

__c.train.print_freq = 50

__c.backbone = edict()

__c.backbone.type = 'resnet50'

__c.backbone.pretrained = 'resnet50.model'

__c.backbone.unfix_layers = ['layer3', 'layer2']
__c.backbone.unfix_steps = [10, 10]
__c.backbone.unfix_lr = [0.1, 0.1]

__c.backbone.kwargs = edict()

#__c.backbone.kwargs.used_layers = [3]

__c.adjust = edict()

__c.adjust.adjust = False

__c.adjust.type = 'AdjustLayer'

__c.adjust.kwargs = edict()

#__c.adjust.kwargs.in_channels = 1024
#__c.adjust.kwargs.out_channels = 256

__c.siamese = edict()

__c.siamese.type = 'UPChannelSiamese'

__c.siamese.kwargs = edict()

#__c.siamese.kwargs.weighted = False
#__c.siamese.kwargs.feat_in = 256

__c.attention = edict()

__c.attention.attention = True

__c.attention.type = 'PixelAttention'

__c.attention.kwargs = edict()

__c.cornerdet = edict()

__c.cornerdet.cornerdet = True

__c.cornerdet.type = 'SepCornerDet'

__c.cornerdet.kwargs = edict()
#__c.cornerdet.kwargs.feat_in = 256



#for dataset
__c.train.train_dataset = edict()

__c.train.train_dataset.names = ['youtubebb', 'got10k', 'vid', 'coco', 'det']

__c.train.train_dataset.youtubebb = edict()

__c.train.train_dataset.youtubebb.root = '/ssd/feiji/Research/Data/y2b_crop511'

__c.train.train_dataset.youtubebb.anno = '/home/feiji/Research/Data/data_preprocess/y2b/train_largeclean.json'

__c.train.train_dataset.youtubebb.num_use = 100000

__c.train.train_dataset.youtubebb.frame_range = 2

__c.train.train_dataset.got10k = edict()

__c.train.train_dataset.got10k.root = '/ssd/feiji/Research/Data/GOT-10k_crop511'

__c.train.train_dataset.got10k.anno = '/home/feiji/Research/Data/data_preprocess/got10k/train_largeclean.json'

__c.train.train_dataset.got10k.num_use = 100000

__c.train.train_dataset.got10k.frame_range = 100

__c.train.train_dataset.vid = edict()

__c.train.train_dataset.vid.root = '/ssd/feiji/Research/Data/VID_crop511'

__c.train.train_dataset.vid.anno = '/home/feiji/Research/Data/data_preprocess/vid/train_largeclean.json'

__c.train.train_dataset.vid.num_use = 50000

__c.train.train_dataset.vid.frame_range = 100

__c.train.train_dataset.coco = edict()

__c.train.train_dataset.coco.root = '/ssd/feiji/Research/Data/COCO_crop511'

__c.train.train_dataset.coco.anno = '/home/feiji/Research/Data/data_preprocess/coco/train2017_largeclean.json'

__c.train.train_dataset.coco.num_use = 50000

__c.train.train_dataset.coco.frame_range = 1

__c.train.train_dataset.det = edict()

__c.train.train_dataset.det.root = '/ssd/feiji/Research/Data/DET_crop511'

__c.train.train_dataset.det.anno = '/home/feiji/Research/Data/data_preprocess/det/train_largeclean.json'

__c.train.train_dataset.det.num_use = 50000

__c.train.train_dataset.det.frame_range = 1

__c.train.train_dataset.video_per_epoch = 350000

__c.train.train_dataset.augmentation = edict()

__c.train.train_dataset.augmentation.neg = 0.2

__c.train.train_dataset.augmentation.gray = 0.0

__c.train.train_dataset.augmentation.norm = 0

__c.train.train_dataset.augmentation.template = edict()

__c.train.train_dataset.augmentation.template.shift = 4

__c.train.train_dataset.augmentation.template.scale = 0.05

__c.train.train_dataset.augmentation.template.blur = 0.0

__c.train.train_dataset.augmentation.template.color = 1.0

__c.train.train_dataset.augmentation.search = edict()

__c.train.train_dataset.augmentation.search.shift = 12

__c.train.train_dataset.augmentation.search.scale = 0.15

__c.train.train_dataset.augmentation.search.blur = 0.0

__c.train.train_dataset.augmentation.search.color = 1.0

#for val dataset
__c.train.val_dataset = edict()

__c.train.val_dataset.names = ['vid']

__c.train.val_dataset.vid = edict()

__c.train.val_dataset.vid.root = '/ssd/feiji/Research/Data/VID_crop511'

__c.train.val_dataset.vid.anno = '/home/feiji/Research/Data/data_preprocess/vid/val_clean.json'

__c.train.val_dataset.vid.num_use = 1000

__c.train.val_dataset.vid.frame_range = 100

__c.train.val_dataset.video_per_epoch = 1000

__c.train.val_dataset.augmentation = edict()

__c.train.val_dataset.augmentation.neg = 0.0

__c.train.val_dataset.augmentation.gray = 0.0

__c.train.val_dataset.augmentation.norm = 0

__c.train.val_dataset.augmentation.template = edict()

__c.train.val_dataset.augmentation.template.shift = 0.0

__c.train.val_dataset.augmentation.template.scale = 0.0

__c.train.val_dataset.augmentation.template.blur = 0.0

__c.train.val_dataset.augmentation.template.color = 0.0

__c.train.val_dataset.augmentation.search = edict()

__c.train.val_dataset.augmentation.search.shift = 0.0

__c.train.val_dataset.augmentation.search.scale = 0.0

__c.train.val_dataset.augmentation.search.blur = 0.0

__c.train.val_dataset.augmentation.search.color = 0.0

__c.train.roi_augmentation = edict()

__c.train.roi_augmentation.ratio = 0.5

__c.train.roi_augmentation.shift = 8

__c.train.roi_augmentation.scale = 0.1


#for lr schedule
__c.train.lr = edict()

__c.train.lr.type = 'log'

__c.train.lr.start_lr = 0.001

__c.train.lr.end_lr = 0.0001

#for pretrain
__c.train.lr.pretrain = edict()

__c.train.lr.pretrain.start_lr = 0.001

__c.train.lr.pretrain.type = 'step'

__c.train.lr.pretrain.step = 1

__c.train.lr.pretrain.epoch = 1

#for warm up
__c.train.lr.warmup = edict()

__c.train.lr.warmup.start_lr = 0.0006

__c.train.lr.warmup.end_lr = 0.001

__c.train.lr.warmup.type = 'step'

__c.train.lr.warmup.step = 1

__c.train.lr.warmup.epoch = 4


__c.track = edict()

__c.track.template_size = 127
__c.track.search_size = 255
__c.track.response_size = 25
__c.track.stride = 8
__c.track.windowing = 'cosine'
__c.track.contex_amount = 0.5
__c.track.penalty_k = 0.055
__c.track.window_influence = 0.42
__c.track.lr = 0.2

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b and type(v) is edict:
      raise KeyError('{} is not a valid config key'.format(k))
    
    # the types must match, too
    if k in b:
      old_type = type(b[k])
      if old_type is not type(v):
        if isinstance(b[k], np.ndarray):
          v = np.array(v, dtype=b[k].dtype)
        else:
          raise ValueError(('Type mismatch ({} vs. {}) '
                            'for config key: {}').format(type(b[k]),
                                                        type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

  _merge_a_into_b(yaml_cfg, __c)

