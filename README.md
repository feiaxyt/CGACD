# [CGACD](https://openaccess.thecvf.com/content_CVPR_2020/html/Du_Correlation-Guided_Attention_for_Corner_Detection_Based_Visual_Tracking_CVPR_2020_paper.html)

## 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.7, Pytorch 1.1.0, CUDA 10.0. Please install related libraries before running this code:
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```
### Add CGACD to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/CGACD:$PYTHONPATH
```


## 2. Test
Download the pretrained model:  [OTB and VOT](https://pan.baidu.com/s/11z74ZUGAPhupPLNrbGN5NQ) (code: 16s0) and put them into `checkpoint` directory.

Download testing datasets and put them into `dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F) or [Google driver](https://drive.google.com/drive/folders/1TC8obz4TvlbvTRWbS4Cn4VwwJ8tXs2sv?usp=sharing). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

```bash
python tools/test.py                                \
	--dataset VOT2018                      \ # dataset_name
	--model checkpoint/CGACD_VOT.pth  \ # tracker_name
    --save_name CGACD_VOT
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Train
### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)

Scripts to prepare training dataset are listed in `training_dataset` directory.

### Download pretrained backbones
Download pretrained backbones from [google driver](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) or [BaiduYun](https://pan.baidu.com/s/1pYe73PjkQx4Ph9cd3ePfCQ) (code: 5o1d) and put them into `pretrained_net` directory.

### Train a model
To train the CGACD model, run `train.py` with the desired configs:

```bash
python tools/train.py 
    --config=experiments/cgacd_resnet/cgacd_resnet.yml \
    -b 64 \
    -j 16 \
    --save_name cgacd_resnet
```

We use two RTX2080TI for training.

## 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1fM36M19LUgd3hI0QFnwkdw) (code: qw69 ) of OTB2015, VOT2018, UAV123, and LaSOT. If you want to evaluate the tracker, please put those results into  `results` directory.

```
python eval.py 	                          \
	-p ./results          \ # result path
	-d VOT2018             \ # dataset_name
	-t CGACD_VOT   # tracker_name
```

## 5. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot) and [PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling). We would like to express our sincere thanks to the contributors.


## 6. Cite
If you use CGACD in your work please cite our paper:
> @InProceedings{Du_2020_CVPR,  
   author = {Du, Fei and Liu, Peng and Zhao, Wei and Tang, Xianglong},  
   title = {Correlation-Guided Attention for Corner Detection Based Visual Tracking},  
   booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
   month = {June},  
   year = {2020}  
}


