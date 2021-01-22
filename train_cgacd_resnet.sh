export PYTHONPATH=$PWD:$PYTHONPATH

#CUDA_VISIBLE_DEVICES=1
python train/train.py \
    --config=experiments/cgacd_resnet/cgacd_resnet.yml \
    -b 64 \
    -j 16 \
    --save_name cgacd_resnet
