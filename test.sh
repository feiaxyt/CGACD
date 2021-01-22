export PYTHONPATH=$PWD:$PYTHONPATH
tracker_name="CGACD_VOT"
config_file="experiments/cgacd_resnet/cgacd_resnet.yml"
START=11
END=19
for s in $(seq $START 1 $END)
do
    python tools/test.py \
    --model "checkpoint/"$tracker_name"/checkpoint_epoch"$s".pth" \
    --config "config/"$config_file \
	  --dataset "VOT2018" \
    --save_name $tracker_name"_"$s
done
