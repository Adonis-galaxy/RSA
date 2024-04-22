exp_name="0422_nyu_siloss_swap_area_percent_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log