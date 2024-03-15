exp_name="0314_2_sigmoid_scale_no_init_larger_lr_tmux_2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log