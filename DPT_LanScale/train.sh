exp_name="0315_0_sigmoid_scale_lr_1en4_exp3_tmux_3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log