exp_name="0514_2d_da_resize_tmux3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_2d_da.py configs/arguments_train_2d_da.txt  2>&1 | tee ./models/${exp_name}/result.log