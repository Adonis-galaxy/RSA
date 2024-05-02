exp_name="0502_2d_new_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_2d.py configs/arguments_train_2d.txt  2>&1 | tee ./models/${exp_name}/result.log