exp_name="0502_depth_anything_minibatch_tmux3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_da.py configs/arguments_train_nyu_da.txt  2>&1 | tee ./models/${exp_name}/result.log