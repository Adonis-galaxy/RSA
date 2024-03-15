exp_name="0314_2_test_tmux_0"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log