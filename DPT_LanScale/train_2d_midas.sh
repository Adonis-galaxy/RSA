exp_name="0509_2d_midas_tmux0"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train_2d_midas.py configs/arguments_train_2d_midas.txt  2>&1 | tee ./models/${exp_name}/result.log