exp_name="0509_nyu_midas_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_midas.py configs/arguments_train_nyu_midas.txt  2>&1 | tee ./models/${exp_name}/result.log