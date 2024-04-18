exp_name="0416_nyu_remove_lambda_100_tmux3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log