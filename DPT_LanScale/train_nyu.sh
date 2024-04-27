exp_name="0425_nyu_ablation_tmux0"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log