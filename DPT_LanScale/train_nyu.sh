exp_name="0411_nyu_text_aug_number_removeprob_0d5_tmux0"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${exp_name}/result.log