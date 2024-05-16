exp_name="0516_nyu_da_resize_combine_words_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_da.py configs/arguments_train_nyu_da.txt  2>&1 | tee ./models/${exp_name}/result.log