exp_name="0506_da_nyu_image_no_roomtype_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_da.py configs/arguments_train_nyu_da.txt  2>&1 | tee ./models/${exp_name}/result.log