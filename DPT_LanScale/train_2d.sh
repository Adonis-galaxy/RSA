exp_name="0506_2d_image_no_roomtype_tmux1"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train_2d.py configs/arguments_train_2d.txt  2>&1 | tee ./models/${exp_name}/result.log