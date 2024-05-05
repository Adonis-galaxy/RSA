exp_name="0504_depth_anything_nyu_room_type_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_da.py configs/arguments_train_nyu_da.txt  2>&1 | tee ./models/${exp_name}/result.log