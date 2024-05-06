exp_name="0506_da_kitti_image_no_roomtype_tmux3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_da.py configs/arguments_train_kittieigen_da.txt  2>&1 | tee ./models/${exp_name}/result.log