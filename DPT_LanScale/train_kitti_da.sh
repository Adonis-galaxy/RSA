exp_name="0504_depth_anything_kitti_nofinetune_clip_tmux1"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train_da.py configs/arguments_train_kittieigen_da.txt  2>&1 | tee ./models/${exp_name}/result.log