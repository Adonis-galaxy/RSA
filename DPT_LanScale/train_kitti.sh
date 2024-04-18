exp_name="0418_kitti_exp_scale_tmux3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${exp_name}/result.log