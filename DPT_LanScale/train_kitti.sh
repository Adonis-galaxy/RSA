exp_name="0416_kitti_exp_scale_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${exp_name}/result.log