exp_name="0416_kitti_default_init_tmux1"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${exp_name}/result.log