exp_name="0510_kitti_buffer_tmux3"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_midas.py configs/arguments_train_kittieigen_midas.txt  2>&1 | tee ./models/${exp_name}/result.log