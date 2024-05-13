exp_name="0512_kitti_da_inv_tmux1"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train_da.py configs/arguments_train_kittieigen_da.txt  2>&1 | tee ./models/${exp_name}/result.log