exp_name="0412_kitti_sigmoid_0d001_exp_0d1_right_data_tmux1"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${exp_name}/result.log