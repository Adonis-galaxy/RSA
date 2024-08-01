exp_name="0731_2d_dpt_car_scale_01_001_tmux3"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_2d.py configs/arguments_train_2d_dpt.txt  2>&1 | tee models/${exp_name}/result.log