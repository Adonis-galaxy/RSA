exp_name="1026_3d_dpt_15_captions_small_tmux3"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_3d.py configs/arguments_train_3d_dpt.txt  2>&1 | tee models/${exp_name}/result.log