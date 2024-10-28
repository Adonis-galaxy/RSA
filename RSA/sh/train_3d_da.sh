exp_name="1027_3d_da_15_captions_small_tmux0"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train_3d.py configs/arguments_train_3d_da.txt  2>&1 | tee models/${exp_name}/result.log