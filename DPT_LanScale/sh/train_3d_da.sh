exp_name="0923_3d_da_10_captions_tmux0"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=1,2,3 python train_3d.py configs/arguments_train_3d_da.txt  2>&1 | tee models/${exp_name}/result.log