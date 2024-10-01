exp_name="1001_3d_da_15_captions_e200_medium_tmux0"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_3d.py configs/arguments_train_3d_da.txt  2>&1 | tee models/${exp_name}/result.log