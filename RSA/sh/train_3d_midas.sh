exp_name="1025_3d_midas_15_captions_small_tmux2"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_3d.py configs/arguments_train_3d_midas.txt  2>&1 | tee models/${exp_name}/result.log