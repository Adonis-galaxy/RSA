exp_name="0710_2d_midas_seg_txt_nyul09_tmux3"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_2d.py configs/arguments_train_2d_midas.txt  2>&1 | tee models/${exp_name}/result.log