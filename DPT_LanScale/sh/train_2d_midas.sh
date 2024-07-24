exp_name="0724_2d_midas_panoptic_seg_txt_NL1_tmux0"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train_2d.py configs/arguments_train_2d_midas.txt  2>&1 | tee models/${exp_name}/result.log