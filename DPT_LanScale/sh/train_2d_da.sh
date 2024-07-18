exp_name="0718_2d_da_seg_txt_norm_loss_huber_tmux3"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=3 python train_2d.py configs/arguments_train_2d_da.txt  2>&1 | tee models/${exp_name}/result.log