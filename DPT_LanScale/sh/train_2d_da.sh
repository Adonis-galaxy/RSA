exp_name="0627_2d_da_lnyu01_tmux2"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_2d.py configs/arguments_train_2d_da.txt  2>&1 | tee models/${exp_name}/result.log