exp_name="0704_2d_da_eps_nyulambda09_tmux0"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train_2d.py configs/arguments_train_2d_da.txt  2>&1 | tee models/${exp_name}/result.log