exp_name="0829_2d_da_llava16vicuna7btext_tmux0"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train_2d.py configs/arguments_train_2d_da.txt  2>&1 | tee models/${exp_name}/result.log