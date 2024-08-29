exp_name="0829_2d_dpt_llava16vicuna7btext_tmux2"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python train_2d.py configs/arguments_train_2d_dpt.txt  2>&1 | tee models/${exp_name}/result.log