exp_name="0829_2d_midas_llava16vicuna7btext_tmux1"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train_2d.py configs/arguments_train_2d_midas.txt  2>&1 | tee models/${exp_name}/result.log