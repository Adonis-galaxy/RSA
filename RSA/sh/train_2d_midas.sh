exp_name="0901_2d_midas_llava16mistral7btext_5captions_tmux1"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train_2d.py configs/arguments_train_2d_midas.txt  2>&1 | tee models/${exp_name}/result.log