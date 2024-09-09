exp_name="0909_3d_midas_llava16vicuna_1captions_tmux1"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train_3d.py configs/arguments_train_3d_midas.txt  2>&1 | tee models/${exp_name}/result.log