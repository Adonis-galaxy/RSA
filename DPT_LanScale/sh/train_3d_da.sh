exp_name="0921_3d_da_llava16mistral_5captions_noaugment_100epoch_4gpu_tmux0"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_3d.py configs/arguments_train_3d_da.txt  2>&1 | tee models/${exp_name}/result.log