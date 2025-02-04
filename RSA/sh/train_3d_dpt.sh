exp_name="train_3d_dpt"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=1 python train_3d.py configs/arguments_train_3d_dpt.txt  2>&1 | tee models/${exp_name}/result.log