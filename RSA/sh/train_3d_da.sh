exp_name="train_3d_da"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train_3d.py configs/arguments_train_3d_da.txt  2>&1 | tee models/${exp_name}/result.log