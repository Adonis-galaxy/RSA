exp_name="0425_2d"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python train_2d.py configs/arguments_train_2d.txt  2>&1 | tee ./models/${exp_name}/result.log