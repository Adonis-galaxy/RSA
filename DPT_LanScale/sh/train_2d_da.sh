exp_name="0604_2d_da_tmuxX"

mkdir ../models/${exp_name}
CUDA_VISIBLE_DEVICES=0 python ../train_2d_da.py ../configs/arguments_train_2d_da.txt  2>&1 | tee ../models/${exp_name}/result.log