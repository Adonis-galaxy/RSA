exp_name="0806_da_void_eval"

mkdir models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python eval_void.py configs/arguments_eval_void.txt  2>&1 | tee models/${exp_name}/result.log