python tools/construct_parallel_dataset.py

nohup bash ./sh/train_finetune_ar_model.sh &
nohup bash ./sh/train_finetune_nar_model.sh &