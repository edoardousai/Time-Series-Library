model_name=STHNN

learning_rate=0.001
train_epochs=10
patience=3
batch_size=32
dropout=0.5

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --learning_rate 0.005\
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --learning_rate 0.01 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size
