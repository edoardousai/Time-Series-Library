model_name=HCNNLSTM

learning_rate=0.01
train_epochs=10
patience=3
batch_size=32
dropout=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0 \
  --learning_rate 0.0037 \
  --filter1_size 4 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.3 \
  --learning_rate 0.0047 \
  --filter1_size 8 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --learning_rate 0.0047 \
  --filter1_size 16 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --learning_rate 0.0036 \
  --filter1_size 32 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size