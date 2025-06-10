model_name=HCNNLSTM

learning_rate=0.01
train_epochs=10
patience=3
batch_size=32
dropout=0.0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.3 \
  --learning_rate 0.003 \
  --filter1_size 16 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --use_multi_gpu \
  --devices 0,1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --learning_rate 0.0086 \
  --filter1_size 32 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --use_multi_gpu \
  --devices 0,1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.3 \
  --learning_rate 0.0076 \
  --filter1_size 16 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --use_multi_gpu \
  --devices 0,1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.5 \
  --learning_rate 0.0025 \
  --filter1_size 4 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --use_multi_gpu \
  --devices 0,1
