for layers in 12
do
    python  train_lora.py --out_file "logs/lora_eval_group_0_adaptor_base_trained..json" -m "bert-base-uncased" --in_file "datasets/hatespeech_user_train_data.json" -flay ${layers}
    python train_adaptor.py --out_file "logs/adaptor_eval_group_0_adaptor_base_trained.json" -m "bert-base-uncased" --in_file "datasets/hatespeech_user_train_data.json" -flay ${layers}
    python train_prefix.py --out_file "logs/prefix_eval_group_0_adaptor_base_trained..json" -m "bert-base-uncased" --in_file "datasets/hatespeech_user_train_data.json" -flay ${layers}
done
