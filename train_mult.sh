for layers in 0
do
    python train_prefix.py -m "bert-base-uncased" --in_file "go_emotion_full_input.json" -frz true -flay ${layers}

done
# python train_adaptor.py -m "bert-base-uncased" --in_file "go_emotion_full_input.json" -flay 12

