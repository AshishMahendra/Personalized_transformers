from p_transformers.dataset import get_dataset
import json
from transformers import AutoTokenizer, AutoAdapterModel
from transformers import RobertaConfig, RobertaModelWithHeads
import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoAdapterModel.from_pretrained("distilbert-base-uncased")

adapters_to_freeze = [int(i) for i in range(1, 11)]
with open("clinc/test_grouped.json", "r") as file:
    g_data = json.load(file)
    group_count = 0
    for group_key, dset in g_data.items():
        # creates a train/test datasets
        dataset_name = "clinc"
        dataset_label_num = len(dset["train"].keys())
        train, test = get_dataset(tokenizer=tokenizer, dataset=dset)
        print("data_load Completed")
        group_count += 1
        if group_count == 1:
            break


# Add a new adapter
model.add_adapter("clinc")
# Add a matching classification head
model.add_classification_head("clinc", num_labels=15)
# Activate the adapter
model.train_adapter("clinc")


training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=compute_accuracy,
)


trainer.train()
