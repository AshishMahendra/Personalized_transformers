from transformers.adapters import LoRAConfig
from dataset import get_dataset
import json
from transformers import (
    AutoTokenizer,
    AutoAdapterModel,
    TrainingArguments,
    AdapterTrainer,
    EvalPrediction,
)
import numpy as np

# class AdapterDropTrainerCallback(TrainerCallback):
#   def on_step_begin(self, args, state, control, **kwargs):
#     skip_layers = [str(i) for i in range(0,frz_layers)]
#     kwargs['model'].set_active_adapters("go_emotion", skip_layers=skip_layers)
#   def on_evaluate(self, args, state, control, **kwargs):
#     skip_layers = [str(i) for i in range(0,frz_layers)]
#     kwargs['model'].set_active_adapters("go_emotion", skip_layers=skip_layers)

def model_setup(model_name = "distilbert-base-uncased",freeze_list=None,freeze=True):
    global model,tokenizer,adapters_to_freeze
    # if freeze_list is not None:
    adapters_to_freeze = freeze_list
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    print(f"model_loaded : {model_name} , Frozen Base : {freeze}, frozen adaptar list : {freeze_list}")


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

def train_module(train, test, uniq_lbl_count, adapter_name,freeze):
    config = LoRAConfig(r=8, alpha=16)
    # Add a new adapter
    model.add_adapter(adapter_name, config=config)
    # Add a matching classification head
    model.add_classification_head(adapter_name, num_labels=uniq_lbl_count)
    # Activate the adapter
    print("after adding adaptor")
    print("adapter to freeze", adapters_to_freeze)
    # if adapters_to_freeze is not None:
    #     model.set_active_adapters(adapter_name, skip_layers=adapters_to_freeze)
    # else:
    #     model.set_active_adapters(adapter_name)
    model.set_active_adapters("go_emotion", skip_layers=adapters_to_freeze)
    model.train_adapter(adapter_name)
    model.freeze_model(freeze=freeze)
    
    print("################# BEFORE TRAINIG ##################")
    training_args = TrainingArguments(
        learning_rate=3e-5,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=1000,
        output_dir="./training_output",
        overwrite_output_dir=True,
        save_total_limit = 2,
        load_best_model_at_end=True,
        evaluation_strategy = "epoch",
        save_strategy = 'epoch')
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_accuracy,
    )
    # trainer.add_callback(AdapterDropTrainerCallback())
    trainer.train()
    print("################# Evaluating ##################")
    print(model.get_labels())
    print(model.get_labels_dict())
    resp = trainer.evaluate(eval_dataset=test)
    model.delete_adapter(adapter_name)
    model.delete_head(adapter_name)
    return resp


# def group_train_eval():
#     import csv

#     with open("output.csv", "w", newline="") as outcsv:
#         writer = csv.DictWriter(outcsv, fieldnames=["Group_number", "Accuracy"])
#         writer.writeheader()
#         with open("clinc/clinc_grouped.json", "r") as file:
#             g_data = json.load(file)
#             group_count = 0
#             for group_key, dset in g_data.items():
#                 # creates a train/test datasets
#                 adapter_name = group_key
#                 grp_uniq_lbl_count = len(dset["train"].keys())
#                 train, test = get_clinc_dataset(tokenizer=tokenizer, dataset=dset)
#                 print(f"Data load Completed for Group {group_key}")
#                 print(f"sample :- train : {train[0]} test : {test[0]}")
#                 eval_output = train_module(train, test, grp_uniq_lbl_count, adapter_name)
#                 writer.writerows({"Group_number": adapter_name, "Accuracy": eval_output["eval_acc"]})
#                 group_count += 1
#                 if group_count == 3:
#                     break

def write_to_file(out_file,eval_output,freeze):
    try:
        with open(out_file,"r") as fp:
            listObj = json.load(fp)
    except:
        listObj=[]
    eval_output["frozen_adapters"]=adapters_to_freeze
    eval_output["freeze_base"]=freeze
    print(listObj)
    if len(listObj)>0:
        listObj.append(eval_output)
    else:
        listObj=[eval_output]
    with open(out_file,"w") as lp:
        json.dump(listObj,lp)


def full_data_eval(input_file, freeze,out_file=None):
    with open(input_file, "r") as file:
        full_data = json.load(file)
    train, test = get_dataset(tokenizer=tokenizer, dataset=full_data)
    print(f"Data load Completed for Group all_user")
    uniq_lbl_count = len(full_data["train"].keys())
    eval_output=train_module(train, test, uniq_lbl_count, "go_emotion",freeze=freeze)
    print(eval_output)
    write_to_file(out_file=out_file,eval_output=eval_output,freeze=freeze)


if __name__ == "__main__":
    global frz_layers
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--out_file",help="output_file_name", default="lora_eval.json")
    parser.add_argument("-if","--in_file",help="input_file_name", default="go_emotion_full_input.json")
    parser.add_argument("-m","--model_name",help="model name",default="prajjwal1/bert-tiny")
    parser.add_argument("-frz","--freeze",help="Freeze base model",type=bool,default=False)
    parser.add_argument("-flay","--frz_layers",help="No. of adaptars layer to freeze",type=int,default=12)
    args = parser.parse_args()
    frz_layers=int(args.frz_layers)

    freeze_list=[str(i) for i in range(0,args.frz_layers)]
    model_setup(model_name=args.model_name,freeze=args.freeze,freeze_list=freeze_list)
    full_data_eval(input_file=args.in_file,out_file=args.out_file,freeze=args.freeze)