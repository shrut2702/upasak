from upasak.fine_tune import TrainingEngine, TrainerConfig
from datasets import load_dataset

output_dir = "your_target_directory"
config = TrainerConfig(
    model_name = "google/gemma-3-270m-it",
    output_dir = output_dir,
    lr = 2e-4,
    epochs = 3,
    batch_size = 4,
    gradient_accumulation = 1,
    logging_steps = 10,
    warmup_ratio = 0.0,
    lr_scheduler_type = "cosine",
    weight_decay = 0.0,
    max_grad_norm = 0.0,
    save_strategy = "no",
    save_steps = 500,
    eval_strategy = "steps",
    eval_steps = 5,
    report_to = "none",
    model_tracker_api_key = "",
    is_lora = False,
    lora_r=None,  # not None value if is_lora is True
    lora_alpha=None, # not None value if is_lora is True
    lora_dropout=None, # not None value if is_lora is True
    merge_adapters=False # not None value if is_lora is True
)

tokenized_train_dataset = load_dataset("json", data_files="./tokenized_train_dataset.json", split="train") #the split is just a name, it doesn't change any workings
tokenized_validation_dataset = load_dataset("json", data_files="./tokenized_validation_dataset.json", split="train") #the split is just a name, it doesn't change any workings

tokenized_dataset = {
    "train": tokenized_train_dataset,
    "validation": tokenized_validation_dataset # this field is optional, only if there is validation dataset
}
hf_token = "your_hf_access_token"

# These datasets contains input_ids and attention_mask, the label_ids will handled inside TrainingEngine. 
# Moreover, the samples inside the tokenized dataset are not padded to equal lengths, this will also be handled inside TrainingEngine.

trainer = TrainingEngine(train_config=config, hf_token=hf_token)
trainer.pipeline(tokenized_dataset)