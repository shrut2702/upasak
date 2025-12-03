from upasak.preprocessing import TokenizerWrapper
from upasak.fine_tune import TrainingEngine, TrainerConfig

model_name = "google/gemma-3-270m-it"
hf_token = "your_hf_access_token"
dataset_path = "./dummy_dataset.json"
output_dir = "your_target_directory"

tokenizer = TokenizerWrapper(model_name, hf_token=hf_token)
tokenized_dataset = tokenizer.pipeline(file_path=dataset_path, max_length=128, split= 0.5)

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
    selected_target_modules=None, # not None value if is_lora is True
    merge_adapters=False # not None value if is_lora is True
)

trainer = TrainingEngine(train_config=config, hf_token=hf_token)
trainer.pipeline(tokenized_dataset)