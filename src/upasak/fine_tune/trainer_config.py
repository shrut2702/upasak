from dataclasses import dataclass

@dataclass
class TrainerConfig:
    model_name: str = "google/gemma-3-270m-it"
    output_dir: str = None
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_seq_len: int = 2048
    logging_steps: int = 10
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    use_4bit: bool = False
    bf16: bool = False
    save_strategy: str = "no"
    save_steps: int = 500
    eval_strategy: str = "no"
    eval_steps: int = 500
    report_to: str = "none"
    model_tracker_api_key: str = ""
    val_split: float = None
    include_num_input_tokens_seen: bool = True
    is_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    selected_target_modules: list = None
    merge_adapters: bool = False