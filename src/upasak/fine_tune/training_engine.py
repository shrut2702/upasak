from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import Dataset
import os
from typing import List

from .trainer_config import TrainerConfig

class TrainingEngine:
    def __init__(self, train_config: TrainerConfig, hf_token: str):
        self.training_args = TrainingArguments(
            output_dir=train_config.output_dir,
            learning_rate=train_config.lr,
            num_train_epochs=train_config.epochs,
            per_device_train_batch_size=train_config.batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation,
            logging_steps=train_config.logging_steps,
            bf16=train_config.bf16,
            include_num_input_tokens_seen=train_config.include_num_input_tokens_seen,
            warmup_ratio=train_config.warmup_ratio,
            weight_decay=train_config.weight_decay,
            lr_scheduler_type=train_config.lr_scheduler_type,
            save_strategy=train_config.save_strategy,
            save_steps=train_config.save_steps,
            max_grad_norm=train_config.max_grad_norm,
            eval_strategy=train_config.eval_strategy,
            eval_steps=train_config.eval_steps,
            report_to=train_config.report_to,
            run_name="Gemma Fine-Tuning" if train_config.report_to != "none" else None,
        )
        self.output_dir = train_config.output_dir
        self.model_name = train_config.model_name
        self.is_lora = train_config.is_lora
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<eos>"})
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({"bos_token": "<bos>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if self.is_lora:
            self.lora_config = LoraConfig(
                r=train_config.lora_r,
                lora_alpha=train_config.lora_alpha,
                lora_dropout=train_config.lora_dropout,
                task_type="CAUSAL_LM",
                target_modules=train_config.selected_target_modules,
                bias="none"
            )
        self.merge_adapters = train_config.merge_adapters
        if train_config.report_to == "wandb":
            os.environ["WANDB_API_KEY"] = train_config.model_tracker_api_key
        elif train_config.report_to == "comet_ml":
            os.environ["COMET_API_KEY"] = train_config.model_tracker_api_key

    def __call__(self, dataset: Dataset, callbacks=[]):
        self.pipeline(dataset, callbacks)

    def pipeline(self, dataset: dict[str, Dataset], callbacks: List = []):
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        if self.is_lora:
            self.model = get_peft_model(self.model, self.lora_config)

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"] if "validation" in dataset else None
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=collator,
            eval_dataset=val_dataset,
            callbacks=callbacks
        )
        trainer.train()
        os.makedirs(os.path.join(self.output_dir, "final_model"), exist_ok=True)
        self.model.save_pretrained(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))

        if self.is_lora:
            if self.merge_adapters:
                base = AutoModelForCausalLM.from_pretrained(self.model_name)
                merged = PeftModel.from_pretrained(base, os.path.join(self.output_dir, "final_model"))
                merged = merged.merge_and_unload()
                merged.save_pretrained(os.path.join(self.output_dir, "merged_adapters"))
                self.tokenizer.save_pretrained(os.path.join(self.output_dir, "merged_adapters"))

        


