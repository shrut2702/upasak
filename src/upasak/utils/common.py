import json
import csv
import zipfile
import tempfile
import os
from io import BytesIO, StringIO
from typing import List

def load_zip_dataset(uploaded)->List:
    data = []

    zip_bytes = BytesIO(uploaded.getvalue())
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        zip_name = os.path.splitext(os.path.basename(uploaded.name))[0]
        dir_path = os.path.join(tmpdir, zip_name)

        if not os.path.isdir(dir_path):
            raise ValueError(
                f"Expected directory '{zip_name}/' inside the ZIP, but it was not found."
            )
        txt_found = False
        for f in os.listdir(dir_path):
            full_path = os.path.join(dir_path, f)
            if f.endswith(".txt"):
                txt_found = True
                with open(full_path, "r", encoding="utf-8", errors="replace") as fin:
                    text = fin.read().strip()
                    if text:
                        data.append({"text": text})
        if not txt_found:
            raise ValueError(
                f"ZIP file did not contain any .txt files inside '{zip_name}/'."
            )
    return data

def load_uploaded_dataset(uploaded)->List:
    """Reads an uploaded dataset (json, jsonl, csv, zip) and returns list/dict."""
    
    filename = uploaded.name.lower()

    if filename.endswith(".json"):
        return json.load(uploaded)
    elif filename.endswith(".jsonl"):
        lines = uploaded.read().decode("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    elif filename.endswith(".csv"):
        text = uploaded.read().decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        return list(reader)
    elif filename.endswith(".zip"):
        return load_zip_dataset(uploaded)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
def validate_float(value_str: str, field: str, positive: bool = False, non_negative: bool = False):
    """
    Float field validator.
    - No empty / whitespace
    - Must be valid float
    - Optional positivity constraints
    Returns: None (valid) or error string.
    """
    if not value_str or str(value_str).strip() == "":
        return f"{field} cannot be empty."
    try:
        value = float(value_str)
    except ValueError:
        return f"{field} must be a valid number."
    if positive and value <= 0:
        return f"{field} must be > 0."
    if non_negative and value < 0:
        return f"{field} must be ≥ 0."
    return None  


def validate_int(value: str, field: str, min_value: int = 1):
    """
    Integer field validator.
    - Must be integer
    - Must be >= min_value
    Returns: None (valid) or error string.
    """
    if value is None:
        return f"{field} cannot be empty."
    try:
        value = int(value)
    except ValueError:
        return f"{field} must be an integer."
    if value < min_value:
        return f"{field} must be ≥ {min_value}."
    return None  

def validate_hyperparams(
    hf_token: str,
    learning_rate: str,
    batch_size: str,
    epochs: str,
    lr_scheduler_type: str,
    max_seq_len: str,
    logging_steps: str,
    gradient_accumulation_steps: str,
    warmup_ratio: str,
    max_grad_norm: str,
    weight_decay: str,
    save_strategy: str,
    save_steps: str,
    eval_strategy: str,
    eval_steps: str,
    report_to: str,
    model_tracker_api_key: str,
    val_split: str,
    is_lora: bool,
    output_dir: str,
    lora_r: str = None,
    lora_dropout: str = None,
    lora_alpha: str = None,
    selected_target_modules: list[str] = None,
    merge_adapters: bool = None
):
    """
    Validate all fields with simple rules.
    Returns:
        None if all valid
        or single error string describing the first failure.
    """
    # HF token
    if hf_token is None or str(hf_token).strip() == "":
        return "HuggingFace Token cannot be empty."

    # Learning rate
    err = validate_float(learning_rate, "Learning Rate", positive=True)
    if err: return err

    # Batch size
    err = validate_int(batch_size, "Batch Size")
    if err: return err

    # Epochs
    err = validate_int(epochs, "Epochs")
    if err: return err

    # Max sequence length
    err = validate_int(max_seq_len, "Max Sequence Length", min_value=128)
    if err: return err

    # Logging steps
    err = validate_int(logging_steps, "Logging Steps")
    if err: return err

    # Gradient accumulation steps
    err = validate_int(gradient_accumulation_steps, "Gradient Accumulation Steps")
    if err: return err

    # Warmup ratio
    err = validate_float(warmup_ratio, "Warmup Ratio", non_negative=True)
    if err: return err

    # Max grad norm
    err = validate_float(max_grad_norm, "Max Gradient Norm", non_negative=True)
    if err: return err
    
    # Weight decay
    err = validate_float(weight_decay, "Weight decay", non_negative=True)
    if err: return err

    # Save steps
    err = validate_int(save_steps, "Save Steps")
    if err: return err

    # Eval steps
    err = validate_int(eval_steps, "Evaluation Steps")
    if err: return err

    # Validation split
    if eval_strategy in ["steps", "epoch"]:
        if not val_split or str(val_split).strip() == "":
            return f"Validation Split Ratio cannot be empty when Evaluation Strategy is set to 'epoch' or 'steps'."
        try:
            val_split_value = float(val_split)
        except ValueError:
            return f"Validation Split Ratio must be a valid number between 0 and 1 (exclusive)."
        if val_split_value <= 0.0 or val_split_value >= 1.0:
            return "Validation Split Ratio must be between 0 and 1 (exclusive)."
    if val_split is not None and str(val_split).strip() != "":
        err = validate_float(val_split, "Validation Split Ratio", non_negative=True)
        if err: return err
        val_split_value = float(val_split)
        if val_split_value <= 0.0 or val_split_value >= 1.0:
            return "Validation Split Ratio must be between 0 and 1 (exclusive). OR Empty for no validation."
        if eval_strategy not in ["steps", "epoch"]:
            return "Evaluation Strategy cannot be None when validation split is done."

    # Model tracker API key if reporting to wandb or comet_ml
    if report_to in ["wandb", "comet_ml"]:
        if model_tracker_api_key is None or str(model_tracker_api_key).strip() == "":
            return f"API Key cannot be empty when reporting to {report_to}."

    if is_lora:
        err = validate_int(lora_r, "LoRA Rank")
        if err: return err

        err = validate_int(lora_alpha, "LoRA Alpha")
        if err: return err

        err = validate_float(lora_dropout, "LoRA Dropout", non_negative=True)
        if err: return err

        if not selected_target_modules or len(selected_target_modules) == 0:
            return "Please select at least one target module for LoRA."
    
    # Output directory    
    if output_dir is None or str(output_dir).strip() == "":
        return "Output directory path cannot be empty."
    else:
        if not os.path.exists(output_dir):
            return "Please enter a valid path."

    return None  #valid

