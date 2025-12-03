# **Basic Fine-Tuning – Using Upasak TrainingEngine**

This example demonstrates how to use **Upasak** to run a **minimal fine-tuning job** using pre-tokenized datasets.

It shows how to:

1. Load tokenized train/validation datasets
2. Configure `TrainerConfig`
3. Launch a full or LoRA fine-tuning job with `TrainingEngine`
4. Save model checkpoints locally

This script is intended for users who want to fine-tune a model **entirely from Python**, without the Streamlit UI.

---

## **File: `run_basic_finetune.py`**

This example fine-tunes `google/gemma-3-270m-it` using default training parameters on JSON tokenized datasets.

---

## **Prerequisites**

### Install Upasak

```bash
pip install upasak
```

### Tokenized Dataset Required

This script expects:

```
tokenized_train_dataset.json
tokenized_validation_dataset.json
```

You may generate them using the `tokenization.py` example.

---

## **HF Token**

Required only when:

* The model requires authentication

Otherwise set:

```python
hf_token = None
```

---

## **LoRA Fine-Tuning (Optional)**

Enable LoRA by setting:

```python
is_lora=True
```

and providing:

```
lora_r
lora_alpha
lora_dropout
merge_adapters
```

---

## **Summary**

This example helps users:

* Understand direct Python usage of `TrainingEngine`
* Fine-tune any supported Hugging Face model with minimal code
* Run training jobs using pre-tokenized datasets
* Save model weights locally
