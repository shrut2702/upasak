# **Tokenization Example – Using Upasak TokenizerWrapper**

This example demonstrates how to use **Upasak** for **standalone dataset tokenization** before fine-tuning.

It shows how to:

1. Load any supported dataset file
2. Tokenize using `TokenizerWrapper`
3. Optionally split into train/validation
4. Save tokenized outputs to disk

This example is ideal when you want to preprocess data **outside** the Streamlit UI or before sending datasets to a custom training pipeline.

---

## **File: `tokenization.py`**

This script tokenizes a dataset using the tokenizer of `google/gemma-3-270m-it`, applies a maximum sequence length, and performs an automatic train/validation split.

---

## **Prerequisites**

### **Install Upasak**

```bash
pip install upasak
```

---

## **Supported Input Files**

Upasak automatically supports:

* `.json`
* `.jsonl`
* `.csv`
* `.zip` (containing `.txt` files)

These schema formats: Alpaca, DAPT, ChatML, QA, PROMPT/RESPONSE, ShareGPT, and QLA are auto-detected and converted internally.


---

## **Notes**

* Setting `split=None` disables train/validation splitting.
* Returns a dictionary with `"train"` and optionally `"validation"` datasets.
* Tokenized datasets can be saved as JSON or passed directly into `TrainingEngine`.

---

## **Summary**

This example helps users:

* Run Upasak tokenization entirely from Python
* Perform dataset splitting and tokenization in one line
* Prepare clean tokenized datasets for fine-tuning workflows
