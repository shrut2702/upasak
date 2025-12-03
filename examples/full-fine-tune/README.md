

This example demonstrates how to use **Upasak** to perform an **end-to-end workflow**:

1. Load a dataset
2. Tokenize using `TokenizerWrapper`
3. Configure a training job
4. Run full or lora fine-tuning using `TrainingEngine`
5. Save fine-tuned model locally.

This script is intended to show the **minimal code required** when using Upasak as a Python package (outside the Streamlit UI).

**Note:** This will not push your model to Hugging Face Hub.

---

## **File: `tokenization_to_finetuning.py`**

This example fine-tunes the `google/gemma-3-270m-it` model on a custom dataset using default hyperparameters.

---

## **Prerequisites**

### **Install Upasak**

```
pip install upasak
```

### **Supported Files**

* `.json`
* `.jsonl`
* `.csv`
* `.zip` (containing `.txt`)


### **HF Token**

You only need an HF token if:

* Model requires authentication

Otherwise, you can set:

```
hf_token = None
```

---

## **Optional: Enable LoRA**

You can use:

```
is_lora=True
```

and add:

```
lora_rank, lora_alpha, lora_dropout, target_modules
```

---

## **Supported Schemas (Auto-Detected)**

Upasak will automatically detect and convert any of the following:

* ALPACA (instruction, output)
* DAPT (text)
* CHATML (messages)
* SHAREGPT (conversations)
* PROMPT/RESPONSE
* QA
* QLA (long answer)

No manual formatting needed.

---

## **Summary**

This example script is designed to help users:
* Understand Upasak's standalone Python usage
* Launch a full fine-tuning job with minimal code

