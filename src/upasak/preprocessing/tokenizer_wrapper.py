import tempfile, zipfile
from functools import partial
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import List

class TokenizerWrapper:
    def __init__(self, model_name: str, hf_token: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if model_name.endswith("-it"):
            self.chat_template_tokenizer = self.tokenizer
        else:
            temp_model_name = model_name.replace("-pt", "-it") if model_name.endswith("-pt") else model_name + "-it"
            self.chat_template_tokenizer = AutoTokenizer.from_pretrained(temp_model_name, token=hf_token)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<eos>"})
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({"bos_token": "<bos>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def detect_schema(self, dataset: Dataset)->str:
        cols = dataset.column_names
        features = dataset.features

        valid_schema = False

        if 'text' in cols:
            if isinstance(dataset[0]['text'], str):
                valid_schema = True
                dataset_type = 'DAPT'
        elif {"instruction", "output"}.issubset(cols):
            valid_schema = True
            dataset_type = 'ALPACA'
        elif 'messages' in cols:
            msg_feature = features['messages']
            if hasattr(msg_feature, 'feature') and isinstance(msg_feature.feature, dict):
                inner_keys = msg_feature.feature.keys()
                if set(inner_keys) == {"role", "content"}:
                    valid_schema = True
                    dataset_type = 'CHATML'
        elif 'conversations' in cols:
            msg_feature = features['conversations']
            if hasattr(msg_feature, 'feature') and isinstance(msg_feature.feature, dict):
                inner_keys = msg_feature.feature.keys()
                if set(inner_keys) == {"from", "value"}:
                    valid_schema = True
                    dataset_type = 'SHARE_GPT'
        elif {"prompt", "response"}.issubset(cols):
            valid_schema = True
            dataset_type = "PROMPT_RESPONSE"
        elif {"question", "answer"}.issubset(cols):
            valid_schema = True
            dataset_type = "QA"
        elif {"question", "long_answer"}.issubset(cols):
            valid_schema = True
            dataset_type = "QLA"
        if not valid_schema:
            raise ValueError("Unsupported dataset schema.")
        return dataset_type

    def load_zip_dataset(self, file_path: str)->Dataset:
        data = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            zip_name = os.path.splitext(os.path.basename(file_path))[0]
            dir_path = os.path.join(tmpdir, zip_name)
            if not os.path.isdir(dir_path):
                raise ValueError(
                    f"Expected directory '{zip_name}/' inside the ZIP, but it was not found."
                )
            txt_found = False
            for f in os.listdir(dir_path):
                if f.endswith(".txt"):
                    txt_found = True
                    with open(os.path.join(dir_path, f), "r", encoding="utf-8", errors="replace") as fin:
                        text = fin.read().strip()
                        if text:
                            data.append({"text": text})
            if not txt_found:
                raise ValueError(
                    f"ZIP file did not contain any .txt files inside '{zip_name}/'."
                )
        dataset = Dataset.from_list(data, split="train")
        return dataset
    
    def load_file(self, file_path: str)->Dataset:
        if not os.path.exists(file_path):
            raise ValueError("File path does not exist.")

        file_extension = os.path.splitext(file_path)[1]

        if file_extension == '.json':
            dataset = load_dataset('json', data_files=file_path, split='train')
        elif file_extension == '.jsonl':
            dataset = load_dataset('json', data_files=file_path, split='train')
        elif file_extension == '.csv':
            dataset = load_dataset("csv", data_files=file_path, split="train")
        elif file_extension == '.zip':
            dataset = self.load_zip_dataset(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        return dataset
    
    def preprocess(self, example, dataset_type: str, max_length: int, overlap: float = 0):
        if dataset_type == 'CHATML':
            formatted_text = self.chat_template_tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
            if not formatted_text.endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            tokenized_example = self.tokenizer(formatted_text, truncation=True, padding=False, max_length=max_length, padding_side='right', add_special_tokens=False)
        elif dataset_type == 'SHARE_GPT':
            for d in example['conversations']:
                role = d.pop("from")
                content = d.pop("content")
                if role.lower() in ["human", "user"]:
                    d["role"] = "user"
                elif role.lower() in ["system"]:
                    d["role"] = "system"
                else:
                    d["role"] = "assistant"
                d['content'] = "" if content is None else content 
            formatted_text = self.chat_template_tokenizer.apply_chat_template(example['conversations'], tokenize=False, add_generation_prompt=False)
            if not formatted_text.endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            tokenized_example = self.tokenizer(formatted_text, truncation=True, padding=False, max_length=max_length, padding_side='right', add_special_tokens=False)
        elif dataset_type == 'ALPACA':
            user_content = example.get('instruction', '')
            if 'input' in example and example['input'].strip() != '':
                user_content += f"\n\nInput: {example['input']}"
            output = example.get('output', '')
            messages = [
                {"role": "user", "content": "" if user_content is None else user_content},
                {"role": "assistant", "content": "" if output is None else output}
            ]
            formatted_text = self.chat_template_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not formatted_text.endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            tokenized_example = self.tokenizer(formatted_text, truncation=True, padding=False, max_length=max_length, padding_side='right', add_special_tokens=False)
        elif dataset_type == 'PROMPT_RESPONSE':
            messages = [
                {"role": "user", "content": "" if example.get('prompt', '') is None else example.get('prompt', '')},
                {"role": "assistant", "content": "" if example.get('response', '') is None else example.get('response', '')}
            ]
            formatted_text = self.chat_template_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not formatted_text.endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            tokenized_example = self.tokenizer(formatted_text, truncation=True, padding=False, max_length=max_length, padding_side='right', add_special_tokens=False)
        elif dataset_type == 'QA':
            messages = [
                {"role": "user", "content": "" if example.get('question', '') is None else example.get('question', '')},
                {"role": "assistant", "content":"" if example.get('answer', '') is None else example.get('answer', '')}
            ]
            formatted_text = self.chat_template_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not formatted_text.endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            tokenized_example = self.tokenizer(formatted_text, truncation=True, padding=False, max_length=max_length, padding_side='right', add_special_tokens=False)
        elif dataset_type == 'QLA':
            messages = [
                {"role": "user", "content": "" if example.get('question', '') is None else example.get('question', '')},
                {"role": "assistant", "content": "" if example.get('long_answer', '') is None else example.get('long_answer', '')}
            ]
            formatted_text = self.chat_template_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not formatted_text.endswith(self.tokenizer.eos_token):
                formatted_text += self.tokenizer.eos_token
            tokenized_example = self.tokenizer(formatted_text, truncation=True, padding=False, max_length=max_length, padding_side='right', add_special_tokens=False)
        elif dataset_type == 'DAPT':
            merged_texts = []
            for i, text in enumerate(example.get("text", "")): 
                if i != 0 and not text.startswith(self.tokenizer.bos_token): # Add BOS token if missing
                    text = self.tokenizer.bos_token + text
                if not text.endswith(self.tokenizer.eos_token): # Add EOS token if missing
                    text = text + self.tokenizer.eos_token
                merged_texts.append(text)
            merged_text = "\n".join(merged_texts) # for dapt we will use batch mapping
            tokens = self.tokenizer(merged_text, return_attention_mask=False)["input_ids"]
            if max_length is None:
                max_length = 512
            total_len = (len(tokens) // max_length) * max_length
            input_ids = [tokens[i:i+max_length] for i in range(0, total_len, max_length - int(max_length * overlap))]
            attention_mask = [[1]*max_length for _ in input_ids]
            tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask} # this is batched tokenized output
        else:
            raise ValueError("Unsupported dataset format")
        return tokenized_example
    
    def pipeline(self, file_path: str | None = None, data: List[dict] | None = None, max_length: int = 512, split: float = None, overlap: float = 0):
        """
        Unified dataset processing pipeline for model fine-tuning and pretraining.

        This method loads a dataset (from a file or an in-memory list), detects its
        schema automatically, converts it into a standardized chat or text format,
        and tokenizes it according to the rules of the detected dataset type
        (e.g., DAPT, Alpaca, ChatML, ShareGPT, QA, prompt/response).

        It handles:
        - File-based datasets (.json, .jsonl, .csv, .zip with .txt files)
        - Raw in-memory Python lists of dictionaries (`data`)
        - Automatic schema detection
        - Chat template formatting using the appropriate tokenizer
        - Sliding-window chunking for DAPT datasets
        - Train/validation splitting

        Parameters
        ----------
        file_path : str, optional
            Path to a dataset file. Supports:
            - `.json`, `.jsonl` (HuggingFace JSON loader)
            - `.csv`
            - `.zip` containing a folder of `.txt` files
            Cannot be used together with `data`.

        data : List[dict], optional
            Dataset provided directly as a Python list of dictionaries.
            Useful for dynamically built datasets or API-supplied data.
            Cannot be used together with `file_path`.

        max_length : int, default=512
            Maximum number of tokens per sequence. Examples longer than this
            are truncated (SFT/QnA/chat formats) or split into sliding-window
            chunks (DAPT).

        split : float, optional
            Proportion of the dataset to use as a validation/test split.
            Example: `0.1` creates 90% train / 10% validation.
            Must be between 0 and 1. If omitted, no split is performed.

        overlap : float, default=0
            Fraction of overlap (0.0–1.0) between consecutive DAPT chunks
            when sliding-window tokenization is applied.
            Ignored for structured SFT/chat datasets.

        Returns
        -------
        dict
            A dictionary containing tokenized HuggingFace `Dataset` objects:
            - If `split` is provided:
                {
                    "train": tokenized_train_dataset,
                    "validation": tokenized_val_dataset
                }
            - If `split` is omitted:
                {
                    "train": tokenized_dataset
                }

        Supported Schemas (Auto-Detected)
        ---------------------------------
        1. **DAPT**
        - Columns: `["text"]`
        - Multi-document text, chunked into token windows.

        2. **ALPACA**
        - Columns: `["instruction", "output"]` (optional: `"input"`)
        - Turns converted to:
            user → assistant

        3. **CHATML**
        - Columns: `["messages"]` with:
            {"role": ..., "content": ...}

        4. **SHARE_GPT**
        - Columns: `["conversations"]` with:
            {"from": ..., "value": ...}
        - Automatically remaps:
            human/user → role:user
            assistant/model → role:assistant

        5. **PROMPT_RESPONSE**
        - Columns: `["prompt", "response"]`

        6. **QA**
        - Columns: `["question", "answer"]`

        7. **QLA** (long answer)
        - Columns: `["question", "long_answer"]`

        Processing Behavior
        -------------------
        - DAPT datasets use **batched=True** with sliding-window chunking.
        - Chat/SFT datasets use **batched=False** to preserve formatting.
        - All chat formats are passed through `apply_chat_template`.
        - EOS/BOS/PAD tokens are added if missing.
        - Dataset columns are normalized to lowercase for compatibility.

        Raises
        ------
        ValueError
            If neither or both of `file_path` and `data` are provided.
            If the dataset schema is unsupported.
            If `split` is not within (0, 1).

        Notes
        -----
        This method provides a one-stop preprocessing solution for most
        LLM fine-tuning workflows, abstracting schema differences, chat
        formatting, and tokenization details.
        """
        if file_path is None and data is None:
            raise ValueError("You must provide either `file_path` or `data`.")
        if file_path is not None and data is not None:
            raise ValueError("Provide only one of `file_path` or `data`, not both.")
        if file_path is not None:
            dataset = self.load_file(file_path)
        if data is not None:
            dataset = Dataset.from_list(data, split="train")
        dataset = dataset.rename_columns({col: col.lower() for col in dataset.column_names})
        dataset_type = self.detect_schema(dataset)
        preprocess_function = partial(self.preprocess, dataset_type=dataset_type, max_length=max_length, overlap=overlap)
        if split is not None:
            if not (0 < split < 1):
                raise ValueError("`split` must be between 0 and 1 (e.g., 0.1 for 10%).")
            split_dataset = dataset.train_test_split(test_size=split)
            train_dataset = split_dataset["train"]
            val_dataset  = split_dataset["test"]
            if dataset_type == 'DAPT':
                train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=1000, remove_columns=dataset.column_names)
                val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True, batch_size=1000, remove_columns=dataset.column_names)
            else:
                train_tokenized_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=dataset.column_names)
                val_tokenized_dataset = val_dataset.map(preprocess_function, batched=False, remove_columns=dataset.column_names)
            return {"train": train_tokenized_dataset, "validation": val_tokenized_dataset}
        else:
            if dataset_type == 'DAPT':
                tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=1000, remove_columns=dataset.column_names)
            else:
                tokenized_dataset = dataset.map(preprocess_function, batched=False, remove_columns=dataset.column_names)
            return {"train": tokenized_dataset}
        
            