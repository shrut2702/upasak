from upasak.preprocessing import TokenizerWrapper

model_name = "google/gemma-3-270m-it"
hf_token = "your_hf_access_token"
dataset_path = "./dummy_dataset.json"

tokenizer = TokenizerWrapper(model_name, hf_token=hf_token)
tokenized_dataset = tokenizer.pipeline(file_path=dataset_path, max_length=128, split= 0.5) # returns a dict with "train" and "validation" keys and respective dataset, "validation" only if split is not None

tokenized_dataset["train"].to_json("./tokenized_train_dataset.json")
tokenized_dataset["validation"].to_json("./tokenized_validation_dataset.json")
