from upasak.data_sanitization import PIISanitizer
import json

file_path = "./pii_test_dataset.jsonl"
sanitized_file_path = "./sanitized_output.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
            if line.strip():  
                data.append(json.loads(line))

sanitizer = PIISanitizer(is_nlp_based=False)
sanitized_data = sanitizer.pipeline(records=data, output_file=sanitized_file_path, action_type="redact", enable_hitl=False) # enable_hitl is for streamlit-app