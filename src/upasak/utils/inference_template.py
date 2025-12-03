import os
import json

def generate_inference_script(config, output_file_name="inference.py"):
    """
    Generate a customized inference.py based on training configuration.
    
    Args:
        config (dict): Training configuration containing:
            - model_name: Model used
            - output_dir: Path to saved model
            - is_lora: Whether LoRA was used
            - merge_adapters: Whether adapters were merged
        output_file_name (str): Path to save the generated inference.py
    """
    
    # Extract config values with defaults
    model_name = config.model_name
    output_dir = config.output_dir
    is_lora = config.is_lora
    merge_adapters = config.merge_adapters

    if not is_lora or (is_lora and not merge_adapters):
        model_path = "./final_model"
    else:
        model_path = "./merged_adapters"
    
    # Generate the inference script
    script_content = f'''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
'''
    
    # Add LoRA imports if needed
    if is_lora and not merge_adapters:
        script_content += '''from peft import PeftModel

'''
    
    script_content += f'''def generate_response(prompt, model_path, model_name, max_length=200):
    """
    Generates a response from the fine-tuned model.

    Args:
        prompt (str): The input text prompt.
        model_path (str): Path to the fine-tuned model directory.

    Returns:
        str: The generated text.
    """
'''
    if not is_lora or (is_lora and merge_adapters):
        script_content += f'''
    try:
        print(f"Loading model from {{model_path}}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
        )
    except Exception as e:
        return f"Error loading model: {{e}}"
'''

    
    # Add LoRA loading logic
    if is_lora and not merge_adapters:
        script_content += f'''
    try:
        print(f"Loading base model {{model_name}}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        print(f"Loading adapters from {{model_path}}...")
        model = PeftModel.from_pretrained(model, model_path)
    except Exception as e:
        return f"Error loading model: {{e}}"
'''
    
    
    script_content += '''    
    use_chat_template = True
    if use_chat_template:
        message = [
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
'''
    
    script_content += f'''
    print("Generating response...")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(outputs.squeeze()[len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
    return response
'''

    script_content += f'''    
if __name__ == "__main__":    
    model_path = "{model_path}"
    model_name = "{model_name}"
    print("--- Fine-tuned Model Inference ---")
    user_prompt = input("Enter your prompt: ")
    
    response = generate_response(user_prompt, model_path=model_path, model_name=model_name)
    
    print("\\n--- Generated Response ---")
    print(response)
'''
    
    # Write to file
    output_path = os.path.join(output_dir, output_file_name)
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"Inference script generated: {output_path}")
    return script_content