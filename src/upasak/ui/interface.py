import streamlit as st
import pandas as pd
import json
import os
import time
from typing import List, Tuple, Optional
import threading
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainingArguments, TrainerControl, TrainerState, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import importlib.resources as res

from ..fine_tune import TrainerConfig, TrainingEngine
from ..preprocessing import TokenizerWrapper
from ..data_sanitization import PIISanitizer
from ..utils import load_uploaded_dataset, validate_hyperparams, generate_inference_script

##############################
### CALLBACKS FOR TRAINING ###
##############################
class StreamlitStopCallback(TrainerCallback):
    def __init__(self, stop_dict, training_status):
        self.stop_dict = stop_dict
        self.training_status = training_status

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.training_status["active"] = True
        return control
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.stop_dict.get("stop", False):
            self.training_status["active"] = False
            control.should_training_stop = True
            print(f"Stopping training at step {state.global_step}")
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.training_status["active"] = False
        print("Training ended")
        return control

class StreamlitPlotCallback(TrainerCallback):
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_dict["train_history"].append({
                "step": state.global_step,
                "loss": logs["loss"],
                "tokens": state.num_input_tokens_seen
            })
            self.loss_dict["total_tokens_seen"] = state.num_input_tokens_seen
            self.loss_dict["last_update"] = time.time()
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict=None, **kwargs):
        """Runs whenever evaluation happens (manual or eval_steps)."""
        if metrics and "eval_loss" in metrics:
            self.loss_dict["eval_history"].append({
                "step": state.global_step,
                "eval_loss": metrics["eval_loss"],
            })
            self.loss_dict["last_update"] = time.time()
        return control
    
class ExperimentTrackerCallback(TrainerCallback):
    def __init__(self, experiment_tracker):
        self.experiment_tracker = experiment_tracker

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs): # currently not working
        trainer = kwargs.get("trainer")
        if trainer and hasattr(trainer, "integrations"):
            for integration in trainer.integrations:
                    if integration.__class__.__name__ == "CometIntegration":
                        exp = integration.experiment
                        if exp is not None:
                            self.experiment_tracker["url"] = exp.url
        return control
    

def train_in_background(config, tokenized_dataset: dict[str, Dataset], hf_token: str, generate_inference: bool, stop_dict: dict, training_status: dict, loss_dict: dict, experiment_tracker: dict):
    try:
        print("Starting training in background thread...")
        trainer = TrainingEngine(config, hf_token=hf_token)
        
        stop_callback = StreamlitStopCallback(stop_dict, training_status)
        plot_callback = StreamlitPlotCallback(loss_dict)
        experiment_tracker_callback= ExperimentTrackerCallback(experiment_tracker)
        
        trainer(tokenized_dataset, callbacks=[stop_callback, plot_callback, experiment_tracker_callback])
        
        training_status["active"] = False
        if not stop_dict.get("stop"):
            loss_dict["training_complete"] = True
            print("Training completed successfully!")
    except Exception as e:
        loss_dict["error"] = str(e)
        training_status["active"] = False
        print(f"Training error: {e}")

    try:
        if generate_inference and loss_dict.get("training_complete") == True:
            script = generate_inference_script(config)
            print("Successfully generated inference.py at {config.output_dir}.")
    except Exception as e:
        print(f"Error {e}")

def plot_losses(train_history: list, eval_history: list, title: str="Training Loss"):
    """Create a matplotlib figure for training loss"""
    if not train_history and not eval_history:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))

    if train_history:
        train_steps = [d["step"] for d in train_history]
        train_losses = [d["loss"] for d in train_history]
        tokens = [d.get("tokens", None) for d in train_history]
        has_all_tokens = all(t is not None for t in tokens)
    
        ax.plot(train_steps, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label="Train Loss")
    else:
        has_all_tokens = False
        tokens = []

    if eval_history:
        eval_steps = [d["step"] for d in eval_history]
        eval_losses = [d["eval_loss"] for d in eval_history]

        ax.plot(eval_steps, eval_losses, 'ro-', linewidth=2, markersize=5, label="Eval Loss")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if has_all_tokens and train_history:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(train_steps)
        ax_top.set_xticklabels([str(t) for t in tokens], rotation=45, ha='left', fontsize=9)
        ax_top.set_xlabel("Tokens (per log step)", fontsize=12)
    
    return fig

def monitor_training_with_live_plot(stop_dict: dict, training_status: dict, loss_dict: dict, experiment_tracker: dict):
    """Monitor training and update plot in real-time"""
    status_placeholder = st.empty()
    plot_placeholder = st.empty()
    col_steps, col_tokens = st.columns(2)
    with col_steps:
        step_metrics_placeholder = st.empty()
    with col_tokens:
        tokens_metrics_placeholder = st.empty()
    
    last_plot_update = 0
    plot_update_interval = 2
    
    while True:
        url = experiment_tracker.get('url', None)
        status_placeholder.info(f"Training in progress.{'' if url is None else f'Track your experiment here: [{url}]({url})'}")

        if loss_dict.get("training_complete", False):
            fig = plot_losses(
                loss_dict.get("train_history", []),
                loss_dict.get("eval_history", []),
                "Training & Evaluation Loss (Completed)"
            )
            if fig:
                plot_placeholder.pyplot(fig, width=700)
                plt.close(fig)
            break
        
        if "error" in loss_dict:
            status_placeholder.error(f"Training error: {loss_dict['error']}")
            break
        
        current_time = time.time()
        if current_time - last_plot_update >= plot_update_interval:
            train_history = loss_dict.get("train_history", [])
            eval_history = loss_dict.get("eval_history", [])
            
            if train_history or eval_history:
                fig = plot_losses(train_history, eval_history, "LLM Loss (Live)")
                plot_placeholder.pyplot(fig, width=700)
                plt.close(fig)
                
                if train_history:
                    latest = train_history[-1]
                    step_metrics_placeholder.metric(
                        label="Current Loss",
                        value=f"{latest['loss']:.4f}",
                        delta=f"Step {latest['step']}"
                    )
                    tokens_metrics_placeholder.metric(
                        label="Total Tokens Seen",
                        value=f"{loss_dict['total_tokens_seen']}"
                    )
                
                last_plot_update = current_time
        
        time.sleep(0.5)

#########################
### PII HITL UI LOGIC ###
#########################
def hitl_ui(detections: List[dict]):
    """Display HITL UI for PII verification"""
    if "hitl_done" not in st.session_state:
        st.session_state.hitl_done = False
    
    st.subheader("Human Verification")
    df = pd.DataFrame(detections)
    df["discard"] = False
    edited = st.data_editor(
        df,
        column_config={
            "discard": st.column_config.CheckboxColumn("Discard?", help="Check to consider this detection as a false positive and exclude it from sanitization."),
            "sample_id": st.column_config.NumberColumn("Sample ID", width="medium"),
            "score": st.column_config.NumberColumn("Confidence Score", format="%.2f", width="medium"),
            "text_key": st.column_config.TextColumn("Field", width="medium"),
            "pattern": st.column_config.TextColumn("Pattern", width="medium"),
            "match": st.column_config.TextColumn("Match", width="medium"),
            "start": None,
            "end": None
        },
        use_container_width=True,
        hide_index=True,
    )
    
    if st.button("Verified"):
        st.session_state.hitl_done = True
        st.session_state.verified_output = edited[edited["discard"] == False].drop("discard", axis=1).to_dict(orient="records")
    
    if not st.session_state.hitl_done:
        st.warning("Please review and click **Verified** to continue.")
        st.stop()
    
    return st.session_state.verified_output

############################
### UI COMPONENT FUNCTIONS ###
############################

def render_sidebar(disabled: bool = False) -> Tuple[str, bool, str]:
    """Render sidebar with model configuration"""
    with st.sidebar:
        st.header("Model Configuration")
        model_name = st.selectbox(
            "Select Model",
            ["google/gemma-3-270m", "google/gemma-3-270m-it", 
             "google/gemma-3-1b-pt", "google/gemma-3-1b-it", 
             "google/gemma-3-4b-pt", "google/gemma-3-4b-it"],
            index=0,
            disabled=disabled
        )
        
        st.markdown("---")
        st.subheader("Hugging Face Hub")
        hf_token = st.text_input(
            "HF Token", 
            type="password", 
            help="Your Hugging Face access token with 'write' permissions.",
            disabled=disabled
        )
    
    return model_name, hf_token

def render_file_upload(disabled: bool = False):
    """Render file upload section"""
    uploaded = st.file_uploader(
        "Upload Dataset (.json, .jsonl, .csv, .zip)", 
        type=["json", "jsonl", "csv", "zip"],
        help="Upload your training dataset in JSON, JSONL, CSV, or ZIP (containing text files only for corpora trainig).",
        disabled=disabled
    )
    uploaded_message_placeholder = st.empty()
    return uploaded, uploaded_message_placeholder

def render_hf_dataset_select(disabled: bool = False):
    """Render Hugging Face dataset dropdown"""
    available_datasets = [
        "None",
        "mlabonne/FineTome-100k",
        "mlabonne/MedQuad-MedicalQnADataset",
        "lavita/MedQuAD",
        "microsoft/wiki_qa",
        "qiaojin/PubMedQA",
        "axiong/pmc_llama_instructions",
        "HuggingFaceH4/ultrachat_200k",
        "HuggingFaceFW/fineweb"
    ]
    selected_dataset = st.selectbox("Select a dataset:", available_datasets, index=0, help="Select a dataset from Hugging Face", disabled=disabled)
    selected_message_placeholder = st.empty()
    return selected_dataset, selected_message_placeholder

def render_pii_section(uploaded, hf_hub_dataset: str, disabled: bool = False) -> Tuple[bool, Optional[str], bool, bool, float, int]:
    """Render PII sanitization section"""
    sanitize = st.toggle(
        "Enable PII Sanitization",
        help="Protects sensitive information in your data by detecting and hiding personal details such as names, phone numbers, emails and other personal and sensitive information.",
        disabled=disabled
    )
    
    if not sanitize:
        return False, None, False, False, 0.3, 10
    
    col1, col2 = st.columns(2)
    with col1:
        is_nlp_based = st.checkbox("AI based detection", help="Automatically find sensitive information using NLP models.", disabled=disabled)
    with col2:
        is_hitl = st.checkbox("Human Review", help="Manually verify uncertain detections before applying sanitization.", disabled=disabled)
    
    action_type = st.selectbox(
        "Choose Sanitization Method",
        ["Redact", "Mask"],
        help=(
                "Choose how detected PII should be hidden in the final output.\n\n"
                "**Redact:** Replace the exact text with a generic placeholder like [REDACTED].\n\n"
                "**Mask:** Replace the text with a category label such as [PHONE_NO], [EMAIL], etc., "
            ),
        disabled=disabled
    )
    
    hitl_ratio = 0.3
    max_hitl = 10
    if is_hitl:
        hitl_ratio = st.slider("HITL Ratio", 0.0, 1.0, 0.3, disabled=disabled)
        max_hitl = st.number_input("Max HITL samples", min_value=1, step=1, value=10, disabled=disabled)
    
    start_pii = st.button("Start PII Sanitization", disabled=disabled or not ((uploaded is not None) ^ (hf_hub_dataset is not None)))
    
    if start_pii:
        st.session_state.start_pii = True
        st.session_state.pii_results = None
        st.session_state.pii_detections = None
        st.session_state.hitl_done = False
    
    return sanitize, action_type, is_nlp_based, is_hitl, hitl_ratio, max_hitl

def render_hyperparameters(disabled: bool = False) -> dict:
    """Render hyperparameters section and return all values"""
    st.header("Hyperparameters")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        learning_rate = st.text_input("Learning Rate", value="0.0002", help="Step size for optimizer during training.", disabled=disabled)
        lr_scheduler_type = st.selectbox(
            "LR Scheduler Type",
            ["linear", "cosine", "constant", "inverse_sqrt"],
            index=0,
            help="Type of learning rate scheduler to use during training.",
            disabled=disabled
        )
    with col4:
        batch_size = st.text_input("Batch Size", value="4", help="Number of samples in each training batch.", disabled=disabled)
        max_seq_len = st.text_input("Max Sequence Length", value="128", help="Maximum number of tokens per training sample.", disabled=disabled)
    with col5:
        epochs = st.text_input("Epochs", value="3", help="Number of times the entire dataset is passed through the model during training.", disabled=disabled)
        logging_steps = st.text_input("Logging Steps", value="10", help="Frequency (in steps) of logging training metrics.", disabled=disabled)
    
    with st.expander("Advanced Hyperparameters"):
        col6, col7, col8 = st.columns(3)
        with col6:
            gradient_accumulation_steps = st.text_input("Gradient Accumulation Steps", value="1", help="Simulate larger batch sizes by accumulating gradients over multiple steps.", disabled=disabled)
            max_grad_norm = st.text_input("Gradient Clipping Norm", value="0.0", help="0.0 means no clipping.", disabled=disabled)
            eval_strategy = st.selectbox("Evaluation Strategy", ["no", "steps", "epoch"], index=0, help="When to run evaluation during training.", disabled=disabled)
            report_to = st.selectbox("Model Tracking Platform", ["none", "wandb", "comet_ml"], index=0, help="Destination for logging training metrics.", disabled=disabled)
        with col7:
            warmup_ratio = st.text_input("Learning Rate Warmup Ratio", value="0.0", help="Proportion of training steps to linearly increase the learning rate from 0 to the set value.", disabled=disabled)
            weight_decay = st.text_input("Weight Decay", value="0.0", help="Regularization technique to prevent overfitting by penalizing large weights.", disabled=disabled)
            eval_steps = st.text_input("Evaluation Steps", value="500", help="Number of steps between evaluations if 'steps' strategy is selected.", disabled=disabled)
            model_tracker_api_key = st.text_input("Model Tracker API Key", value="", type="password", help="API key for the selected model tracking platform.", disabled=disabled)
        with col8:
            save_strategy = st.selectbox("Checkpoint Save Strategy", ["no", "epoch", "steps"], index=0, help="When to save model checkpoints during training.", disabled=disabled)
            save_steps = st.text_input("Checkpoint Save Steps", value="500", help="Number of steps between saving model checkpoints if 'steps' strategy is selected.", disabled=disabled)
            val_split = st.text_input("Validation Split Ratio", value=None, help="Proportion of the dataset to use for validation.", disabled=disabled)
    
    return {
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "epochs": epochs,
        "logging_steps": logging_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "val_split": val_split,
        "report_to": report_to,
        "model_tracker_api_key": model_tracker_api_key
    }

def render_lora_section(disabled: bool = False) -> Tuple[bool, dict]:
    """Render LoRA configuration section"""
    is_lora = st.toggle("Enable LoRA", help="mention info", disabled=disabled)
    
    lora_config = {}
    if is_lora:
        col9, col10 = st.columns(2)
        with col9:
            lora_config["lora_r"] = st.text_input("Rank", value="16", help="LoRA rank hyperparameter.", disabled=disabled)
            lora_config["lora_dropout"] = st.text_input("Dropout", value="0.0", help="Dropout rate for LoRA layers.", disabled=disabled)
        with col10:
            lora_config["lora_alpha"] = st.text_input("Alpha", value="32", help="LoRA alpha hyperparameter.", disabled=disabled)
            available_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            lora_config["selected_target_modules"] = st.multiselect(
                "Select LoRA Target Modules",
                options=available_modules,
                default=["q_proj", "k_proj", "v_proj", "o_proj"],
                help="Choose which model layers LoRA will be applied to.",
                disabled=disabled
            )
        lora_config["merge_adapters"] = st.checkbox("Merge Adapters", value=False, help="Merge LoRA adapters into base model after training.", disabled=disabled)
    
    st.caption("Note: Larger batch sizes require more GPU memory.")
    
    return is_lora, lora_config

def render_training_controls(disabled: bool = False) -> Tuple[str, bool, bool]:
    """Render training section with start/stop buttons"""
    st.header("Training")
    output_dir = st.text_input("Output Directory Path", value=None, disabled=disabled)
    
    generate_inference = st.checkbox("Generate inference.py", value=True, help="Generate a simple inference script after training.", disabled=disabled)

    col_start, col_stop = st.columns(2)
    with col_start:
        start_training = st.button(
            "Start Training", 
            type="primary", 
            disabled=disabled or st.session_state.training_started
        )  
    with col_stop:
        stop_training = st.button(
            "Stop Training", 
            type="secondary", 
            disabled=not st.session_state.training_started
        )
        if stop_training:
            st.session_state.stop_dict["stop"] = True
            st.session_state.training_started = False 
            st.warning("Stop Signal Sent!")
              
    return output_dir, start_training, stop_training, generate_inference

def render_hub_push_section(hf_token: str, disabled: bool = False):
    """Render Hugging Face Hub push section"""
    st.header("Push Fine-Tuned Model to Hugging Face Hub")
    
    hf_username = ""
    hf_repo_name = ""
    output_message = {"status": None, "message": None}
    col_11, col_12 = st.columns(2)
    with col_11:
        hf_username = st.text_input("Hugging Face Username", disabled=disabled, help="Your Hugging Face username.")
    with col_12:
        hf_repo_name = st.text_input("Hugging Face Repository Name", disabled=disabled, help="Name of the repository to push the model to.")

    if st.button("Push to Hub", disabled=disabled):
        st.session_state.push_to_hub = True
        st.rerun()

    if st.session_state.push_to_hub:
        if st.session_state.loss_dict.get("training_complete", True):
            if not hf_token:
                output_message["status"] = "error"
                output_message["message"] = "Please enter your Hugging Face token in the sidebar."
                st.session_state.push_hub_msg.update(output_message)
                st.session_state.push_to_hub = False
                st.rerun()
            elif not hf_username:
                output_message["status"] = "error"
                output_message["message"] = "Please enter your Hugging Face username."
                st.session_state.push_hub_msg.update(output_message)
                st.session_state.push_to_hub = False
                st.rerun()
            elif not hf_repo_name:
                output_message["status"] = "error"
                output_message["message"] = "Please enter the Hugging Face repository name."
                st.session_state.push_hub_msg.update(output_message)
                st.session_state.push_to_hub = False
                st.rerun()
            else:
                if "current_config" not in st.session_state:
                    output_message["status"] = "error"
                    output_message["message"] = "Please finish model training."
                    st.session_state.push_hub_msg.update(output_message)
                    st.session_state.push_to_hub = False
                    st.rerun()
                else:
                    try:
                        with st.spinner("Pushing..."):
                            repo_id = f"{hf_username}/{hf_repo_name}"
                            if st.session_state.current_config.merge_adapters and st.session_state.current_config.is_lora:
                                model_path = os.path.join(st.session_state.current_config.output_dir, "merged_adapters")
                            else:
                                model_path = os.path.join(st.session_state.current_config.output_dir, "final_model")
                
                            #loading saved model
                            model = AutoModelForCausalLM.from_pretrained(model_path, token=hf_token)
                            tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
                            
                            #pushing to hub
                            push_obj = tokenizer.push_to_hub(repo_id, token=hf_token)
                            push_obj = model.push_to_hub(repo_id, token=hf_token)
                        
                        #show URL
                        repo_url = push_obj.repo_url.url
                        success_msg = f"Model pushed to: [{repo_url}]({repo_url})"
                        output_message["status"] = "success"
                        output_message["message"] = success_msg
                        st.session_state.push_hub_msg.update(output_message)
                        st.session_state.push_to_hub = False
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error pushing model to Hub: {e}"
                        output_message["status"] = "error"
                        output_message["message"] = error_msg
                        st.session_state.push_hub_msg.update(output_message)
                        st.session_state.push_to_hub = False
                        st.rerun()
        else:
            output_message["status"] = "error"
            output_message["message"] = "Please finish model training."
            st.session_state.push_hub_msg.update(output_message)
            st.session_state.push_to_hub = False
            st.rerun()

################################
### MAIN STREAMLIT APP LOGIC ###
################################
def main():
    logo_path = res.files(__package__).joinpath("assets/upasak_logo.png")
    st.set_page_config(page_title="Upasak", page_icon=logo_path, layout="wide")
    st.title("Upasak - LLM Fine-Tuning")
    
    # Initialize session state
    if "stop_dict" not in st.session_state:
        st.session_state.stop_dict = {"stop": False}
    if "loss_dict" not in st.session_state:
        st.session_state.loss_dict = {"train_history": [], "eval_history": [], "training_complete": False}
    if "training_status" not in st.session_state:
        st.session_state.training_status = {"active": False}
    if "training_started" not in st.session_state:
        st.session_state.training_started = False
    if "experiment_tracker" not in st.session_state:
        st.session_state.experiment_tracker = {"url": None}
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "prev_upload_file_id" not in st.session_state:
        st.session_state.prev_upload_file_id = None
    if "prev_hf_dataset" not in st.session_state:
        st.session_state.prev_hf_dataset = None
    if "dataset_mode" not in st.session_state:
        st.session_state.dataset_mode = None
    if "start_pii" not in st.session_state:
        st.session_state.start_pii = False
    if "prev_is_nlp_based" not in st.session_state:
        st.session_state.prev_is_nlp_based = None
    if "push_to_hub" not in st.session_state:
        st.session_state.push_to_hub = False
    if "push_hub_msg" not in st.session_state:
        st.session_state.push_hub_msg = {"status": None, "message": None}
    
    # Determine if UI should be disabled
    is_training = st.session_state.training_started
    is_pushing = st.session_state.push_to_hub
    should_disable = is_training or is_pushing
    
    # Render all UI components
    model_name, hf_token = render_sidebar(disabled=should_disable)
    
    st.header("Data")
    hf_hub_dataset, selected_message_placeholder = render_hf_dataset_select(disabled=should_disable)
    hf_hub_dataset = None if hf_hub_dataset == "None" else hf_hub_dataset
    _, center, _ = st.columns(3)
    with center:
        st.markdown("<h6 style='text-align: center;'>OR</h6>", unsafe_allow_html=True)
    uploaded, uploaded_message_placeholder = render_file_upload(disabled=should_disable)
    sanitize, action_type, is_nlp_based, is_hitl, hitl_ratio, max_hitl = render_pii_section(uploaded, hf_hub_dataset, disabled=should_disable)
    
    # Load dataset
    if uploaded and not hf_hub_dataset:
        try:
            if st.session_state.dataset is None or uploaded.file_id != st.session_state.prev_upload_file_id or st.session_state.dataset_mode != "uploaded":
                st.session_state.prev_upload_file_id = uploaded.file_id
                st.session_state.dataset = load_uploaded_dataset(uploaded)
                st.session_state.dataset_mode = "uploaded" # to track the change in dataset mode for new dataset loading
                st.session_state.start_pii = False # because without this when new dataset was loaded after running pii on older one, the ui would still render results from old pii
                uploaded_message_placeholder.success("Dataset loaded successfully!")
        except Exception as e:
            uploaded_message_placeholder.error(f"Error loading dataset: {e}")
            st.stop()
    elif not uploaded and hf_hub_dataset:
        try:
            if st.session_state.dataset is None or hf_hub_dataset != st.session_state.prev_hf_dataset or st.session_state.dataset_mode != "selected":
                with st.spinner("Loading selected dataset (wait a few minutes)..."):
                    st.session_state.prev_hf_dataset = hf_hub_dataset
                    ds = load_dataset(hf_hub_dataset, split="train")
                    st.session_state.dataset = ds.to_list()
                    st.session_state.dataset_mode = "selected" # to track the change in dataset mode for new dataset loading
                    st.session_state.start_pii = False # because without this when new dataset was loaded after running pii on older one, the ui would still render results from old pii
                selected_message_placeholder.success("Dataset loaded successfully!")
        except Exception as e:
            selected_message_placeholder.error(f"Error loading dataset: {e}")
            st.stop()
    elif uploaded and hf_hub_dataset:
        st.error("Select one source only: upload a file or pick from the dropdown.")
        st.stop()
    
    
    # Handle PII sanitization
    if ((uploaded is not None) ^ (hf_hub_dataset is not None))  and sanitize and st.session_state.get("start_pii"):
        if st.session_state.pii_results is None:
            if "obj" not in st.session_state or (st.session_state.prev_is_nlp_based != is_nlp_based):
                st.session_state.obj = PIISanitizer(is_nlp_based=is_nlp_based)
                st.session_state.prev_is_nlp_based = is_nlp_based
            
            if is_hitl:
                results = st.session_state.obj.pipeline(
                    st.session_state.dataset, 
                    action_type=action_type.lower(), 
                    hitl_ratio=hitl_ratio, 
                    max_hitl=max_hitl, 
                    enable_hitl=is_hitl, 
                    hitl_fn=hitl_ui, 
                    session_state=st.session_state
                )
            else:
                results = st.session_state.obj.pipeline(
                    st.session_state.dataset, 
                    action_type=action_type.lower(), 
                    enable_hitl=is_hitl, 
                    session_state=st.session_state
                )
            st.session_state.pii_results = results
            st.rerun()
        
        st.success("PII Sanitization finished successfully.")
        st.markdown("### Final Sanitized Output Sample")
        st.json(st.session_state.pii_results[:3])
    
    # Hyperparameters
    hyperparams = render_hyperparameters(disabled=should_disable)
    is_lora, lora_config = render_lora_section(disabled=should_disable)
    
    # Training controls
    # Training controls
    output_dir, start_training, _, generate_inference = render_training_controls(disabled=should_disable)
    
    # Handle training start
    if start_training:
        st.session_state.stop_dict["stop"] = False
        st.session_state.loss_dict = {"train_history": [], "eval_history": [], "training_complete": False}
        st.session_state.training_status = {"active": False}
        st.session_state.training_started = True
        
        # Get training dataset
        full_dataset = None
        if ((uploaded is not None) ^ (hf_hub_dataset is not None)) and st.session_state.dataset:
            if sanitize and st.session_state.get("pii_results") is None:
                st.error("Finish PII Sanitization first.")
                st.session_state.training_started = False
                st.stop()
            elif sanitize and st.session_state.pii_results is not None:
                full_dataset = st.session_state.pii_results
            else:
                full_dataset = st.session_state.dataset
        else:
            st.error("Please upload a file OR select a dataset.")
            st.session_state.training_started = False
            st.stop()
        
        # Validate hyperparameters
        validation_params = {
            "hf_token": hf_token,
            "output_dir": output_dir,
            "is_lora": is_lora,
            **hyperparams
        }
        if is_lora:
            validation_params.update(lora_config)
        
        err = validate_hyperparams(**validation_params)
        if err:
            st.error(err)
            st.session_state.training_started = False
            st.stop()
        
        # Create config
        config = TrainerConfig(
            model_name=model_name,
            lr=float(hyperparams["learning_rate"]),
            epochs=int(hyperparams["epochs"]),
            batch_size=int(hyperparams["batch_size"]),
            output_dir=output_dir,
            logging_steps=int(hyperparams["logging_steps"]),
            gradient_accumulation=int(hyperparams["gradient_accumulation_steps"]),
            max_seq_len=int(hyperparams["max_seq_len"]),
            weight_decay=float(hyperparams["weight_decay"]),
            warmup_ratio=float(hyperparams["warmup_ratio"]),
            lr_scheduler_type=hyperparams["lr_scheduler_type"],
            save_strategy=hyperparams["save_strategy"],
            report_to=hyperparams["report_to"],
            model_tracker_api_key=hyperparams["model_tracker_api_key"],
            val_split=float(hyperparams["val_split"]) if hyperparams["val_split"] is not None and hyperparams["val_split"].strip() != "" else None,
            save_steps=int(hyperparams["save_steps"]),
            eval_strategy=hyperparams["eval_strategy"],
            eval_steps=int(hyperparams["eval_steps"]),
            max_grad_norm=float(hyperparams["max_grad_norm"]),
            is_lora=is_lora,
            lora_r=int(lora_config["lora_r"]) if is_lora else None,
            lora_alpha=int(lora_config["lora_alpha"]) if is_lora else None,
            lora_dropout=float(lora_config["lora_dropout"]) if is_lora else None,
            selected_target_modules=lora_config.get("selected_target_modules") if is_lora else None,
            merge_adapters=lora_config.get("merge_adapters", False) if is_lora else False
        )
        st.session_state.current_config = config
        
        # Tokenize dataset
        try:
            with st.spinner("Tokenizing the dataset..."):
                tokenizer = TokenizerWrapper(model_name, hf_token=hf_token)
                tokenized_dataset = tokenizer.pipeline(data=full_dataset, max_length=config.max_seq_len, split= config.val_split)
        except Exception as e:
            st.error(f"Tokenization error: {e}")
            st.session_state.training_started = False
            st.stop()
        
        # Start training thread
        training_thread = threading.Thread(
            target=train_in_background,
            args=(config, tokenized_dataset, hf_token, generate_inference,
                  st.session_state.stop_dict, st.session_state.training_status, st.session_state.loss_dict, st.session_state.experiment_tracker),
            daemon=True
        )
        training_thread.start()                
        st.rerun()
    
    # Monitor training or show results
    if st.session_state.training_started:
        monitor_training_with_live_plot(
            st.session_state.stop_dict,
            st.session_state.training_status,
            st.session_state.loss_dict,
            st.session_state.experiment_tracker
        )
        st.session_state.training_started = False
        if st.session_state.loss_dict.get("training_complete", False):
            st.rerun()
    elif st.session_state.loss_dict.get("train_history") or st.session_state.loss_dict.get("eval_history"):
        st.subheader("Previous Training Results")
        fig = plot_losses(st.session_state.loss_dict["train_history"], st.session_state.loss_dict["eval_history"], "LLM Loss")
        if fig:
            st.pyplot(fig, width=700)
            plt.close(fig)
    
    # Push to Hub section
    render_hub_push_section(hf_token, disabled=should_disable)
    if st.session_state.push_hub_msg["status"] is not None:
        if st.session_state.push_hub_msg["status"] == "error":
            st.error(st.session_state.push_hub_msg["message"])
            st.session_state.push_hub_msg = {"status": None, "message": None}
        elif st.session_state.push_hub_msg["status"] == "success":
            st.success(st.session_state.push_hub_msg["message"])
            st.session_state.push_hub_msg = {"status": None, "message": None}

if __name__ == "__main__":
    main()