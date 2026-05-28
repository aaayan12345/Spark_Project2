#!/usr/bin/env python3
"""
Q1(4): Data Agent Demo — Gradio Web Interface
==============================================
Fine-tuned Qwen3.5-0.8B with LoRA on DataMind-12K.
Pattern follows Qwen3 official web_demo.py with gr.State history management.
"""

import sys
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio is not installed. Run: pip install gradio")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "/root/Hadoop_Project2/qwen_model_local"
LORA_PATH = "/root/Hadoop_Project2/output/qwen-datamind-lora-v2/best_model"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_CONTEXT_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are a data analysis and data science assistant. "
    "Help users analyze data, write Python code, create visualizations, "
    "and interpret results."
)

EXAMPLE_QUESTIONS = [
    "How do I handle missing values in a pandas DataFrame?",
    "Write Python code to create a scatter plot with seaborn.",
    "Explain the difference between L1 and L2 regularization.",
    "How do I perform A/B testing analysis?",
    "Write a function to calculate moving average of a time series.",
]

CSS = """
#title { text-align: center; font-size: 1.5em; margin-bottom: 0.5em; }
#status-ok { color: #27ae60; font-weight: bold; }
#status-err { color: #e74c3c; font-weight: bold; }
footer { visibility: hidden; }
"""


# ============================================================
# Model Loading
# ============================================================

def load_model_and_tokenizer():
    """Load base model + LoRA adapter + tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, local_files_only=True, torch_dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    print(f"Loading LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded. Params: {n_params/1e6:.0f}M, "
          f"Trainable: {n_trainable/1e3:.0f}K ({n_trainable/n_params*100:.2f}%)")
    print(f"Device: {device.upper()}")

    return model, tokenizer


# ============================================================
# Chat Logic
# ============================================================

def build_messages(query, history, system_prompt):
    """Build ChatML message list from query and history tuples."""
    messages = [{"role": "system", "content": system_prompt}]
    for q, r in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": r})
    messages.append({"role": "user", "content": query})
    return messages


def chat_stream(model, tokenizer, query, history, system_prompt,
                max_tokens, temperature, top_p):
    """Generate streaming response. Yields new_text chunks."""
    messages = build_messages(query, history, system_prompt)

    # Context window check: trim old history
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    token_count = len(tokenizer.encode(full_text))
    while token_count + max_tokens > MAX_CONTEXT_TOKENS and len(messages) > 2:
        messages.pop(1)  # Remove oldest exchange
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        token_count = len(tokenizer.encode(full_text))

    inputs = tokenizer([full_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else 1.0,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


# ============================================================
# UI Callbacks
# ============================================================

def predict(query, chatbot, task_history, system_prompt, max_tokens, temperature):
    """Streaming predict: appends user message, streams assistant response."""
    if not query.strip():
        return chatbot, task_history

    # Append user message and empty assistant placeholder
    chatbot.append({"role": "user", "content": query})
    chatbot.append({"role": "assistant", "content": ""})

    full_response = ""
    for new_text in chat_stream(
        model, tokenizer, query, task_history, system_prompt,
        max_tokens, temperature, TOP_P,
    ):
        full_response += new_text
        chatbot[-1] = {"role": "assistant", "content": full_response}
        yield chatbot, task_history

    task_history.append((query, full_response))
    print(f"Q: {query[:80]}...")
    print(f"A: {full_response[:120]}...")


def regenerate(chatbot, task_history, system_prompt, max_tokens, temperature):
    """Remove last exchange and redo prediction."""
    if not task_history:
        yield chatbot, task_history
        return

    last_query, _ = task_history.pop(-1)
    # Remove last user+assistant pair from chatbot
    chatbot.pop(-1)  # assistant
    chatbot.pop(-1)  # user

    chatbot.append({"role": "user", "content": last_query})
    chatbot.append({"role": "assistant", "content": ""})

    full_response = ""
    for new_text in chat_stream(
        model, tokenizer, last_query, task_history, system_prompt,
        max_tokens, temperature, TOP_P,
    ):
        full_response += new_text
        chatbot[-1] = {"role": "assistant", "content": full_response}
        yield chatbot, task_history

    task_history.append((last_query, full_response))


def reset_state(chatbot, task_history):
    """Clear conversation history."""
    task_history.clear()
    chatbot.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return chatbot, task_history


def reset_user_input():
    """Clear text input after submit."""
    return ""


# ============================================================
# Build UI
# ============================================================

def build_ui():
    """Build and return the Gradio Blocks interface."""

    model_status = (
        '<span id="status-ok">&#9679; Model loaded (LoRA)</span>'
        if model_ok else
        f'<span id="status-err">&#9679; Model error: {load_error}</span>'
    )

    with gr.Blocks(title="Data Agent Demo") as demo:
        # Header
        gr.HTML(f"""\
<div style="text-align:center; margin-bottom:1em;">
  <h1>&#128202; Data Agent Demo</h1>
  <p style="color:#666;">Fine-tuned Qwen3.5-0.8B + LoRA on DataMind-12K &nbsp;|&nbsp;
  Val Loss: 0.1177 &nbsp;|&nbsp; Device: {device.upper()}</p>
  {model_status}
</div>""")

        with gr.Row():
            with gr.Column(scale=3):
                # --- Chat area ---
                chatbot = gr.Chatbot(
                    label="Data Agent Chat",
                    elem_classes="control-height",
                    height=550,
                    show_label=False,
                )

                with gr.Row():
                    query = gr.Textbox(
                        lines=2,
                        placeholder="Ask a data analysis question...",
                        show_label=False,
                        scale=5,
                    )

                with gr.Row():
                    submit_btn = gr.Button(
                        "Submit", variant="primary", scale=1,
                    )
                    regen_btn = gr.Button(
                        "Regenerate", variant="secondary", scale=1,
                    )
                    clear_btn = gr.Button(
                        "Clear History", variant="stop", scale=1,
                    )

                # Examples
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=query,
                    label="Example questions",
                )

            with gr.Column(scale=1):
                # --- Settings panel ---
                gr.Markdown("### Generation Parameters")

                max_tokens = gr.Slider(
                    64, 1024, value=MAX_NEW_TOKENS, step=64,
                    label="Max New Tokens",
                )
                temperature = gr.Slider(
                    0.0, 2.0, value=TEMPERATURE, step=0.1,
                    label="Temperature (0 = greedy)",
                )
                system_prompt = gr.Textbox(
                    value=SYSTEM_PROMPT,
                    label="System Prompt",
                    lines=4,
                    interactive=True,
                )

                gr.Markdown("### Model Info")
                gr.Markdown(f"""\
| Property | Value |
|----------|-------|
| Base Model | qwen_model_local |
| LoRA | Loaded (r=16, alpha=32) |
| Parameters | 759M |
| Trainable | 6.4M (0.84%) |
| Val Loss | **0.1177** |
| Device | {device.upper()} |
""")

        # Hidden state for conversation tracking (query, response) pairs
        task_history = gr.State([])

        # --- Wire up events ---
        submit_event = submit_btn.click(
            predict,
            inputs=[query, chatbot, task_history, system_prompt, max_tokens, temperature],
            outputs=[chatbot, task_history],
            show_progress="hidden",
        )
        submit_event.then(reset_user_input, outputs=[query])

        query.submit(
            predict,
            inputs=[query, chatbot, task_history, system_prompt, max_tokens, temperature],
            outputs=[chatbot, task_history],
            show_progress="hidden",
        ).then(reset_user_input, outputs=[query])

        clear_btn.click(
            reset_state,
            inputs=[chatbot, task_history],
            outputs=[chatbot, task_history],
            show_progress="hidden",
        )

        regen_btn.click(
            regenerate,
            inputs=[chatbot, task_history, system_prompt, max_tokens, temperature],
            outputs=[chatbot, task_history],
            show_progress="hidden",
        )

    return demo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    global model, tokenizer, device, model_ok, load_error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ok = True
    load_error = None

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None
        tokenizer = None
        model_ok = False
        load_error = str(e)

    demo = build_ui()
    print("\nStarting Gradio server...")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS,
    )
