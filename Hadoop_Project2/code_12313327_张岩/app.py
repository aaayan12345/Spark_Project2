#!/usr/bin/env python3
"""
Q1(4): Data Agent Demo — Gradio Web Interface
==============================================
Fine-tuned Qwen3.5-0.8B with LoRA on DataMind-12K.
Features: file upload (CSV/Excel) + code highlight & execute.
"""

import sys
import os
import gc
import re
import io
import textwrap
import base64
import traceback
import subprocess
import tempfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(BASE_DIR, "..", "qwen_model_local")
LORA_PATH = os.path.join(BASE_DIR, "output", "qwen-datamind-lora-v2", "best_model")
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_CONTEXT_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are a data analysis and data science assistant. "
    "Help users analyze data, write Python code, create visualizations, and interpret results.\n"
    "IMPORTANT rules:\n"
    "1. ALWAYS explain your analysis approach and findings in natural language BEFORE and AFTER the code.\n"
    "2. When writing Python code, put it inside ```python code blocks (NOT <code> tags).\n"
    "3. Use the variable `df` to refer to the uploaded dataset — it is already loaded and ready.\n"
    "4. After generating a chart, explain what the chart shows and what insights can be drawn."
)

EXAMPLE_QUESTIONS = [
    "Show me the basic statistics of the dataset",
    "Write Python code to create a scatter plot of the first two numeric columns",
    "How do I handle missing values in this dataset?",
    "Detect outliers using IQR method",
    "Plot the distribution of each numeric column",
]

CSS = """
#title { text-align: center; font-size: 1.5em; margin-bottom: 0.5em; }
#status-ok { color: #27ae60; font-weight: bold; }
#status-err { color: #e74c3c; font-weight: bold; }
footer { visibility: hidden; }
.code-output { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 8px;
               font-family: 'Courier New', monospace; font-size: 13px;
               max-height: 200px; overflow-y: auto; white-space: pre-wrap; }
"""


# ============================================================
# Model Loading
# ============================================================

def load_model_and_tokenizer():
    """Load base model + LoRA adapter + tokenizer."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, local_files_only=True, dtype=dtype,
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
# File Upload Logic
# ============================================================

def parse_file(file):
    """Parse uploaded CSV/Excel file, return (df, preview_df, data_context)."""
    if file is None:
        return None, gr.update(value=None, visible=False), ""

    filepath = file.name
    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            return None, gr.update(value=None, visible=False), "Unsupported file format"

        preview = df.head(10).copy()
        # Truncate long strings for display
        for col in preview.select_dtypes(include=["object"]).columns:
            preview[col] = preview[col].astype(str).str[:50]

        context = build_data_context(df)
        return df, gr.update(value=preview, visible=True), context

    except Exception as e:
        return None, gr.update(value=None, visible=False), f"Error parsing file: {e}"


def build_data_context(df):
    """Build a text summary of the dataframe for the prompt."""
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()

    lines = [
        f"**Uploaded Dataset Summary:**",
        f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"- Columns: {list(df.columns)}",
        f"- Dtypes:\n{df.dtypes.to_string()}",
        f"- Missing values:\n{df.isnull().sum().to_string()}",
        f"- First 5 rows:\n{df.head().to_string()}",
    ]
    return "\n".join(lines)


def handle_upload(file):
    """Gradio callback for file upload."""
    df, preview_update, context = parse_file(file)
    if df is not None:
        filename = os.path.basename(file.name)
        status = f"Loaded: {filename} ({df.shape[0]} rows × {df.shape[1]} cols)"
        return df, preview_update, status, context
    else:
        return None, gr.update(value=None, visible=False), context, ""


def clear_upload():
    """Clear uploaded data."""
    return None, gr.update(value=None, visible=False), "", ""


# ============================================================
# Code Execution
# ============================================================

def extract_code_blocks(text):
    """Extract code blocks from text. Supports ```python, ```, and <code> tags."""
    all_matches = []
    # Try all patterns and combine results
    patterns = [
        (r"```python\s*\n(.*?)```", re.DOTALL),
        (r"```\s*\n(.*?)```", re.DOTALL),
        (r"<code>\s*\n?(.*?)</code>", re.DOTALL),
    ]
    for pattern, flags in patterns:
        matches = re.findall(pattern, text, flags)
        all_matches.extend(textwrap.dedent(m).strip() for m in matches if m.strip())
    return all_matches


def execute_code(code, df_state):
    """Execute Python code in a subprocess with the dataframe available.
    Returns (stdout_output, plot_base64)."""
    if not code.strip():
        return "", None

    # Write dataframe to temp CSV
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, "data.csv")
    plot_path = os.path.join(tmpdir, "plot.png")

    if df_state is not None:
        df_state.to_csv(csv_path, index=False)

    # Strip `df = pd.read_csv(...)` lines from generated code — df is pre-loaded
    code = re.sub(r'^\s*df\s*=\s*pd\.read_csv\([^)]*\).*$', '# df is pre-loaded with your data', code, flags=re.MULTILINE)

    script = f"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io, os, base64

# Load data (try to parse dates automatically)
csv_path = {csv_path!r}
plot_path = {plot_path!r}
try:
    df = pd.read_csv(csv_path, parse_dates=True)
except Exception:
    df = None

# User code
{code}

# Save plot if there is a figure
fig = plt.gcf()
if fig.get_axes():
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    print("__PLOT_SAVED__")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
            cwd=tmpdir,
        )
        stdout = result.stdout
        stderr = result.stderr
        if stderr:
            stdout += "\n[stderr]\n" + stderr

        # Check for plot
        plot_b64 = None
        if os.path.exists(plot_path) and os.path.getsize(plot_path) > 0:
            with open(plot_path, "rb") as f:
                plot_b64 = base64.b64encode(f.read()).decode()

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

        return stdout.strip() or "(no output)", plot_b64

    except subprocess.TimeoutExpired:
        return "Execution timed out (30s limit)", None
    except Exception as e:
        return f"Execution error: {traceback.format_exc()}", None


def extract_and_format_code(text):
    """Extract last python block from text, return (code, display_text)."""
    blocks = extract_code_blocks(text)
    if blocks:
        return blocks[-1]
    return ""


# ============================================================
# Chat Logic
# ============================================================

def build_messages(query, history, system_prompt, data_context=""):
    """Build ChatML message list. Appends data context to system prompt."""
    full_system = system_prompt
    if data_context:
        full_system += "\n\n" + data_context

    messages = [{"role": "system", "content": full_system}]
    for q, r in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": r})
    messages.append({"role": "user", "content": query})
    return messages


def chat_stream(model, tokenizer, query, history, system_prompt,
                max_tokens, temperature, top_p, data_context=""):
    """Generate streaming response."""
    messages = build_messages(query, history, system_prompt, data_context)

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    token_count = len(tokenizer.encode(full_text))
    while token_count + max_tokens > MAX_CONTEXT_TOKENS and len(messages) > 2:
        messages.pop(1)
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

def predict(query, chatbot, task_history, system_prompt, max_tokens,
            temperature, data_context, df_state):
    """Streaming predict with data context injection."""
    if not query.strip():
        return chatbot, task_history, "", None

    chatbot.append({"role": "user", "content": query})
    chatbot.append({"role": "assistant", "content": "▊ Generating..."})

    print(f"[predict] Starting generation for: {query[:80]}...")
    print(f"[predict] Device: {model.device}, Model device: {next(model.parameters()).device}")

    full_response = ""
    try:
        for new_text in chat_stream(
            model, tokenizer, query, task_history, system_prompt,
            max_tokens, temperature, TOP_P, data_context,
        ):
            full_response += new_text
            chatbot[-1] = {"role": "assistant", "content": full_response}
            yield chatbot, task_history, "", None
    except Exception as e:
        print(f"[predict] ERROR: {e}")
        chatbot[-1] = {"role": "assistant", "content": f"**Error:** {e}"}
        yield chatbot, task_history, "", None
        return

    task_history.append((query, full_response))

    # Extract code for execution panel
    code = extract_and_format_code(full_response)
    yield chatbot, task_history, code, None

    print(f"Q: {query[:80]}...")
    print(f"A: {full_response[:120]}...")


def regenerate(query, chatbot, task_history, system_prompt, max_tokens,
               temperature, data_context, df_state):
    """Remove last exchange and redo prediction."""
    if not task_history:
        return chatbot, task_history, "", None

    last_query, _ = task_history.pop(-1)
    chatbot.pop(-1)
    chatbot.pop(-1)

    chatbot.append({"role": "user", "content": last_query})
    chatbot.append({"role": "assistant", "content": ""})

    full_response = ""
    for new_text in chat_stream(
        model, tokenizer, last_query, task_history, system_prompt,
        max_tokens, temperature, TOP_P, data_context,
    ):
        full_response += new_text
        chatbot[-1] = {"role": "assistant", "content": full_response}
        yield chatbot, task_history, "", None

    task_history.append((last_query, full_response))

    code = extract_and_format_code(full_response)
    yield chatbot, task_history, code, None


def run_code(code, df_state):
    """Execute the code and return output + plot."""
    if not code.strip():
        return "", None
    output, plot_b64 = execute_code(code, df_state)
    # Convert base64 to a temp file for gr.Image
    plot_file = None
    if plot_b64:
        plot_file = os.path.join(tempfile.gettempdir(), "gradio_plot.png")
        with open(plot_file, "wb") as f:
            f.write(base64.b64decode(plot_b64))
    return output, plot_file


def reset_state(chatbot, task_history):
    """Clear conversation history."""
    task_history.clear()
    chatbot.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return chatbot, task_history, "", None


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
            # --- Left: Chat ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Data Agent Chat",
                    elem_classes="control-height",
                    height=500,
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
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    regen_btn = gr.Button("Regenerate", variant="secondary", scale=1)
                    clear_btn = gr.Button("Clear History", variant="stop", scale=1)

                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=query,
                    label="Example questions",
                )

                # --- Code Execution Panel ---
                gr.Markdown("### Code Runner")
                with gr.Row():
                    code_area = gr.Textbox(
                        lines=6,
                        placeholder="Python code extracted from assistant response will appear here...",
                        label="Python Code",
                    )
                with gr.Row():
                    run_btn = gr.Button("Run Code", variant="primary", size="sm")
                with gr.Row():
                    code_output = gr.Textbox(
                        lines=6,
                        label="Output",
                        interactive=False,
                    )
                    plot_output = gr.Image(
                        label="Plot",
                        visible=True,
                    )

            # --- Right: Data & Params ---
            with gr.Column(scale=2):
                gr.Markdown("### Upload Data")
                file_upload = gr.File(
                    label="CSV or Excel file",
                    file_types=[".csv", ".xlsx", ".xls"],
                    file_count="single",
                )
                upload_status = gr.Markdown("")

                data_preview = gr.Dataframe(
                    label="Data Preview (first 10 rows)",
                    visible=False,
                    wrap=True,
                )

                clear_data_btn = gr.Button("Clear Data", variant="secondary", size="sm")

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

        # Hidden states
        task_history = gr.State([])
        df_state = gr.State(None)
        data_context = gr.State("")

        # --- Wire up: File Upload ---
        file_upload.change(
            handle_upload,
            inputs=[file_upload],
            outputs=[df_state, data_preview, upload_status, data_context],
        )
        clear_data_btn.click(
            clear_upload,
            outputs=[df_state, data_preview, upload_status, data_context],
        )

        # --- Wire up: Chat ---
        predict_inputs = [
            query, chatbot, task_history, system_prompt,
            max_tokens, temperature, data_context, df_state,
        ]
        predict_outputs = [chatbot, task_history, code_area, plot_output]

        submit_event = submit_btn.click(
            predict,
            inputs=predict_inputs,
            outputs=predict_outputs,
        )
        submit_event.then(reset_user_input, outputs=[query])

        query.submit(
            predict,
            inputs=predict_inputs,
            outputs=predict_outputs,
        ).then(reset_user_input, outputs=[query])

        clear_btn.click(
            reset_state,
            inputs=[chatbot, task_history],
            outputs=predict_outputs,
            show_progress="hidden",
        )

        regen_btn.click(
            regenerate,
            inputs=predict_inputs,
            outputs=predict_outputs,
            show_progress="hidden",
        )

        # --- Wire up: Code Execution ---
        run_btn.click(
            run_code,
            inputs=[code_area, df_state],
            outputs=[code_output, plot_output],
        )

    return demo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    global model, tokenizer, device, model_ok, load_error

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
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

    # Warmup: first inference on MPS is slow, do it ahead of time
    if model_ok:
        print("Warming up model (first inference on MPS is slow)...")
        warmup_msg = [{"role": "user", "content": "Hello"}]
        warmup_text = tokenizer.apply_chat_template(warmup_msg, tokenize=False, add_generation_prompt=True)
        warmup_inputs = tokenizer([warmup_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**warmup_inputs, max_new_tokens=8, do_sample=False)
        print("Warmup complete!")

    demo = build_ui()
    print("\nStarting Gradio server...")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="localhost",
        server_port=7862,
        share=False,
        css=CSS,
    )
