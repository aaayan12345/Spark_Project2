# Data Agent Demo — Qwen3.5-0.8B + LoRA on DataMind-12K

Fine-tuned Qwen3.5-0.8B data analysis assistant with Gradio web interface.
Supports file upload (CSV/Excel), streaming chat, and one-click code execution.

## Project Structure

```
Hadoop_Project2/
├── app.py                  # Gradio web UI (main entrypoint)
├── data_processing.py      # Data quality scoring + selection pipeline
├── ifd_scoring.py          # Model-based IFD (Instruction-Following Difficulty) scoring
├── deepseek_scoring.py     # DeepSeek API-based trajectory quality scoring
├── train.py                # Ray Train LoRA fine-tuning script
├── Q1_1_Data_Selection_Methods.md  # Methodology documentation
├── data/
│   ├── train_data.jsonl    # 2000 training samples (ChatML format)
│   ├── val_data.jsonl      # 500 validation samples (ChatML format)
│   └── selection_report.txt
└── output/
    └── qwen-datamind-lora-v2/
        └── best_model/     # Trained LoRA adapter weights
```

## Quick Start

### 1. Environment

```bash
pip install torch transformers peft gradio pandas matplotlib seaborn openpyxl
```

### 2. Model Files

Ensure the base model `qwen_model_local` is in the parent directory:
```
Spark_Project2/
├── qwen_model_local/       # Qwen3.5-0.8B base model
└── Hadoop_Project2/        # This project
```

### 3. Run the Web App

```bash
cd Hadoop_Project2
python app.py
```

Open **http://localhost:7862** in your browser.

### Features

- **File Upload** — Upload CSV/Excel files, auto-preview first 10 rows, inject dataset info into the prompt
- **Streaming Chat** — Ask data analysis questions, get real-time streaming responses
- **Code Runner** — Python code blocks from the assistant can be executed with one click (`df` variable available), shows output + matplotlib plots

## Full Pipeline (Data → Model → App)

### Step 1: Data Selection (Q1.2)

```bash
# Heuristic quality scoring + K-Center diversity selection
python data_processing.py

# Optional: DeepSeek API scoring (needs API key)
export DEEPSEEK_API_KEY="your-key"
python deepseek_scoring.py

# Optional: IFD model-based scoring
python ifd_scoring.py
```

Output: `data/train_data.jsonl` (2000 samples) + `data/val_data.jsonl` (500 samples)

### Step 2: LoRA Fine-Tuning (Q1.3)

```bash
python train.py
```

Uses Ray Train for distributed training with:
- LoRA rank=16, alpha=32
- Mixed precision (bfloat16/fp32)
- Gradient checkpointing
- Single GPU / MPS / CPU compatible

Output: `output/qwen-datamind-lora-ray/best_model/`

### Step 3: Launch Web Demo (Q1.4)

```bash
python app.py
```

## Configuration

Key parameters in `app.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_NEW_TOKENS` | 512 | Max tokens per generation |
| `TEMPERATURE` | 0.7 | Sampling temperature (0 = greedy) |
| `TOP_P` | 0.9 | Nucleus sampling threshold |
| `MAX_CONTEXT_TOKENS` | 2048 | Max context window |
| `server_port` | 7862 | Gradio server port |

## Notes

- On Apple Silicon (MPS), the first inference includes a warmup phase (~10-30s). The app runs this at startup.
- `Trainable: 0K` displayed at startup is a display artifact — LoRA weights are correctly loaded, this is a PEFT eval() mode quirk.
- The `fast path` warning from transformers is harmless — torch fallback handles it.
