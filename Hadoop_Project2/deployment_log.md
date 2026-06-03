# 项目部署与调试记录

## 环境信息

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11 Home China (原项目为 Linux) |
| Python | 3.14 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8GB) |
| CUDA | 12.8 |
| 基础模型 | Qwen3.5-0.8B |
| 微调方法 | LoRA (r=16, alpha=32) |
| 训练数据 | DataMind-12K → 筛选后 2000 train + 500 val |

---

## 问题 1：基础模型缺失

**现象**：`qwen_model_local/` 目录不在仓库中（被 .gitignore 排除）

**解决**：从 HuggingFace 下载 Qwen3.5-0.8B 到 `Spark_Project2/qwen_model_local/`

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='E:/hadoop_project2/Spark_Project2/qwen_model_local')"
```

**要点**：模型约 1.7GB，部署时需要确保模型目录与代码中引用的路径一致。

---

## 问题 2：硬编码 Linux 路径

**现象**：所有 Python 脚本使用 `/root/Hadoop_Project2/` 绝对路径，Windows 上无法找到文件

**涉及文件**：
- `data_processing.py` — 6 处路径（数据文件、输出文件、评分缓存）
- `train.py` — 4 处路径（模型路径、训练/验证数据、输出目录）
- `deepseek_scoring.py` — 2 处路径（输入数据、评分输出）
- `ifd_scoring.py` — 3 处路径（模型路径、LoRA checkpoint、数据文件）

**解决**：全部替换为基于 `os.path.dirname(os.path.abspath(__file__))` 的相对路径

```python
# 修改前
INPUT = "/root/Hadoop_Project2/data/datamind_12k.json"

# 修改后
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "data", "datamind_12k.json")
```

**要点**：跨平台项目必须使用动态路径，避免硬编码绝对路径。

---

## 问题 3：torch/torchvision 版本不兼容

**现象**：
```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'PreTrainedModel'
```

**原因**：预装的 `torch 2.12.0.dev` (开发版) 与 `torchvision 0.25.0` 不兼容

**解决**：卸载后重装 CPU 版，然后安装 CUDA 12.8 版以启用 GPU 加速

```bash
pip uninstall torchvision -y
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

最终版本：`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`

**验证**：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

**要点**：部署前需确认 torch 版本与 CUDA 驱动版本匹配。

---

## 问题 4：transformers 版本不支持 qwen3_5

**现象**：
```
The checkpoint you are trying to load has model type `qwen3_5` but Transformers does not recognize this architecture.
```

**原因**：`transformers 4.57.3` 不包含 Qwen3.5 的模型定义

**解决**：升级到最新版

```bash
pip install --upgrade transformers
```

最终版本：`transformers 5.9.0`

**要点**：新模型需要较新版本的 transformers。

---

## 问题 5：代码块格式不匹配（Code Runner 无法提取代码）

**现象**：模型生成了代码，但 Code Runner 面板始终为空

**原因**：模型用 DataMind-12K 微调，训练数据使用 `<code>...</code>` XML 标签包裹代码，但 `app.py` 的 `extract_code_blocks()` 只识别 ` ```python...``` ` 格式

**解决**：扩展正则匹配，支持三种格式

```python
patterns = [
    (r"```python\s*\n(.*?)```", re.DOTALL),   # 标准 Markdown
    (r"```\s*\n(.*?)```", re.DOTALL),         # 无语言标记
    (r"<code>\s*\n?(.*?)</code>", re.DOTALL),  # DataMind XML 格式
]
```

同时将"只取第一个匹配到的模式"改为"合并所有匹配结果"，避免漏掉代码块。

**要点**：模型输出格式取决于训练数据格式，前端需要兼容多种代码块格式。

---

## 问题 6：代码执行中的 pd.read_csv 路径错误

**现象**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'uploaded_dataset.csv'
```

**原因**：模型生成的代码写 `df = pd.read_csv('uploaded_dataset.csv')`，但执行环境中不存在该文件。实际上 `df` 变量已被预先加载。

**解决**：在代码注入前自动替换

```python
code = re.sub(r'^\s*df\s*=\s*pd\.read_csv\([^)]*\).*$',
              '# df is pre-loaded with your data', code, flags=re.MULTILINE)
```

**要点**：模型不知道执行环境变量 `df` 已经存在，需要在执行前清理冗余的数据加载代码。

---

## 问题 7：CSV 解析后日期列仍为 object 类型

**现象**：
```
AttributeError: Can only use .dt accessor with datetimelike values
```

**原因**：`pd.to_csv()` 保存后 `pd.read_csv()` 重新读取，日期列变回了字符串

**解决**：执行脚本中加 `parse_dates=True`

```python
df = pd.read_csv(csv_path, parse_dates=True)
```

**要点**：CSV 不保留数据类型信息，回写再读取需要显式处理日期等特殊类型。

---

## 问题 8：gr.Image 无法显示 base64 图片

**现象**：Output 区域正常显示 `__PLOT_SAVED__`，但 Plot 区域空白

**原因**：`execute_code()` 返回 base64 字符串，但 Gradio 的 `gr.Image` 组件需要文件路径、numpy 数组或 PIL Image

**解决**：将 base64 解码写入临时 PNG 文件后返回路径

```python
if plot_b64:
    plot_file = os.path.join(tempfile.gettempdir(), "gradio_plot.png")
    with open(plot_file, "wb") as f:
        f.write(base64.b64decode(plot_b64))
return output, plot_file
```

**要点**：Gradio 各组件对数据格式有特定要求，需查阅文档确认。

---

## 问题 9：代码块缩进异常

**现象**：
```
IndentationError: unindent does not match any outer indentation level
```

**原因**：从模型响应中提取的代码保留了原始缩进（例如缩进了 4 格或被 `<think>` 标签包裹），直接执行导致缩进错误

**解决**：使用 `textwrap.dedent()` 统一去除公共前导空白

```python
all_matches.extend(textwrap.dedent(m).strip() for m in matches if m.strip())
```

**要点**：LLM 输出的代码缩进不可靠，执行前需标准化。

---

## 问题 10：System Prompt 优化

**现象**：模型输出只有代码没有文字解释，且有时用 `<code>` 标签而非 markdown

**解决**：在 System Prompt 中明确要求：

```
1. ALWAYS explain your analysis approach and findings in natural language BEFORE and AFTER the code.
2. When writing Python code, put it inside ```python code blocks (NOT <code> tags).
3. Use the variable `df` to refer to the uploaded dataset — it is already loaded and ready.
4. After generating a chart, explain what the chart shows and what insights can be drawn.
```

**要点**：微调模型的输出习惯可能与期望不一致，通过 System Prompt 引导。

---

## 最终项目结构

```
Spark_Project2/
├── qwen_model_local/          # Qwen3.5-0.8B 基础模型 (~1.7GB)
└── Hadoop_Project2/
    ├── app.py                 # Gradio Web UI (主入口)
    ├── data_processing.py     # 数据质量评分 + 多样性筛选
    ├── ifd_scoring.py         # 模型级 IFD 评分
    ├── deepseek_scoring.py    # DeepSeek API 评分
    ├── train.py               # Ray Train LoRA 微调
    ├── data/
    │   ├── train_data.jsonl   # 2000 条训练样本
    │   ├── val_data.jsonl     # 500 条验证样本
    │   └── selection_report.txt
    ├── output/
    │   └── qwen-datamind-lora-v2/best_model/
    │       ├── adapter_config.json
    │       └── adapter_model.safetensors  # LoRA 权重 (~25MB)
    └── sample_sales_data.csv  # 测试用 CSV
```

## 快速启动

```bash
cd E:\hadoop_project2\Spark_Project2\Hadoop_Project2
python app.py
# 浏览器打开 http://localhost:7862
```
