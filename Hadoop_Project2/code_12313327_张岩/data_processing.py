#!/usr/bin/env python3
"""
Q1(2): DataMind-12K Data Processing and Selection
===================================================
Methods (based on Survey Paper analysis in Q1(1)):
  1. Heuristic quality scoring (9 features: reasoning depth, code, tools, etc.)
  2. DeepSeek API scoring (InstructMining — LLM evaluates each trajectory)
  3. Model-based IFD scoring (conditional vs unconditional log-probability)
  4. K-Center Greedy: Diversity-driven coreset selection for domain coverage

Pipeline: Quality Filter → [optional API/IFD re-rank] → Diversity Selection → Qwen format

Output:
  - train_data.jsonl  (2,000 samples, ChatML format)
  - val_data.jsonl    (500 samples, ChatML format)
  - selection_report.txt
"""

import json
import random
import os
import numpy as np
from collections import Counter

# ============================================================
# 1. Load Data
# ============================================================

def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "datamind_12k.json")
    """Load the DataMind-12K JSON file."""
    if not os.path.exists(path):
        # Try alternative locations
        alt_paths = [
            "datamind_12k.json",
            "data/datamind_12k.json",
        ]
        for ap in alt_paths:
            if os.path.exists(ap):
                path = ap
                break
        else:
            raise FileNotFoundError(
                f"Cannot find datamind_12k.json at {path}"
            )

    print(f"Loading data from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} trajectories.")
    return data


# ============================================================
# 2. IFD-Inspired Quality Scoring
# ============================================================
# IFD measures how well an instruction guides the model response:
#   IFD(Q, A) = s_theta(A|Q) / s_theta(A)
# High IFD = instruction is crucial for producing the answer (good training signal).
#
# We approximate this using features that correlate with instruction-following
# difficulty and trajectory quality for data science tasks.

def compute_quality_score(traj):
    """
    Compute IFD-inspired quality score for a data science trajectory.
    Features capture: reasoning depth, code complexity, analytical richness.
    """
    messages = traj.get("messages", [])
    if not messages:
        return 0.0

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    asst_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    total_content = " ".join(m["content"] for m in messages)

    # Feature 1: Multi-step reasoning depth (more turns = deeper reasoning)
    num_turns = len(messages)
    reasoning_score = min(num_turns / 12, 1.0) * 0.15

    # Feature 2: Code complexity (code = non-trivial data analysis)
    has_code = "```" in total_content
    code_blocks = total_content.count("```") // 2
    code_score = (0.15 if has_code else 0) + min(code_blocks / 4, 1.0) * 0.10

    # Feature 3: Thinking/reasoning structure
    has_thinking = "## Thought" in total_content or "## thought" in total_content.lower()
    has_observation = "## Observation" in total_content or "## observation" in total_content.lower()
    reasoning_structure = (0.10 if has_thinking else 0) + (0.05 if has_observation else 0)

    # Feature 4: Instruction detail (detailed questions → harder tasks)
    avg_user_len = sum(len(u.split()) for u in user_msgs) / max(len(user_msgs), 1)
    instr_complexity = min(avg_user_len / 150, 1.0) * 0.10

    # Feature 5: Response richness (longer responses = deeper analysis)
    avg_asst_len = sum(len(a.split()) for a in asst_msgs) / max(len(asst_msgs), 1)
    response_depth = min(avg_asst_len / 300, 1.0) * 0.15

    # Feature 6: Data science tool use
    tools = ["import ", "pd.", "np.", "plt.", "sns.", "sklearn", "pandas",
             "read_csv", "groupby", "merge", "dropna", "value_counts",
             "matplotlib", "seaborn", "scipy", "numpy", "plot"]
    tool_count = sum(1 for t in tools if t in total_content)
    tool_score = min(tool_count / 8, 1.0) * 0.10

    # Feature 7: Analytical/mathematical vocabulary
    analytical_terms = [
        "correlation", "regression", "distribution", "statistical",
        "hypothesis", "variance", "mean", "median", "p-value",
        "classification", "cluster", "normalize", "standardize",
        "accuracy", "precision", "recall", "feature", "training",
        "test", "validation", "predict", "forecast", "trend",
    ]
    analytical_count = sum(1 for t in analytical_terms if t.lower() in total_content.lower())
    analytical_score = min(analytical_count / 6, 1.0) * 0.10

    # Feature 8: Error handling (shows robustness)
    has_error = any(kw in total_content.lower() for kw in
                    ["error", "exception", "traceback", "fix", "issue", "problem"])
    error_score = 0.05 if has_error else 0

    # Feature 9: Level difficulty metadata
    level = traj.get("level", "N/A")
    if level == "Highly Complex":
        level_score = 0.20
    elif level == "Complex":
        level_score = 0.15
    elif level == "Moderate":
        level_score = 0.10
    elif level in ("Simple", "easy"):
        level_score = 0.05
    else:
        level_score = 0.10  # N/A → default moderate

    # Composite score (0.0 - 1.0+)
    score = (
        reasoning_score + code_score + reasoning_structure +
        instr_complexity + response_depth + tool_score +
        analytical_score + error_score + level_score
    )

    return round(score, 4)


def quality_filter(data, top_ratio=0.75):
    """
    IFD-inspired quality filtering. Keeps top top_ratio samples.
    """
    print(f"\n{'='*60}")
    print(f"Step 1: IFD-Inspired Quality Filtering (keep top {top_ratio*100:.0f}%)")
    print(f"{'='*60}")

    scored = [(compute_quality_score(t), i, t) for i, t in enumerate(data)]
    scored.sort(key=lambda x: x[0], reverse=True)

    scores = [s[0] for s in scored]
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Mean: {sum(scores)/len(scores):.4f}, Median: {sorted(scores)[len(scores)//2]:.4f}")

    # Quintile breakdown
    n = len(scores)
    for q in range(5):
        start = q * n // 5
        end = min((q + 1) * n // 5, n)
        print(f"  Q{q+1}: {scores[start]:.4f} - {scores[end-1]:.4f}")

    keep_count = int(len(scored) * top_ratio)
    kept = scored[:keep_count]
    rejected = scored[keep_count:]

    print(f"Passed: {len(kept)}, Rejected: {len(rejected)}")

    # Show top/bottom examples
    for label, items in [("Top-3", kept[:3]), ("Bottom-3 (of kept)", kept[-3:])]:
        print(f"\n{label}:")
        for score, idx, traj in items:
            msgs = traj.get("messages", [])
            first = msgs[1]["content"][:80] if len(msgs) > 1 else "N/A"
            level = traj.get("level", "N/A")
            print(f"  score={score:.4f} | level={level} | id={traj.get('id','?')} | "
                  f"'{first}...'")

    return [t for _, _, t in kept], [t for _, _, t in rejected]


# ============================================================
# 3. K-Center Greedy Diversity Selection
# ============================================================

def extract_instruction(traj):
    """Get the first meaningful user instruction."""
    for m in traj.get("messages", []):
        if m["role"] == "user":
            return m["content"]
    return ""


def build_tf_embeddings(texts, vocab_size=2000):
    """
    Build TF (term frequency) embeddings for diversity measurement.
    No external model downloads needed.
    """
    # Build frequency-based vocabulary
    word_doc_count = Counter()
    doc_words = []
    for text in texts:
        tokens = text.lower().split()
        words = set()
        for t in tokens:
            w = "".join(c for c in t if c.isalnum())
            if len(w) > 2:
                words.add(w)
        doc_words.append(words)
        for w in words:
            word_doc_count[w] += 1

    # Keep most frequent words (skip too-rare or too-common)
    top_words = [w for w, _ in word_doc_count.most_common(vocab_size)
                 if 5 < word_doc_count[w] < len(texts) * 0.8]
    vocab = {w: i for i, w in enumerate(top_words[:vocab_size])}
    print(f"Vocabulary: {len(vocab)} words from {len(word_doc_count)} unique")

    # Build TF vectors
    vecs = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, words in enumerate(doc_words):
        wc = Counter()
        for w in words:
            if w in vocab:
                wc[w] = wc.get(w, 0) + 1
        max_c = max(wc.values()) if wc else 1
        for w, c in wc.items():
            vecs[i, vocab[w]] = c / max_c

    return vecs, vocab


def k_center_greedy(embeddings, k, start_idx=None, verbose=True):
    """
    K-Center Greedy: pick k points that maximize coverage.
    Each iteration selects the point farthest from all previously selected.
    """
    n = embeddings.shape[0]
    if k >= n:
        return list(range(n))

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = embeddings / norms

    # Initialize
    rng = random.Random(42)
    selected = [start_idx] if start_idx is not None else [rng.randint(0, n - 1)]
    min_dist = 1.0 - (emb @ emb[selected[0]])

    for iteration in range(1, k):
        farthest = int(np.argmax(min_dist))
        selected.append(farthest)
        new_dist = 1.0 - (emb @ emb[farthest])
        min_dist = np.minimum(min_dist, new_dist)

        if verbose and (iteration + 1) % 500 == 0:
            print(f"  Progress: {iteration + 1}/{k} selected")

    return selected


def diversity_select(pool, n_train, n_val):
    """
    Select diverse train/val sets using K-Center Greedy.
    Validation set selected first (independently), then training from remainder.
    """
    print(f"\n{'='*60}")
    print(f"Step 2: K-Center Greedy Diversity Selection")
    print(f"  Selecting {n_train} train + {n_val} val from {len(pool)} candidates")
    print(f"{'='*60}")

    texts = [extract_instruction(t) for t in pool]
    print("Building TF embeddings...")
    embeddings, vocab = build_tf_embeddings(texts)

    # Select validation set (from full pool)
    print(f"\nSelecting {n_val} validation samples...")
    val_idx = k_center_greedy(embeddings, n_val, verbose=True)
    val_data = [pool[i] for i in val_idx]

    # Remove val samples from pool
    mask = np.ones(len(pool), dtype=bool)
    mask[val_idx] = False
    train_pool_emb = embeddings[mask]
    train_pool_data = [pool[i] for i in range(len(pool)) if mask[i]]

    # Select training set (from remaining)
    print(f"\nSelecting {n_train} training samples from {len(train_pool_data)} remaining...")
    train_idx = k_center_greedy(train_pool_emb, n_train, verbose=True)
    train_data = [train_pool_data[i] for i in train_idx]

    print(f"\nTraining: {len(train_data)}, Validation: {len(val_data)}")
    return train_data, val_data


# ============================================================
# 4. Format for Qwen3.5 Training (ChatML)
# ============================================================

def convert_to_qwen_format(data, output_path):
    """
    Convert to Qwen ChatML JSONL format.
    DataMind already uses the right message structure with:
      messages: [{role, content}, ...]
    """
    print(f"\n{'='*60}")
    print(f"Step 3: Convert to Qwen ChatML Format → {output_path}")
    print(f"{'='*60}")

    converted = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for traj in data:
            messages = traj.get("messages", [])

            # Validate: must have user and assistant turns
            if not any(m["role"] == "user" for m in messages) or \
               not any(m["role"] == "assistant" for m in messages):
                skipped += 1
                continue

            # Validate all messages have required fields
            valid = [{"role": m["role"], "content": m["content"]}
                     for m in messages
                     if m.get("role") in ("user", "assistant", "system") and m.get("content")]

            if len(valid) < 2:
                skipped += 1
                continue

            f.write(json.dumps({"messages": valid}, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted: {converted}, Skipped: {skipped}")
    return converted


# ============================================================
# 5. Selection Analysis
# ============================================================

def generate_report(train_data, val_data, output_path):
    """Generate analysis report of the selection process."""
    print(f"\n{'='*60}")
    print(f"Generating Report → {output_path}")
    print(f"{'='*60}")

    def stats(data, tag):
        msgs_list = [t.get("messages", []) for t in data]
        contents = [" ".join(m["content"] for m in msgs) for msgs in msgs_list]
        word_counts = [len(c.split()) for c in contents]
        turn_counts = [len(msgs) for msgs in msgs_list]
        levels = Counter(t.get("level", "N/A") for t in data)

        return {
            "tag": tag,
            "count": len(data),
            "avg_words": np.mean(word_counts),
            "min_words": int(np.min(word_counts)),
            "max_words": int(np.max(word_counts)),
            "avg_turns": np.mean(turn_counts),
            "min_turns": int(np.min(turn_counts)),
            "max_turns": int(np.max(turn_counts)),
            "levels": dict(levels.most_common()),
        }

    train_s = stats(train_data, "Training Set")
    val_s = stats(val_data, "Validation Set")

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DataMind-12K Selection Report\n")
        f.write("Methods: IFD-inspired Quality Filter + K-Center Greedy\n")
        f.write("=" * 60 + "\n\n")

        for s in [train_s, val_s]:
            f.write(f"{s['tag']}:\n")
            f.write(f"  Samples: {s['count']}\n")
            f.write(f"  Words: avg={s['avg_words']:.0f}, "
                    f"range=[{s['min_words']}, {s['max_words']}]\n")
            f.write(f"  Turns: avg={s['avg_turns']:.1f}, "
                    f"range=[{s['min_turns']}, {s['max_turns']}]\n")
            f.write(f"  Levels: {s['levels']}\n\n")

    for s in [train_s, val_s]:
        print(f"\n{s['tag']}:")
        print(f"  Count: {s['count']}")
        print(f"  Avg words: {s['avg_words']:.0f} [{s['min_words']}, {s['max_words']}]")
        print(f"  Avg turns: {s['avg_turns']:.1f} [{s['min_turns']}, {s['max_turns']}]")
        print(f"  Levels: {s['levels']}")


# ============================================================
# 6. Optional: IFD Model-Based Re-Ranking
# ============================================================

def re_rank_by_ifd(quality_pool, ifd_score_path, ifd_weight=0.5):
    """
    Combine heuristic quality scores with model-based IFD scores.
    ifd_weight: 0.0 = pure heuristic, 1.0 = pure IFD.
    """
    with open(ifd_score_path, "r") as f:
        ifd_scores = json.load(f)
    ifd_map = {s["index"]: s["ifd_score"] for s in ifd_scores}

    print(f"\n{'='*60}")
    print(f"Step 1b: IFD Re-Ranking (weight={ifd_weight})")
    print(f"{'='*60}")
    print(f"Loaded {len(ifd_map)} IFD scores, covering "
          f"{len(set(ifd_map.keys()) & set(range(len(quality_pool))))} "
          f"of {len(quality_pool)} quality pool items")

    # Re-score: combine heuristic + IFD
    scored = []
    for i, traj in enumerate(quality_pool):
        h_score = compute_quality_score(traj)
        ifd = ifd_map.get(i, None)
        if ifd is not None:
            # Normalize IFD: cap extreme values
            ifd_clipped = max(0.5, min(2.0, ifd))
            combined = (1 - ifd_weight) * h_score + ifd_weight * ifd_clipped
        else:
            combined = h_score
        scored.append((combined, i, traj))

    scored.sort(key=lambda x: x[0], reverse=True)

    scores = [s[0] for s in scored]
    print(f"Combined score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Mean: {sum(scores)/len(scores):.4f}")

    return [t for _, _, t in scored]


# ============================================================
# 7. Optional: DeepSeek API-Based Scoring
# ============================================================

def re_rank_by_deepseek(quality_pool, score_path):
    """
    Use DeepSeek API composite scores to rank trajectories.
    Higher composite score = better training data quality.
    Unscored trajectories are placed at median rank.
    """
    with open(score_path, "r") as f:
        api_scores = json.load(f)

    valid = [s for s in api_scores if "error" not in s]
    score_map = {s["index"]: s["composite"] for s in valid}

    print(f"\n{'='*60}")
    print(f"Step 1b: DeepSeek API Re-Ranking")
    print(f"{'='*60}")
    print(f"Loaded {len(score_map)} API scores, "
          f"covering {len(set(score_map.keys()) & set(range(len(quality_pool))))} "
          f"of {len(quality_pool)} quality pool items")

    # Median heuristic for unscored
    h_scores = [compute_quality_score(t) for t in quality_pool]
    median_h = sorted(h_scores)[len(h_scores) // 2]

    scored = []
    unscored_count = 0
    for i, traj in enumerate(quality_pool):
        if i in score_map:
            api_norm = score_map[i] / 10.0  # normalize 1-10 → 0.0-1.0
            scored.append((api_norm, i, traj))
        else:
            scored.append((median_h, i, traj))
            unscored_count += 1

    scored.sort(key=lambda x: x[0], reverse=True)

    scores = [s[0] for s in scored]
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"  {unscored_count} unscored placed at median position")

    return [t for _, _, t in scored]


# ============================================================
# Main
# ============================================================

def main():
    import time
    t0 = time.time()

    print("=" * 60)
    print("DataMind-12K → Qwen3.5 Training Data Pipeline")
    print("=" * 60)

    # Config
    N_TRAIN = 2000
    N_VAL = 500
    QUALITY_RATIO = 0.75

    # Scoring method: "heuristic" | "ifd" | "deepseek"
    SCORING_METHOD = "ifd"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # IFD scoring settings
    IFD_SCORE_PATH = os.path.join(DATA_DIR, "ifd_scores.json")
    IFD_WEIGHT = 0.5  # blend heuristic (0.5) + IFD (0.5)

    # DeepSeek API settings
    DEEPSEEK_SCORE_PATH = os.path.join(DATA_DIR, "deepseek_scores.json")

    # Paths
    INPUT = os.path.join(DATA_DIR, "datamind_12k.json")
    TRAIN_OUT = os.path.join(DATA_DIR, "train_data.jsonl")
    VAL_OUT = os.path.join(DATA_DIR, "val_data.jsonl")
    REPORT_OUT = os.path.join(DATA_DIR, "selection_report.txt")
    REJECTED_OUT = os.path.join(DATA_DIR, "rejected.json")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 0: Load
    all_data = load_data(INPUT)
    print(f"Total: {len(all_data)}")

    # Step 1: Quality filter
    quality_pool, rejected = quality_filter(all_data, QUALITY_RATIO)

    with open(REJECTED_OUT, "w") as f:
        json.dump(rejected, f, indent=2)
        print(f"Rejected saved → {REJECTED_OUT}")

    # Step 1b: Optional model/API re-ranking
    if SCORING_METHOD == "ifd":
        if os.path.exists(IFD_SCORE_PATH):
            quality_pool = re_rank_by_ifd(quality_pool, IFD_SCORE_PATH, IFD_WEIGHT)
        else:
            print(f"\nWARNING: IFD scores not found ({IFD_SCORE_PATH}). "
                  f"Run ifd_scoring.py first. Using heuristic-only.")
    elif SCORING_METHOD == "deepseek":
        if os.path.exists(DEEPSEEK_SCORE_PATH):
            quality_pool = re_rank_by_deepseek(quality_pool, DEEPSEEK_SCORE_PATH)
        else:
            print(f"\nWARNING: DeepSeek scores not found ({DEEPSEEK_SCORE_PATH}). "
                  f"Run deepseek_scoring.py first. Using heuristic-only.")

    # Step 2: Diversity selection
    train_data, val_data = diversity_select(quality_pool, N_TRAIN, N_VAL)

    # Step 3: Convert format
    convert_to_qwen_format(train_data, TRAIN_OUT)
    convert_to_qwen_format(val_data, VAL_OUT)

    # Step 4: Report
    generate_report(train_data, val_data, REPORT_OUT)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done! Time: {elapsed:.1f}s")
    print(f"  Train: {TRAIN_OUT}")
    print(f"  Val:   {VAL_OUT}")
    print(f"  Report: {REPORT_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
