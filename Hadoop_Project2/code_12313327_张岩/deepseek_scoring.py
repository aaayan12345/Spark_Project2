#!/usr/bin/env python3
"""
Q1(2): DeepSeek API-Based Trajectory Scoring
==============================================
Uses the DeepSeek API to evaluate DataMind-12K trajectory quality
across multiple dimensions for instruction-tuning data selection.

This implements the InstructMining approach: using an external LLM
to predict data quality without actual fine-tuning.

Dimensions scored (1-10 each):
  1. reasoning_depth     - How deep/rigorous is the analytical reasoning?
  2. code_correctness    - Is the Python/SQL code correct and runnable?
  3. instruction_clarity - How clear and well-defined is the user's task?
  4. answer_completeness - How complete and accurate is the final answer?
  5. educational_value   - How useful is this trajectory for training?

Setup:
  pip install openai
  export DEEPSEEK_API_KEY="your-deepseek-api-key"

Get an API key at: https://platform.deepseek.com/
"""

import json
import os
import time
import sys
import numpy as np

# ============================================================
# Configuration
# ============================================================

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "datamind_12k.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "deepseek_scores.json")

# Scoring control
MAX_SAMPLES = 500        # limit API calls
BATCH_DELAY = 1.0        # seconds between calls
MAX_RETRIES = 3
MAX_CHARS_PER_TRAJ = 8000  # truncate long trajectories

# ============================================================
# Scoring Prompt
# ============================================================

SCORING_SYSTEM_PROMPT = """You are an expert evaluator of AI training data quality for data science and mathematical reasoning. Score data analysis agent trajectories on multiple dimensions.

For each trajectory, you will see a multi-turn conversation between a user (data analysis questions) and an assistant (reasoning, code, answers).

Score each dimension from 1 (worst) to 10 (best):
- reasoning_depth: Does the assistant show step-by-step analytical thinking? Are assumptions stated? Is the logic sound?
- code_correctness: Is the Python/SQL code syntactically correct, runnable, and appropriate? (Score 5 if no code)
- instruction_clarity: Is the user's data analysis question clear, specific, and well-defined?
- answer_completeness: Does the assistant provide a complete, accurate final answer?
- educational_value: How useful would this trajectory be as a training example for data analysis?

Output ONLY a valid JSON object (no other text):
{"reasoning_depth": X, "code_correctness": X, "instruction_clarity": X, "answer_completeness": X, "educational_value": X, "overall_comment": "brief summary"}"""


def build_scoring_text(trajectory):
    """Extract and format trajectory for API evaluation."""
    messages = trajectory.get("messages", [])

    lines = []
    for i, m in enumerate(messages):
        role = m.get("role", "unknown")
        content = m.get("content", "")

        if len(content) > 1200:
            content = content[:600] + "\n... [truncated] ...\n" + content[-600:]

        lines.append(f"[Turn {i}] {role.upper()}:")
        lines.append(content)
        lines.append("")

    text = "\n".join(lines)
    if len(text) > MAX_CHARS_PER_TRAJ:
        text = text[:MAX_CHARS_PER_TRAJ]
    return text


def score_trajectory(client, trajectory, index, total):
    """Score a single trajectory via DeepSeek API. Returns dict or None."""
    scoring_text = build_scoring_text(trajectory)

    user_msg = f"""Score this data analysis agent trajectory (1-10 per dimension):

{scoring_text}

Output a JSON object with scores."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=256,
            )

            raw = response.choices[0].message.content.strip()

            # DeepSeek may wrap JSON in ```json blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)

            dims = ["reasoning_depth", "code_correctness", "instruction_clarity",
                    "answer_completeness", "educational_value"]
            scores = {}
            for d in dims:
                val = result.get(d, 5)
                scores[d] = float(max(1, min(10, val)))

            scores["overall_comment"] = result.get("overall_comment", "")
            scores["composite"] = round(sum(scores[d] for d in dims) / len(dims), 2)

            return scores

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            print(f"  WARNING: Failed to parse JSON after {MAX_RETRIES} attempts. Raw: {raw[:100]}")
            return None

        except Exception as e:
            error_msg = str(e)
            if "rate" in error_msg.lower() or "429" in error_msg:
                wait = (attempt + 1) * 8
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if "insufficient" in error_msg.lower() or "balance" in error_msg.lower():
                print(f"  ERROR: Insufficient balance. Top up your DeepSeek account.")
                return None
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            print(f"  ERROR (idx={index}): {e}")
            return None

    return None


# ============================================================
# Batch Scoring
# ============================================================

def score_batch(trajectories, sample_size=MAX_SAMPLES, resume=True):
    """Score trajectories in batch via DeepSeek API. Supports resuming."""
    import random
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable.")
        print("  Get a key at: https://platform.deepseek.com/")
        print("  Then: export DEEPSEEK_API_KEY='sk-xxx'")
        return []

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_API_BASE)

    # Sample
    n = min(sample_size, len(trajectories)) if sample_size else len(trajectories)
    indices = sorted(random.Random(42).sample(range(len(trajectories)), n))
    sample = [(idx, trajectories[idx]) for idx in indices]

    # Resume from existing
    results = []
    scored_indices = set()
    if resume and os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r") as f:
                existing = json.load(f)
            results = existing
            scored_indices = {r["index"] for r in results}
            print(f"Resuming: {len(results)} scored, {n - len(scored_indices)} remaining")
        except Exception:
            pass

    pending = [(idx, traj) for idx, traj in sample if idx not in scored_indices]

    print(f"\nScoring {len(pending)} trajectories via DeepSeek API")
    print(f"Model: {DEEPSEEK_MODEL}, Delay: {BATCH_DELAY}s between calls")
    print(f"Estimated time: ~{len(pending) * BATCH_DELAY / 60:.0f} min\n")

    t0 = time.time()
    for i, (orig_idx, traj) in enumerate(pending):
        scores = score_trajectory(client, traj, orig_idx, len(pending))

        if scores is not None:
            scores["index"] = orig_idx
            results.append(scores)
            scored_indices.add(orig_idx)

            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1) * 60
            eta = (len(pending) - i - 1) / max(rate, 0.01)
            comp = scores.get("composite", "?")

            if (i + 1) % 10 == 0 or i < 3:
                print(f"  [{i + 1}/{len(pending)}] idx={orig_idx} "
                      f"composite={comp} | ~{rate:.0f}/min | ETA {eta:.0f}s")
        else:
            results.append({"index": orig_idx, "error": True, "composite": 2.5})
            scored_indices.add(orig_idx)

        # Save incrementally every 50
        if (i + 1) % 50 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

        time.sleep(BATCH_DELAY)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    successful = sum(1 for r in results if "error" not in r)
    print(f"\nDone! {successful}/{len(pending)} scored in {elapsed:.0f}s")
    print(f"Scores saved to {OUTPUT_PATH}")
    return results


# ============================================================
# Analysis & Filtering
# ============================================================

def analyze_scores(results):
    """Print score distribution and statistics."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid scores.")
        return

    dims = ["reasoning_depth", "code_correctness", "instruction_clarity",
            "answer_completeness", "educational_value", "composite"]

    print(f"\n{'='*60}")
    print(f"DeepSeek Scoring Analysis ({len(valid)} trajectories)")
    print(f"{'='*60}")

    for dim in dims:
        values = [r[dim] for r in valid]
        print(f"  {dim:25s}: mean={np.mean(values):.2f}  "
              f"median={np.median(values):.2f}  "
              f"std={np.std(values):.2f}  "
              f"[{min(values):.1f}, {max(values):.1f}]")

    composites = sorted([r["composite"] for r in valid])
    for q in range(5):
        start = q * len(composites) // 5
        end = (q + 1) * len(composites) // 5
        seg = composites[start:min(end, len(composites))]
        if seg:
            print(f"  Q{q + 1}: {seg[0]:.2f} - {seg[-1]:.2f}")

    sorted_valid = sorted(valid, key=lambda x: x["composite"])
    print(f"\n  Bottom-3:")
    for r in sorted_valid[:3]:
        print(f"    composite={r['composite']:.2f} | {r.get('overall_comment', '')[:80]}")
    print(f"\n  Top-3:")
    for r in sorted_valid[-3:]:
        print(f"    composite={r['composite']:.2f} | {r.get('overall_comment', '')[:80]}")

    # Dimension correlations
    print(f"\n  Dimension correlations:")
    dim_values = {d: [r[d] for r in valid] for d in dims}
    header = "           " + " ".join(f"{d.replace('_','')[:8]:>8s}" for d in dims)
    print(header)
    for d1 in dims:
        row = f"  {d1[:10]:>10s}"
        for d2 in dims:
            corr = np.corrcoef(dim_values[d1], dim_values[d2])[0, 1]
            row += f"  {corr:5.2f} "
        print(row)

    return valid


def deepseek_filter(scored_results, all_trajectories, keep_ratio=0.75):
    """
    Filter trajectories by DeepSeek composite score.
    Keeps top keep_ratio of scored trajectories.
    Unscored trajectories pass through.
    """
    valid = [r for r in scored_results if "error" not in r]
    valid.sort(key=lambda x: x["composite"], reverse=True)

    keep_n = int(len(valid) * keep_ratio)
    cut_score = valid[keep_n - 1]["composite"] if keep_n > 0 else 0

    kept = [all_trajectories[r["index"]] for r in valid[:keep_n]]
    rejected_scored = [all_trajectories[r["index"]] for r in valid[keep_n:]]

    # Unscored pass through
    scored_set = {r["index"] for r in scored_results}
    unscored = [t for i, t in enumerate(all_trajectories) if i not in scored_set]

    print(f"\nDeepSeek Filter (top {keep_ratio*100:.0f}%):")
    print(f"  Kept:   {len(kept)} scored + {len(unscored)} unscored")
    print(f"  Cutoff: composite >= {cut_score:.2f}")
    print(f"  Range:  [{valid[keep_n-1]['composite']:.2f}, {valid[0]['composite']:.2f}]")

    return kept + unscored


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek API Trajectory Quality Scoring")
    print("=" * 60)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} trajectories")

    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_SAMPLES
    action = sys.argv[2] if len(sys.argv) > 2 else "score"

    if action == "score":
        results = score_batch(all_data, sample_size=sample_size)
        analyze_scores(results)

    elif action == "filter":
        if not os.path.exists(OUTPUT_PATH):
            print(f"No scores at {OUTPUT_PATH}. Run 'score' first.")
            sys.exit(1)
        with open(OUTPUT_PATH, "r") as f:
            results = json.load(f)
        analyze_scores(results)
        pool = deepseek_filter(results, all_data, keep_ratio=0.75)
        print(f"\nFiltered pool: {len(pool)} trajectories")

    elif action == "analyze":
        if not os.path.exists(OUTPUT_PATH):
            print(f"No scores at {OUTPUT_PATH}. Run 'score' first.")
            sys.exit(1)
        with open(OUTPUT_PATH, "r") as f:
            results = json.load(f)
        analyze_scores(results)

    else:
        print(f"Unknown action: {action}")
        print("Usage: python3 deepseek_scoring.py [sample_size] [score|filter|analyze]")
