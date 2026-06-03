#!/usr/bin/env python3
"""
Q1(2): Model-Based IFD Scoring (Fixed)
=========================================
IFD(Q, A) = s_θ(A|Q) / s_θ(A)

Uses raw text concatenation (not chat_template) for reliable
assistant-message boundary detection.
"""

import json
import os
import math
import time
import numpy as np
import torch

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen_model_local")
LORA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "qwen-datamind-lora-v2", "best_model")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "datamind_12k.json")
HF_ENDPOINT = "https://hf-mirror.com"

MAX_SAMPLE_SIZE = 500
MAX_MODEL_LENGTH = 2048


def load_scoring_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, trust_remote_code=True,
    )
    model.to(device)

    print(f"Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    print(f"Model ready. Device: {device}")
    return model, tokenizer, device


def build_raw_text(messages, up_to_assistant_idx=None):
    """
    Build raw conversation text with clear role markers.
    Avoids chat_template to keep character positions predictable.
    """
    lines = []
    for i, m in enumerate(messages):
        if up_to_assistant_idx is not None and i > up_to_assistant_idx:
            break
        role = m.get("role", "unknown")
        content = m.get("content", "")
        lines.append(f"### {role}:\n{content}\n")
    return "\n".join(lines)


def compute_token_logprob(model, input_ids, assistant_start_token, assistant_end_token, device):
    """
    Compute sum of log-probabilities for assistant tokens.
    For position i, logits[i] predicts token at i+1.
    """
    with torch.no_grad():
        outputs = model(input_ids.to(device))
        logits = outputs.logits  # [1, seq_len, vocab]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    total = 0.0
    count = 0
    seq_len = input_ids.shape[1]
    for i in range(assistant_start_token, min(assistant_end_token, log_probs.shape[1])):
        if i + 1 < seq_len:
            target_id = input_ids[0, i + 1].item()
            total += log_probs[0, i, target_id].item()
            count += 1
    return total, count


def compute_ifd_single(model, tokenizer, trajectory, device):
    """
    Compute IFD for one trajectory.

    For each assistant turn:
      conditional = P(answer | full context up to assistant)
      unconditional = P(answer | system prompt only)
      IFD_turn = exp((cond_logprob - uncond_logprob) / num_tokens)
    """
    messages = trajectory.get("messages", [])
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_indices:
        return None, None

    # Find system prompt (for unconditional baseline)
    system_msg = None
    for m in messages:
        if m["role"] == "system":
            system_msg = m
            break

    turn_scores = []
    for aidx in assistant_indices:
        try:
            assistant_content = messages[aidx]["content"]
            if len(assistant_content) < 20:  # skip tiny responses
                continue

            # ---- Conditional ----
            cond_text = build_raw_text(messages, up_to_assistant_idx=aidx)
            cond_encoding = tokenizer(
                cond_text, return_offsets_mapping=True, add_special_tokens=False,
                truncation=True, max_length=MAX_MODEL_LENGTH,
            )
            cond_ids = torch.tensor([cond_encoding["input_ids"]], dtype=torch.long)
            cond_offsets = cond_encoding["offset_mapping"]

            # Find assistant span in conditional text
            astart_char = cond_text.rfind(assistant_content)
            if astart_char == -1:
                # Fallback: last 40% of tokens
                astart_token = int(len(cond_ids[0]) * 0.6)
                aend_token = len(cond_ids[0])
            else:
                aend_char = astart_char + len(assistant_content)
                astart_token = None
                aend_token = len(cond_ids[0])
                for i, (cs, ce) in enumerate(cond_offsets):
                    if astart_token is None and cs <= astart_char < ce:
                        astart_token = i
                    if cs < aend_char <= ce:
                        aend_token = i + 1
                        break
                if astart_token is None:
                    astart_token = int(len(cond_ids[0]) * 0.6)

            if aend_token - astart_token < 5:
                continue

            cond_logprob, cond_n = compute_token_logprob(
                model, cond_ids, astart_token, aend_token, device
            )

            # ---- Unconditional (system + assistant only) ----
            uncond_lines = []
            if system_msg:
                uncond_lines.append(f"### system:\n{system_msg['content']}\n")
            uncond_lines.append(f"### assistant:\n{assistant_content}\n")
            uncond_text = "\n".join(uncond_lines)

            uncond_encoding = tokenizer(
                uncond_text, return_offsets_mapping=True, add_special_tokens=False,
                truncation=True, max_length=MAX_MODEL_LENGTH,
            )
            uncond_ids = torch.tensor([uncond_encoding["input_ids"]], dtype=torch.long)
            uncond_offsets = uncond_encoding["offset_mapping"]

            # Assistant text is at the end of unconditional text
            ustart_char = uncond_text.rfind(assistant_content)
            if ustart_char == -1:
                ustart_token = int(len(uncond_ids[0]) * 0.5)
                uend_token = len(uncond_ids[0])
            else:
                uend_char = ustart_char + len(assistant_content)
                ustart_token = None
                uend_token = len(uncond_ids[0])
                for i, (cs, ce) in enumerate(uncond_offsets):
                    if ustart_token is None and cs <= ustart_char < ce:
                        ustart_token = i
                    if cs < uend_char <= ce:
                        uend_token = i + 1
                        break
                if ustart_token is None:
                    ustart_token = int(len(uncond_ids[0]) * 0.5)

            if uend_token - ustart_token < 5:
                continue

            uncond_logprob, uncond_n = compute_token_logprob(
                model, uncond_ids, ustart_token, uend_token, device
            )

            if cond_n == 0 or uncond_n == 0:
                continue

            # Length-normalized IFD
            cond_avg = cond_logprob / cond_n
            uncond_avg = uncond_logprob / uncond_n
            turn_ifd = math.exp(cond_avg - uncond_avg)

            turn_scores.append({
                "turn": aidx,
                "ifd": turn_ifd,
                "cond_tokens": cond_n,
                "uncond_tokens": uncond_n,
            })

        except Exception as e:
            continue

    if not turn_scores:
        return None, None

    avg_ifd = sum(t["ifd"] for t in turn_scores) / len(turn_scores)
    return avg_ifd, turn_scores


def compute_ifd_batch(trajectories, sample_size=MAX_SAMPLE_SIZE):
    """Compute IFD for a batch, with incremental save."""
    model, tokenizer, device = load_scoring_model()

    import random
    n = min(sample_size, len(trajectories)) if sample_size else len(trajectories)
    indices = sorted(random.Random(42).sample(range(len(trajectories)), n))

    results = []
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ifd_scores.json")

    # Resume
    scored_set = set()
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                results = json.load(f)
            scored_set = {r["index"] for r in results}
            print(f"Resuming: {len(results)} already scored")
        except Exception:
            pass

    pending = [(i, trajectories[i]) for i in indices if i not in scored_set]

    t0 = time.time()
    for idx, (orig_idx, traj) in enumerate(pending):
        ifd, details = compute_ifd_single(model, tokenizer, traj, device)
        if ifd is not None:
            results.append({
                "index": orig_idx,
                "ifd_score": round(ifd, 4),
                "num_turns": len(details),
            })
        else:
            results.append({
                "index": orig_idx,
                "ifd_score": None,
                "num_turns": 0,
            })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed * 60
            print(f"  Progress: {idx + 1}/{len(pending)}, ~{rate:.0f}/min")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    n_scored = sum(1 for r in results if r.get("ifd_score") is not None)
    print(f"Scored {n_scored}/{len(results)} in {elapsed:.0f}s "
          f"({len(results) / elapsed * 60:.0f}/min)")
    print(f"Saved to {output_path}")
    return results


def compare_scoring_methods(ifd_results, heuristic_data=None):
    """Print IFD score statistics."""
    valid = [r for r in ifd_results if r.get("ifd_score") is not None]
    if not valid:
        print("ERROR: No valid IFD scores computed.")
        return

    scores = np.array([r["ifd_score"] for r in valid])

    print(f"\n{'='*60}")
    print(f"IFD Scoring Analysis ({len(valid)} trajectories)")
    print(f"{'='*60}")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")

    sorted_scores = np.sort(scores)
    for q in range(5):
        start = q * len(sorted_scores) // 5
        end = (q + 1) * len(sorted_scores) // 5
        seg = sorted_scores[start:min(end, len(sorted_scores))]
        if len(seg):
            print(f"  Q{q + 1}: {seg[0]:.4f} - {seg[-1]:.4f}")

    # Examples
    sorted_valid = sorted(valid, key=lambda x: x["ifd_score"])
    print(f"\n  Bottom-3 (low IFD — instruction adds little):")
    for r in sorted_valid[:3]:
        print(f"    IFD={r['ifd_score']:.4f} | turns={r['num_turns']}")

    print(f"\n  Top-3 (high IFD — instruction is crucial):")
    for r in sorted_valid[-3:]:
        print(f"    IFD={r['ifd_score']:.4f} | turns={r['num_turns']}")

    # Heuristic correlation
    if heuristic_data:
        from data_processing import compute_quality_score
        ifd_map = {r["index"]: r["ifd_score"] for r in valid}
        paired = []
        for i, traj in enumerate(heuristic_data):
            if i in ifd_map:
                paired.append((compute_quality_score(traj), ifd_map[i]))
        if paired:
            h_arr = np.array([p[0] for p in paired])
            i_arr = np.array([p[1] for p in paired])
            corr = np.corrcoef(h_arr, i_arr)[0, 1]
            print(f"\n  Heuristic-IFD correlation: {corr:.4f} ({len(paired)} pairs)")

    return valid


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Model-Based IFD Scoring for DataMind-12K")
    print("=" * 60)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} trajectories")

    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_SAMPLE_SIZE
    print(f"Scoring {sample_size} trajectories...")

    results = compute_ifd_batch(all_data, sample_size=sample_size)
    compare_scoring_methods(results, all_data)
