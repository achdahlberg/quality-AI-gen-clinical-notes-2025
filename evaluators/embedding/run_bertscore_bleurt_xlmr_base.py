"""
BERTScore (xlm-roberta-base) + BLEURT, verbose logs.
"""

import os
import platform
import time
import traceback

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evaluate
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -------- Config --------
BASE_DIR = "X"
OUTPUT_PATH = "Y.csv"

LANG_FOLDERS = ["english_cases", "finnish_cases", "swedish_cases"]
BERTSCORE_LANG_MAP = {"english": "en", "finnish": "fi", "swedish": "sv"}
PERTURBATIONS = ["delete", "modify", "synonyms"]

# BERTScore settings
BERTSCORE_MODEL_TYPE = "xlm-roberta-base"
BERTSCORE_DEVICE = "cpu"  # For mac, change if needed
BERTSCORE_BATCH_SIZE = 1

# BLEURT local-or-hub
BLEURT_LOCAL_DIR = os.path.expanduser("~/hf-models/bleurt-base-128")
BLEURT_HUB_ID = "Elron/bleurt-base-128"


def log(msg: str):
    print(msg, flush=True)


def pick_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        log(f"[READ] Failed to read {path}: {e}")
        return ""


# -------- Load metrics and models --------
start_load = time.time()
log(f"[LOAD] BERTScore model_type: {BERTSCORE_MODEL_TYPE}, device: {BERTSCORE_DEVICE}")
bertscore_metric = evaluate.load("bertscore")

device = pick_device()
log(f"[LOAD] BLEURT device: {device.upper()}")

required = [
    "pytorch_model.bin",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]
local_ok = os.path.isdir(BLEURT_LOCAL_DIR) and all(
    os.path.exists(os.path.join(BLEURT_LOCAL_DIR, f)) for f in required
)
bleurt_source = BLEURT_LOCAL_DIR if local_ok else BLEURT_HUB_ID
local_only = local_ok

if local_ok:
    log(f"[LOAD] Using local BLEURT at {BLEURT_LOCAL_DIR}")
else:
    log("[LOAD] Local BLEURT incomplete, loading from hub")

tokenizer = AutoTokenizer.from_pretrained(bleurt_source, local_files_only=local_only)
bleurt_model = (
    AutoModelForSequenceClassification.from_pretrained(
        bleurt_source, local_files_only=local_only
    )
    .to(device)
    .eval()
)
log(f"[LOAD] Models ready in {time.time() - start_load:.1f}s.")


@torch.inference_mode()
def compute_bleurt_pt(ref: str, hyp: str) -> float:
    inputs = tokenizer(hyp, ref, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = bleurt_model(**inputs).logits
    return float(logits.squeeze().cpu().item())


def compute_bertscore(ref: str, hyp: str, lang_code: str):
    # baseline rescale is not used for xlm-roberta-base
    res = bertscore_metric.compute(
        predictions=[hyp],
        references=[ref],
        lang=lang_code,
        model_type=BERTSCORE_MODEL_TYPE,
        device=BERTSCORE_DEVICE,
        batch_size=BERTSCORE_BATCH_SIZE,
        rescale_with_baseline=False,
    )
    return float(res["precision"][0]), float(res["recall"][0]), float(res["f1"][0])


def main():
    rows = []

    existing_langs = [
        lf for lf in LANG_FOLDERS if os.path.isdir(os.path.join(BASE_DIR, lf))
    ]
    log(f"[SCAN] Languages found: {existing_langs or 'none'}")

    for lang_folder in existing_langs:
        lang_path = os.path.join(BASE_DIR, lang_folder)
        language = lang_folder.replace("_cases", "")
        bert_lang = BERTSCORE_LANG_MAP.get(language, "en")
        log(
            f"\n=== Language: {language} (BERTScore lang={bert_lang}, model={BERTSCORE_MODEL_TYPE}) ==="
        )

        case_dirs = [
            d
            for d in os.listdir(lang_path)
            if os.path.isdir(os.path.join(lang_path, d))
        ]
        log(f"[SCAN] Found {len(case_dirs)} case folders in {lang_folder}.")

        for idx, case_dir in enumerate(case_dirs, start=1):
            case_path = os.path.join(lang_path, case_dir)
            log(f"\n[CASE {idx}/{len(case_dirs)}] {case_dir}")

            orig_dir = os.path.join(case_path, "original")
            if not os.path.isdir(orig_dir):
                log(f"[SKIP] No 'original' folder in {case_dir}")
                continue

            orig_files = [f for f in os.listdir(orig_dir) if f.endswith(".csv")]
            if not orig_files:
                log(f"[SKIP] No CSV in {orig_dir}")
                continue

            orig_file = orig_files[0]
            orig_path = os.path.join(orig_dir, orig_file)
            log(f"[READ] Original file: {orig_path}")
            orig_text = read_text(orig_path)
            if not orig_text:
                log(f"[SKIP] Empty original in {orig_path}")
                continue

            try:
                log("[METRIC] Baseline BERTScore + BLEURT (original vs original)")
                p, r, f1 = compute_bertscore(orig_text, orig_text, bert_lang)
                bleurt = compute_bleurt_pt(orig_text, orig_text)
                rows.append(
                    {
                        "case": case_dir,
                        "language": language,
                        "type": "original",
                        "file": orig_file,
                        "bertscore_model": BERTSCORE_MODEL_TYPE,
                        "bertscore_precision": p,
                        "bertscore_recall": r,
                        "bertscore_f1": f1,
                        "bleurt": bleurt,
                        "bleurt_source": bleurt_source,
                        "device_bleurt": device,
                        "device_bertscore": BERTSCORE_DEVICE,
                    }
                )
                log(f"[DONE] Baseline OK. F1={f1:.4f}, BLEURT={bleurt:.4f}")
            except Exception:
                log("[ERROR] Baseline failed:")
                log(traceback.format_exc())

            for variation in PERTURBATIONS:
                vpath = os.path.join(case_path, variation)
                if not os.path.isdir(vpath):
                    log(f"[SKIP] No '{variation}' folder in {case_dir}")
                    continue
                vfiles = sorted([f for f in os.listdir(vpath) if f.endswith(".csv")])
                log(f"[SCAN] {variation}: {len(vfiles)} files.")
                for v_i, file in enumerate(vfiles, start=1):
                    path = os.path.join(vpath, file)
                    log(f"[READ] ({variation} {v_i}/{len(vfiles)}) {path}")
                    txt = read_text(path)
                    if not txt:
                        log(f"[SKIP] Empty text in {path}")
                        continue
                    try:
                        log("[METRIC] BERTScore + BLEURT")
                        p, r, f1 = compute_bertscore(orig_text, txt, bert_lang)
                        bleurt = compute_bleurt_pt(orig_text, txt)
                        rows.append(
                            {
                                "case": case_dir,
                                "language": language,
                                "type": variation,
                                "file": file,
                                "bertscore_model": BERTSCORE_MODEL_TYPE,
                                "bertscore_precision": p,
                                "bertscore_recall": r,
                                "bertscore_f1": f1,
                                "bleurt": bleurt,
                                "bleurt_source": bleurt_source,
                                "device_bleurt": device,
                                "device_bertscore": BERTSCORE_DEVICE,
                            }
                        )
                        log(f"[DONE] {file} OK. F1={f1:.4f}, BLEURT={bleurt:.4f}")
                    except Exception:
                        log(f"[ERROR] Metrics failed for {file}:")
                        log(traceback.format_exc())

    log("\n[SAVE] Writing CSV...")
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    log(f"[SAVE] Wrote {len(rows)} rows to {OUTPUT_PATH}")
    log("=== Run finished ===")


if __name__ == "__main__":
    main()
