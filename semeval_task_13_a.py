
# ============================================================
# SEMEVAL 2026 TASK 13 — LANGUAGE & DOMAIN AWARE ENSEMBLE
# TARGET: 0.60–0.70 MACRO-F1
# UniXCoder + GraphCodeBERT
# ============================================================

from gettext import install
from gettext import install
import os

os.environ["WANDB_DISABLED"] = "true"

import re
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from scipy.special import softmax

# -----------------------------
# CONFIG
# -----------------------------
TEST_PATH = "/kaggle/input/test-p/test.parquet"

UNICHECKPOINT   = "//kaggle/input/unixcoder/results_unixcoder-base/checkpoint-41668"
GRAPHCHECKPOINT = "/kaggle/input/graphcodebert/results_graphcodebert-base/checkpoint-41668"

MAX_LEN = 384

W_UNI   = 0.6
W_GRAPH = 0.4

# -----------------------------
# LANGUAGE SETS
# -----------------------------
SEEN_LANGS   = {"python", "java", "cpp"}
UNSEEN_LANGS = {"c", "go", "php", "js", "csharp"}

# -----------------------------
# LOAD TEST DATA
# -----------------------------
test_df = pd.read_parquet(TEST_PATH)
test_df["code"] = test_df["code"].astype(str)
assert "ID" in test_df.columns

# -----------------------------
# LANGUAGE DETECTION (ROBUST)
# -----------------------------
def detect_language(code):
    code = code.lower()
    if "def " in code or "import " in code:
        return "python"
    if "public static void main" in code:
        return "java"
    if "#include" in code:
        if "std::" in code:
            return "cpp"
        return "c"
    if "package main" in code:
        return "go"
    if "$" in code:
        return "php"
    if "console.log" in code:
        return "js"
    if "using system" in code:
        return "csharp"
    return "unknown"

# -----------------------------
# DOMAIN DETECTION
# -----------------------------
def detect_domain(code):
    code = code.lower()
    if any(k in code for k in ["leetcode", "stdin", "stdout", "int main", "scanner"]):
        return "algorithmic"
    return "production"

test_df["lang"] = test_df["code"].apply(detect_language)
test_df["domain"] = test_df["code"].apply(detect_domain)

# -----------------------------
# DATASET BUILDER
# -----------------------------
def build_dataset(tokenizer):
    def tok(batch):
        return tokenizer(batch["code"], truncation=True, padding="max_length", max_length=MAX_LEN)
    ds = Dataset.from_pandas(test_df[["code"]])
    ds = ds.map(tok, batched=True, remove_columns=["code"])
    ds.set_format("torch", columns=["input_ids", "attention_mask"])
    return ds

# -----------------------------
# MODEL INFERENCE
# -----------------------------
tokenizer_u = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model_u = AutoModelForSequenceClassification.from_pretrained(UNICHECKPOINT)
trainer_u = Trainer(model=model_u)
logits_u = trainer_u.predict(build_dataset(tokenizer_u)).predictions

tokenizer_g = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model_g = AutoModelForSequenceClassification.from_pretrained(GRAPHCHECKPOINT)
trainer_g = Trainer(model=model_g)
logits_g = trainer_g.predict(build_dataset(tokenizer_g)).predictions

# -----------------------------
# ENSEMBLE
# -----------------------------
ensemble_logits = W_UNI * logits_u + W_GRAPH * logits_g
machine_probs = softmax(ensemble_logits, axis=1)[:, 1]

# -----------------------------
# ROUTED THRESHOLDING (CRITICAL)
# -----------------------------
final_preds = []

for p, lang, dom in zip(machine_probs, test_df["lang"], test_df["domain"]):

    if lang in SEEN_LANGS and dom == "algorithmic":
        threshold = 0.60
    elif lang in SEEN_LANGS:
        threshold = 0.65
    elif lang in UNSEEN_LANGS and dom == "algorithmic":
        threshold = 0.70
    else:
        threshold = 0.75

    final_preds.append(1 if p >= threshold else 0)

final_preds = np.array(final_preds)

print("Prediction distribution:", np.bincount(final_preds))

# -----------------------------
# SUBMISSION
# -----------------------------
submission = pd.DataFrame({
    "ID": test_df["ID"].values,
    "label": final_preds
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ submission.csv ready")
print(submission.head(10))
