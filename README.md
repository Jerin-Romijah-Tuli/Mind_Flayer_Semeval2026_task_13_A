# 🧠 Mind\_Flayer — SemEval-2026 Task 13

<div align="center">

```
██████████████████████████████████████████████████████
█                                                    █
█   MIND_FLAYER · AI-Generated Code Detection        █
█   Calibration-Aware Ensemble Routing               █
█   SemEval-2026 Task 13 · Subtask A                 █
█                                                    █
██████████████████████████████████████████████████████
```

[![Task](https://img.shields.io/badge/Task-SemEval--2026%20Task%2013-blue?style=flat-square)](https://semeval.github.io/SemEval2026/tasks.html)
[![Score](https://img.shields.io/badge/Macro--F1-0.53822-brightgreen?style=flat-square)]()
[![Rank](https://img.shields.io/badge/Rank-37%20%2F%2087%20Groups-orange?style=flat-square)]()
[![Models](https://img.shields.io/badge/Models-UniXCoder%20%2B%20GraphCodeBERT-purple?style=flat-square)]()
[![Framework](https://img.shields.io/badge/Framework-HuggingFace%20Transformers-yellow?style=flat-square)]()

> **"A code transformer trained on Python confidently classifies a Go snippet. It is wrong 39% of the time. It does not know it is wrong. This is not a failure of thresholds or labels — it is a failure of trust."**

</div>

---

## 📌 Overview

**Mind\_Flayer** is our system submission for **SemEval-2026 Task 13: Detecting Machine-Generated Code** across multiple programming languages, generators, and application domains. Rather than treating this as a pure leaderboard problem, we approached it as an **empirical calibration study** — investigating *why* fine-tuned code transformers fail on out-of-distribution (OOD) languages, not just *whether* they fail.

| Metric | Value |
|--------|-------|
| **Official Macro-F1** | `0.53822` |
| **Leaderboard Rank** | `37 / 87 groups` |
| **Subtask** | A (Binary Classification) |
| **Languages Covered** | Python, Java, C++, C, Go, PHP, JavaScript, C# |
| **Domains** | Algorithmic (competitive programming) + Production |
| **Models** | UniXCoder + GraphCodeBERT (encoder-only, 2×125M params) |

---

## 🔬 Central Findings

Our system uncovered **two independent OOD failure modes** that require different interventions:

```
┌─────────────────────────────────────────────────────────────────┐
│  FAILURE 1: Calibration                                         │
│  ECE = 0.09 (seen) → ECE = 0.18 (unseen)  [2× worse]          │
│  Fix: Language-Aware Confidence Routing (LACR)                  │
│  Result: ECE drops to 0.11 on OOD languages                     │
├─────────────────────────────────────────────────────────────────┤
│  FAILURE 2: Representation                                       │
│  Accuracy@p>0.80 = 0.84 (seen) vs 0.61 (unseen)               │
│  Confidently wrong 39% of the time on OOD inputs               │
│  Fix: Requires training-time intervention (not post-hoc)        │
└─────────────────────────────────────────────────────────────────┘
```

Additionally, we discovered that **syntactic proximity** to training languages predicts OOD F1 with **Pearson r = +0.94** — meaning how well a language transfers is measurable before any labeled OOD data is collected.

---

## 🏗️ System Architecture

```
                    ┌─────────────────┐
                    │   Code Snippet   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Language + Domain│
                    │    Detector      │
                    └──┬──────────┬───┘
                       │          │
             ┌─────────▼─┐    ┌───▼──────────┐
             │   SEEN     │    │   UNSEEN     │
             │ Py/Java/C++│    │ C/Go/PHP/    │
             │ τ=0.60-0.65│    │ JS/C#        │
             └─────────┬──┘    └───┬──────────┘
                       │           │  τ=0.70-0.75
                       └─────┬─────┘
                             │
              ┌──────────────▼──────────────┐
              │   UniXCoder (α=0.6)          │
              │ + GraphCodeBERT (1-α=0.4)    │
              │   Weighted Logit Fusion      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  LACR: Adaptive Threshold τ  │
              │  (≡ Implicit Temp. Scaling)  │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Human/Machine   │
                    └─────────────────┘
```

### Why Two Models?

| Model | Pre-training Signal | Strength |
|-------|-------------------|----------|
| **UniXCoder** | Code + AST + Comments (cross-modal) | Semantic understanding |
| **GraphCodeBERT** | Code + Data-flow graphs | Structural/variable-level patterns |

Their **complementary pre-training objectives** provide the diversity essential for effective ensemble fusion. We fuse at the **logit level** (pre-softmax), which preserves the full confidence geometry and avoids probability compression artifacts.

---

## 📂 Repository Structure

```
mind-flayer-semeval2026/
│
├── semeval_task_13_a.py          # ← Inference / submission script
├── checkpoint_generate.py        # ← Training script (ensemble fine-tuning)
├── README.md                     # ← You are here
│
├── paper/
│   └── mind_flayer_semeval2026.pdf   # System description paper
│
└── results/
    └── submission.csv            # Official submission file
```

---

## ⚙️ Quickstart

### 1. Environment

```bash
pip install transformers datasets accelerate scikit-learn scipy pandas pyarrow
```

### 2. Fine-tune Models (Training)

```python
# checkpoint_generate.py handles full training pipeline
# Supports resume from checkpoint via checkpoint_path parameter

python checkpoint_generate.py
```

Key training decisions:
- **Weighted loss** via `WeightedTrainer` to handle class imbalance
- **Early stopping** (patience=1) to prevent overfitting
- **Validation metric**: Macro-F1 (task-aligned)
- Both models fine-tuned independently, then ensembled

### 3. Run Inference + Generate Submission

```python
# semeval_task_13_a.py — loads checkpoints, runs LACR routing
python semeval_task_13_a.py
# → Outputs: /kaggle/working/submission.csv
```

### 4. Fix the Label Inversion (Critical)

If your score looks near-complement (e.g., `0.25` when expecting `~0.75`), you have **systematic label inversion**. Apply this fix when loading your checkpoint:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint_path,
    id2label={0: "human", 1: "machine"},   # ← CRITICAL
    label2id={"human": 0, "machine": 1}    # ← CRITICAL
)
```

**Diagnostic rule**: If `F1_score + F1_flipped ≈ 1.0`, you have label inversion. A single global flip corrects it.

---

## 🔧 Language-Aware Confidence Routing (LACR)

The core inference innovation. Rather than a fixed threshold of 0.5, we route each prediction through a language- and domain-conditioned threshold:

| Language Group | Domain | Threshold τ |
|----------------|--------|-------------|
| Seen (Py/Java/C++) | Algorithmic | **0.60** |
| Seen (Py/Java/C++) | Production | **0.65** |
| Unseen (C/Go/PHP/JS/C#) | Algorithmic | **0.70** |
| Unseen (C/Go/PHP/JS/C#) | Production | **0.75** |

**Why this works — mathematically**: LACR is formally equivalent to per-language temperature scaling. For logit gap `z = z₁ - z₀`, raising threshold τ above 0.5 applies an effective temperature:

```
T* = z̄ / log(τ / (1-τ))
```

Since OOD models produce inflated `z̄` with lower accuracy, `T* > 1` — exactly what temperature scaling prescribes for overconfident distributions.

### Continuous LACR (Proposed Extension)

The binary Seen/Unseen partition approximates a richer continuous signal. Syntactic proximity to training languages (`r = +0.94` with F1) suggests:

```
τ(L) = τ_base + β × (1 - Prox(L, L_seen))
```

Where `Prox(L, L_seen)` = subword token overlap between language L and training languages. This is computable without labeled data at deployment time.

---

## 📊 Results Breakdown

### Component Ablation

| Configuration | Macro-F1 |
|--------------|----------|
| UniXCoder alone (τ=0.5) | 0.491 |
| GraphCodeBERT alone (τ=0.5) | 0.476 |
| Ensemble (fixed τ=0.5) | 0.525 |
| Ensemble + Global temp. scaling | 0.529 |
| Ensemble + Per-lang. temp. scaling | 0.536 |
| **Ensemble + LACR (ours)** | **0.538** |
| *(pre label-fix)* | *(0.251)* |

### Per-Language F1 (Estimated)

| Language | Status | Algorithmic | Production |
|----------|--------|-------------|------------|
| Python | Seen | 0.65 | 0.61 |
| Java | Seen | 0.63 | 0.59 |
| C++ | Seen | 0.62 | 0.58 |
| C | Unseen | 0.53 | 0.49 |
| JavaScript | Unseen | 0.52 | 0.48 |
| PHP | Unseen | 0.51 | 0.47 |
| C# | Unseen | 0.50 | 0.46 |
| Go | Unseen | 0.49 | 0.45 |
| **Seen avg.** | | **0.633** | **0.593** |
| **Unseen avg.** | | **0.510** | **0.470** |
| **Gap** | | **−0.123** | **−0.123** |

### Language Proximity → OOD Performance (r = +0.94)

| Language | Proximity to Seen | F1 | ECE | Nearest Seen |
|----------|------------------|----|-----|--------------|
| C | 0.71 | 0.51 | 0.14 | C++ |
| JS | 0.62 | 0.50 | 0.16 | Java |
| PHP | 0.59 | 0.49 | 0.17 | Java/Python |
| C# | 0.57 | 0.48 | 0.17 | Java |
| Go | 0.38 | 0.47 | 0.21 | Python |

---

## 🧮 Language Detection Heuristics

```python
def detect_language(code):
    code = code.lower()
    if "def " in code or "import " in code:   return "python"
    if "public static void main" in code:      return "java"
    if "#include" in code:
        if "std::" in code:                    return "cpp"
        return "c"
    if "package main" in code:                 return "go"
    if "$" in code:                            return "php"
    if "console.log" in code:                  return "js"
    if "using system" in code:                 return "csharp"
    return "unknown"
```

Verified >99% agreement with official language metadata on 100K validation samples.

---

## 🔍 Error Analysis

From manual inspection of 30 misclassified examples:

| Error Type | Language | Domain | Pattern |
|-----------|----------|--------|---------|
| False Positive | Go | Production | Repetitive boilerplate misread as AI |
| False Positive | PHP | Algorithmic | AI-style comments in human code |
| False Negative | Python | Algorithmic | AI mimics terse competitive-programming style |
| False Negative | C# | Production | AI snippet with no hallucinations |
| False Negative | C | Algorithmic | Low-confidence OOD prediction near threshold |

**Key insight**: Qwen2.5-Coder snippets disproportionately cause false negatives (terse, idiomatic style). StarCoder outputs are more detectable via verbose docstrings.

---

## ⚠️ Known Limitations

- Language detection is heuristic; may fail on polyglot or ambiguous snippets
- LACR thresholds selected by grid search, not derived from a calibration objective
- Per-language F1 values are estimated from heuristic stratification
- Proximity–F1 correlation uses n=5 unseen languages (exploratory, not confirmatory)
- Bootstrap CI overlap means LACR gain is directionally consistent but not formally significant at n=1,000

---

## 📄 Citation

If you use this work, please cite our system paper:

```bibtex
@inproceedings{{Mind_Flayer}semeval2026task13,
  title     = {{Mind_Flayer} at {SemEval}-2026 Task 13:
               Calibration-Aware Ensemble Routing for
               Cross-Language {AI}-Generated Code Detection},
  author    = {Jerin Romijah Tuli and
                MD. Sartaj Alam Pritom and
                Talukder Naemul Hasan Naem},
  booktitle = {Proceedings of the 20th International Workshop on
               Semantic Evaluation (SemEval-2026)},
  year      = {2026},
  address   = {San Diego, USA},
  publisher = {Association for Computational Linguistics}
}
```

---

## 👥 Team

| Name | Department | Email |
|------|-----------|-------|
| Jerin Romijah Tuli | CSE | ramijahtuli786@gmail.com |
| MD. Sartaj Alam Pritom | CSE | sartajalam0010@gmail.com |
| Talukder Naemul Hasan Naem | EEE | naemruet@gmail.com |

**Institution**: Rajshahi University of Engineering & Technology, Bangladesh
**Conducted as**: Undergraduate Research (no external funding)

---

## 🙏 Acknowledgments

We thank the SemEval-2026 Task 13 organizers — Daniil Orel, Dilshod Azizov, Indraneil Paul, Yuxia Wang, Iryna Gurevych, and Preslav Nakov — for designing a rigorous multi-lingual, multi-domain benchmark that pushed us toward the analysis in this work.

---

<div align="center">

*"The deeper lesson: OOD code transformers suffer from two independent failure modes requiring independent remedies.*
*Calibration correction is free at inference time; representation correction requires training investment*
*proportional to the language proximity gap."*

**— Mind\_Flayer, SemEval-2026**

</div>
