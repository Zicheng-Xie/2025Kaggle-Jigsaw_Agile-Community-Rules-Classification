# 2025Kaggle-Jigsaw_Agile-Community-Rules-Classification
A _**Bronze Medal (174th/2437 teams)**_ implementation for 2025 Kaggle Jigsaw-Agile Community Rules Classification Competition

**Submitted implementation**: fork-of-mocode.ipynb

**Training implementation**: fork-of-mocode.ipynb (the same notebook, further refined during the preparation of the final submission)

**Baseline 1**: https://www.kaggle.com/code/wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct

**Baseline 2:** jigsaw-speed-run-10-min-triplet-and-faiss-af5e7a.ipynb

# Jigsaw Rule Violation Detection – Final Solution

![python](https://img.shields.io/badge/python-3.10+-blue.svg)
![kaggle](https://img.shields.io/badge/Kaggle-Jigsaw_Rule_Violation_Detection-20BEFF.svg)
![model](https://img.shields.io/badge/Model-LLM_%2B_Contrastive_Embeddings-orange.svg)
![status](https://img.shields.io/badge/Status-Final_Submission-brightgreen.svg)

---

## 🎓 Introduction

This repository contains my solution for the **Jigsaw Rule Violation Detection** Kaggle competition.

The goal is to determine whether a given **comment violates a specific subreddit rule**, and to produce predictions suitable for the competition’s evaluation setup.

To tackle the task, I adopt a **“two-route + multi-model ensemble”** strategy:

> **Route 1 – Instruction-tuned LLM classifier**  
> **Route 2 – Embedding-based similarity models with contrastive learning**  

All model outputs are combined through a **rank-based weighted ensemble**, designed to balance:

- **Cross-rule generalization** (handle new rules and subreddits)
- **Fine-grained semantic discrimination** (distinguish borderline violations)

---

## 🧭 Two-Route + Multi-Model Ensemble Strategy

The overall pipeline is:

> **Data preparation → Route 1 (LLM) → Route 2 (Embeddings) → Rank-based weighted ensemble → Submission**

- **Route 1** focuses on **direct classification** using an instruction-tuned LLM that outputs “Yes/No” for rule violation.
- **Route 2** focuses on **semantic similarity** between a rule and its positive / negative examples using **contrastive learning** in embedding space.
- The final predictions are obtained via a **rank-based fusion** of all models (LLM classifier + embedding models), which improves robustness and stability across different rules and subreddits.

---

## 🤖 Route 1 – Instruction-Tuned LLM Classifier (Model 1)

For the LLM route, I use **Llama 3.2 3B Instruct** as the base model and apply **LoRA** for parameter-efficient fine-tuning, turning it into a **binary classifier** that directly outputs **“Yes/No”** for rule violation.

### 2.1 Unified Data Construction

To train the LLM effectively, I build a **unified data format** using both `train.csv` and **part of** `test.csv`.

**Training set (`train.csv`)**

- Loaded fields:
  - `body`
  - `rule`
  - `subreddit`
  - `rule_violation`
- For each training instance:
  - Randomly attach **positive and negative examples** as **in-context demonstrations**.
  - Combine:
    - System prompts
    - Current rule
    - Subreddit
    - Comment body
    - Selected demonstrations  
    into a **single instruction-style prompt**.

**Pseudo-labeled data from `test.csv`**

- Use `test.csv` example pairs to create additional training samples:
  - “Violating examples” → label **1**
  - “Non-violating examples” → label **0**
- These pseudo-labels:
  - Expand the effective training set
  - Improve coverage of **unseen rules** and subreddits

### 2.2 Training and Inference Settings

To satisfy Kaggle’s **12-hour** time limit and **GPU memory** constraints:

- **Parameter-efficient fine-tuning**
  - Use **4-bit quantization (nf4)** via `bitsandbytes`
  - Use **DeepSpeed ZeRO Stage 2** for memory efficiency
  - Apply **LoRA** only to:
    - Key attention matrices
    - Feed-forward layers

- **Inference setup**
  - Deploy via **vLLM**, loading:
    - The base Llama 3.2 3B Instruct model
    - LoRA weights
  - Constrain outputs to `"Yes"` or `"No"` using a **multiple-choice logits processor**.
  - For each prompt:
    1. Sample **one token** (`Yes`/`No`)
    2. Read its **log-probabilities**
    3. Convert them into **violation probabilities** via softmax
  - Parallelize inference on **two GPUs** for faster throughput.

- The primary output of this route is **`submission5.csv`**, which contains the LLM classifier predictions.

---

## 🔗 Route 2 – Embedding-Based Similarity Models  
*(Model 2 & 3, baseline1)*

For the second route, I design **contrastive learning–based embedding models** that explicitly model the relationship between:

- A **rule**  
- Its **compliant examples** (non-violating)
- Its **violating examples**

The idea is to learn an embedding space where:

- Compliant comments are **closer to the rule**
- Violating comments are **pushed away** from the rule

These models serve as **Model 2**, **Model 3**, and **baseline1** in the final ensemble.

### 3.1 Text Normalization & URL Handling

All texts are passed through a **normalization / cleaning function**:

- URLs are replaced by a compact placeholder:
  ```text
  "<url>: (domain/path)"
This design:

Preserves semantic information (which domain / path is referenced)

Reduces noise and sequence length

Makes sequences more consistent across different comments

### 3.2 Triplet Construction (Contrastive Learning)

To train the embedding models, I construct **triplets** of the form:

- **Anchor**: the *rule text*
- **Positive sample**: comments labeled as **`negative_example`** (i.e., **compliant**)
- **Negative sample**: comments labeled as **`positive_example`** (i.e., **violating**)

The training objective encourages:

- `Embedding(rule)` to stay **closer** to `Embedding(compliant_comment)`
- `Embedding(rule)` to be **farther** from `Embedding(violating_comment)`


In this Jigsaw Kaggle competition, I adopted a “two-route + ensemble” approach: an instruction-tuned LLM classifier and contrastive sentence-embedding models, whose outputs are combined by rank-based weighted fusion. This design aims to balance the LLM’s strong contextual understanding with embedding models’ stable rule-centric similarity modeling, especially under unseen rules and low-data conditions.

For the LLM route, I fine-tuned Llama 3.2 3B Instruct with LoRA as a binary “Yes/No” rule-violation classifier. I unified data from train.csv and pseudo-labeled examples from test.csv into a single prompt format that includes system instructions, subreddit, rule text, positive/negative demonstrations, and the current comment. To satisfy Kaggle’s computational limits, I used 4-bit quantization, parameter-efficient updates on attention and feed-forward layers, and vLLM-based constrained decoding over the {Yes, No} space to obtain calibrated violation probabilities.

For the embedding route, I used BGE-large and BGE-base within a contrastive learning framework, treating each rule as an anchor and its compliant/violating examples as positive/negative samples in triplets. After fine-tuning, I built rule-level compliant and violating centroids in embedding space and scored each test comment by its relative distance to these centroids, optionally augmented with labeled training data for domain adaptation. Finally, I applied rank normalization to the outputs of the LLM and the two embedding models, and formed a weighted ensemble (0.5 for the LLM, 0.25 for each embedding model) to produce the final submission, effectively performing a ranking-based vote across diverse yet complementary models.



This formed triplets of the form:

```text
(rule, compliant_comment, violating_comment)
