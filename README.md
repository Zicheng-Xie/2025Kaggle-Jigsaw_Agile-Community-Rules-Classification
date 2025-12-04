# 2025Kaggle-Jigsaw_Agile-Community-Rules-Classification
A _**Bronze Medal (174th/2437 teams)**_ implementation for 2025 Kaggle Jigsaw-Agile Community Rules Classification Competition

**Submitted implementation**: fork-of-mocode.ipynb

**Training implementation**: fork-of-mocode.ipynb (the same notebook, further refined during the preparation of the final submission)

**Baseline 1**: https://www.kaggle.com/code/wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct

**Baseline 2:** jigsaw-speed-run-10-min-triplet-and-faiss-af5e7a.ipynb

In this Jigsaw Kaggle competition, I adopted a “two-route + ensemble” approach: an instruction-tuned LLM classifier and contrastive sentence-embedding models, whose outputs are combined by rank-based weighted fusion. This design aims to balance the LLM’s strong contextual understanding with embedding models’ stable rule-centric similarity modeling, especially under unseen rules and low-data conditions.

For the LLM route, I fine-tuned Llama 3.2 3B Instruct with LoRA as a binary “Yes/No” rule-violation classifier. I unified data from train.csv and pseudo-labeled examples from test.csv into a single prompt format that includes system instructions, subreddit, rule text, positive/negative demonstrations, and the current comment. To satisfy Kaggle’s computational limits, I used 4-bit quantization, parameter-efficient updates on attention and feed-forward layers, and vLLM-based constrained decoding over the {Yes, No} space to obtain calibrated violation probabilities.

For the embedding route, I used BGE-large and BGE-base within a contrastive learning framework, treating each rule as an anchor and its compliant/violating examples as positive/negative samples in triplets. After fine-tuning, I built rule-level compliant and violating centroids in embedding space and scored each test comment by its relative distance to these centroids, optionally augmented with labeled training data for domain adaptation. Finally, I applied rank normalization to the outputs of the LLM and the two embedding models, and formed a weighted ensemble (0.5 for the LLM, 0.25 for each embedding model) to produce the final submission, effectively performing a ranking-based vote across diverse yet complementary models.


## Jigsaw Rule Violation Detection: Method Overview

### 1. Two-Route + Multi-Model Ensemble Strategy

In this Jigsaw Kaggle competition, I followed a **“two-route + multi-model ensemble”** strategy:

- **Route 1:** An **instruction-tuned large language model (LLM) classifier**  
- **Route 2:** A **sentence-embedding–based similarity model** trained with **contrastive learning**

All model outputs are combined using a **rank-based weighted ensemble**, so that the final system can balance:

- **Cross-rule generalization**
- **Fine-grained semantic discrimination**

---

### 2. Route 1 – Instruction-Tuned LLM Classifier (Model 1)

For the LLM route, I used **Llama 3.2 3B Instruct** as the base model and applied **LoRA** for parameter-efficient fine-tuning, turning it into a binary classifier that directly outputs **“Yes/No”** for rule violation.

#### 2.1 Unified Data Construction

I built a unified data format using both `train.csv` and part of `test.csv`:

- **Training set (`train.csv`):**
  - Loaded fields: `body`, `rule`, `subreddit`, `rule_violation`
  - Randomly attached **positive and negative examples** as in-context demonstrations
  - Combined with **system prompts** and the **current comment** to form a complete prompt

- **Test set (`test.csv`):**
  - Converted example pairs into **pseudo-labeled samples**:
    - “Violating examples” → label **1**
    - “Non-violating examples” → label **0**
  - This **expanded the training set** and improved coverage of **unseen rules**

#### 2.2 Training and Inference Settings

To satisfy Kaggle’s **12-hour** and **GPU memory** constraints:

- Used **4-bit quantization (nf4)** with `bitsandbytes` and **DeepSpeed ZeRO Stage 2**
- Applied **LoRA only** to key attention and feed-forward matrices

For inference:

- Deployed via **vLLM**, loading the base model plus LoRA weights
- Constrained the output to **“Yes”** or **“No”** using a **multiple-choice logits processor**
- Sampled **only one token**, then:
  - Took its **log probabilities**
  - Converted them into **violation probabilities** via softmax
- Parallelized inference across **two GPUs**

This produced **`submission5.csv`** as the **main LLM classifier output**.

---

### 3. Route 2 – Embedding-Based Similarity Models (Model 2 & 3, baseline1)

For the embedding-based route (Model 2 and Model 3), I designed a **contrastive learning framework** that explicitly models the relationship between **rules** and their **positive/negative examples** in embedding space.

#### 3.1 Text Normalization and URL Handling

All texts were normalized with a `cleaner` function:

- **URLs** were replaced by a compact form  
  `"<url>: (domain/path)"`  
  to:
  - Preserve semantic information
  - Reduce noise and sequence length

#### 3.2 Triplet Construction

I treated:

- The **rule text** as the **anchor**
- Comments labeled as `negative_example` (**compliant**) as the **positive** sample
- Comments labeled as `positive_example` (**violating**) as the **negative** sample

This formed triplets of the form:

```text
(rule, compliant_comment, violating_comment)
