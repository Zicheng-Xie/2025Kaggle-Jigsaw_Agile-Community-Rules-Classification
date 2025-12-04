# 2025Kaggle-Jigsaw_Agile-Community-Rules-Classification
A _**Bronze Medal (174th/2437 teams)**_ implementation for 2025 Kaggle Jigsaw-Agile Community Rules Classification Competition

**Submitted implementation**: fork-of-mocode.ipynb

**Training implementation**: fork-of-mocode.ipynb (the same notebook, further refined during the preparation of the final submission)

**Baseline 1**: https://www.kaggle.com/code/wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct

**Baseline 2:** jigsaw-speed-run-10-min-triplet-and-faiss-af5e7a.ipynb

In this Jigsaw Kaggle competition, I adopted a “two-route + ensemble” approach: an instruction-tuned LLM classifier and contrastive sentence-embedding models, whose outputs are combined by rank-based weighted fusion. This design aims to balance the LLM’s strong contextual understanding with embedding models’ stable rule-centric similarity modeling, especially under unseen rules and low-data conditions.

For the LLM route, I fine-tuned Llama 3.2 3B Instruct with LoRA as a binary “Yes/No” rule-violation classifier. I unified data from train.csv and pseudo-labeled examples from test.csv into a single prompt format that includes system instructions, subreddit, rule text, positive/negative demonstrations, and the current comment. To satisfy Kaggle’s computational limits, I used 4-bit quantization, parameter-efficient updates on attention and feed-forward layers, and vLLM-based constrained decoding over the {Yes, No} space to obtain calibrated violation probabilities.

For the embedding route, I used BGE-large and BGE-base within a contrastive learning framework, treating each rule as an anchor and its compliant/violating examples as positive/negative samples in triplets. After fine-tuning, I built rule-level compliant and violating centroids in embedding space and scored each test comment by its relative distance to these centroids, optionally augmented with labeled training data for domain adaptation. Finally, I applied rank normalization to the outputs of the LLM and the two embedding models, and formed a weighted ensemble (0.5 for the LLM, 0.25 for each embedding model) to produce the final submission, effectively performing a ranking-based vote across diverse yet complementary models.
