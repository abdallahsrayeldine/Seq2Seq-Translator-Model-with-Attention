# French↔English Seq2Seq Translator with Attention 🚀

**Bidirectional French↔English Seq2Seq (Bi-GRU encoder + GRU decoder + attention). Trained on ~210k Tatoeba pairs (vocab=5k). Token-level masked accuracy ≈ **80% (Fr→En)** and **≈77% (En→Fr)**. Attention maps show sensible alignments.** 🎯 

## Quick start

* Clone, install requirements, open the notebook.
* Prepare `tatoeba.tsv` in `data/`.
* Run preprocessing → `train.ipynb` → evaluate.
* Batches: 64, epochs up to 35, early stopping (patience=3).

## Dataset & preprocessing 🧹

* Source: Tatoeba French–English (CC BY 2.0).
* ~210k sentence pairs; ASCII NFKD, lowercase, punctuation tokenized, [START]/[END] added.
* Vocab capped at **5,000**; rare tokens → `[UNK]`.
* Split: 80% train / 20% val. Bidirectional use: same pairs swapped to double training signal.

## Model (compact)

* **Encoder:** Embedding(5k,256) + bidirectional GRU (256, sum merge).
* **Attention:** Single-head MultiHeadAttention + residual + LayerNorm.
* **Decoder:** Embedding + unidirectional GRU (256) + cross-attention → Dense logits.
* ≈ **10M parameters**.

## Training ⚙️

* Loss: masked SparseCategoricalCrossentropy (ignore padding).
* Optimizer: Adam (lr=0.001).
* EarlyStopping on val loss; validate every 100 batches.

## Results & analysis 📈

* Masked token accuracy: **~80% (Fr→En)**, **~77% (En→Fr)**.
* BLEU-style checks show clear improvement over random baseline.
* Errors: `[UNK]`, word-order issues in long sentences, article/conjugation mistakes.

## Interpretability

* Attention heatmaps show strong alignments (diagonals for short sentences). Single head is informative.

## Limitations & next steps 🔧

* Limited vocab → many `[UNK]`.
* RNNs struggle on long-range dependencies.
* Recommended upgrades: subword tokenization (BPE), Transformer encoder/decoder, multilingual pretraining, stronger regularization (dropout/layernorm).

## Environment

* TensorFlow 2.18.0, TensorFlow-Text 2.18.1, Python 3.x. Seeds fixed for reproducibility.
