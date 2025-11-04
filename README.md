## French-to-English Translation with an Attention Seq2Seq Model

**Abstract**
This project implements a sequence-to-sequence (Seq2Seq) neural network with a
custom attention mechanism to translate between **French and English in both
directions**. Using the Tatoeba Project’s French–English parallel corpus, we
preprocessed over 200,000 sentence pairs, applied Unicode normalization and
punctuation tokenization, and vectorized tokens into a 5,000-word vocabulary. Our
encoder leverages a bidirectional GRU, and the decoder is a unidirectional GRU
augmented by Multi-Head Attention. Trained for 35 epochs with Adam optimization
and early stopping, the model achieved a validation masked accuracy of ~ 80 %
from French to English and ~80% from English to French and BLEU-style
improvements over random guessing. Attention visualizations demonstrate that the
model effectively focuses on relevant source tokens during generation.

**Keywords:** neural machine translation, sequence-to-sequence, attention,
TensorFlow, natural language processing, bidirectional


## 1. Project Proposal

**Objective**
The goal is to build a deep-learning system capable of translating both **from French
to English and from English to French**. This bidirectional capability enhances the
versatility of the model and makes it applicable in real-world settings like
multilingual chatbots or language learning tools. Machine translation remains a
crucial NLP task, impacting communication, information access, and localization.
By employing Seq2Seq architectures enhanced with attention, we aim to allow the
model to dynamically focus on relevant parts of the input, improving translation
quality on variable-length sentences.

**Dataset**
We use the Tatoeba Project’s French–English parallel corpus under CC BY 2.0 FR.
The TSV file comprises three columns (English, French, attribution); we extracted
the first two columns only. After loading, the dataset contained approximately
210,000 sentence pairs, sorted by length. Preprocessing includes ASCII
normalization (NFKD), lowercasing, removal of extraneous characters, and
punctuation segmentation. We used the **same pairs for both directions** , simply
swapping source and target roles for the second pass of training.

**Expected Challenges**

1. **Vocabulary Size and OOV** : With only a 5,000-word vocabulary, rare words
    map to [UNK], potentially hindering translation of less frequent tokens.
2. **Long-Sentence Modeling** : RNN-based models can struggle to retain context
    over long sequences; attention may mitigate but not eliminate this issue.
3. **Computational Constraints** : Training large RNNs with attention on a
    CPU/GPU‐limited environment was really slow and prone to overfitting
    **_especially that we trained the model locally since we had no access to_**
    **_cloud computing services_**.


## 2. Data Preparation and Preprocessing

**Data Cleaning**
All sentence pairs were non-empty and already tokenized into simple ASCII
punctuation. We verified there were no null entries after splitting the TSV file. No
further outlier removal was necessary.

**Data Augmentation**
While no synthetic examples were generated (e.g., via back-translation or synonym
replacement), **we leveraged the dataset bidirectionally by reversing each
French–English pair into English–French.** This effectively **doubles the training
data** and exposes the model to richer language patterns from both linguistic
perspectives. This augmentation improves generalization and helps the model
disambiguate similar structures in both source and target roles.

**Normalization/Standardization**
We applied Unicode NFKD normalization and converted text to lowercase. A
regular expression retained only letters, spaces, and .?!,¿. Punctuation was
surrounded by spaces to ensure they become individual tokens. Special tokens
[START] and [END] were added to each sequence.

**Data Splitting**
We randomly assigned 80% of the data to training and 20% to validation, batching
at 64 sentences per batch. **Here validation serves to monitor generalization
during training.**


## 3. Model Architecture and Design

**Model Choice**
A custom Seq2Seq architecture with attention was selected. RNNs (GRUs) are
effective for sequential data, and attention helps mitigate vanishing-gradient issues
on long sentences. The same Seq2Seq model with attention was used for both
translation directions. For English→French, the English sentences were treated as
context and French as target; for French→English, the order was reversed by
changing the names of the dataset variables (from context to target and vice versa).

**Layer Design**

- **Encoder:** Embedding layer (vocab size = 5,000, embedding dim = 256)
    followed by a bidirectional GRU (256 units, sum merge).
- **Attention:** A single-head MultiHeadAttention (key_dim=256) wrapped in a
    residual connection and layer normalization.
- **Decoder:** Embedding + unidirectional GRU (256 units), followed by the
    custom CrossAttention layer, and a Dense output layer projecting to
    vocabulary logits.

All recurrent layers used Glorot uniform initialization. Activation for the output layer
is softmax (implicit via logits in SparseCategoricalCrossentropy).

**Model Complexity**
With approximately 10 million parameters, the model balances capacity and
overfitting risk. We limit complexity by capping vocabulary size and using a single
attention head.


## 4. Training Process

**Training Strategy**
We trained from scratch for up to 35 epochs, repeating the training dataset
indefinitely and validating every 100 batches. EarlyStopping (patience=3 on
validation loss) prevented overtraining.

**Loss Function and Optimization**

- **Loss:** Custom masked_loss ignores padding tokens, based on
    SparseCategoricalCrossentropy (from_logits=True).
- **Optimizer:** Adam with default learning rate (0.001).
    Validation metrics included masked_acc (token-level accuracy ignoring
    padding) and masked loss.

**Hyperparameter Tuning**
Key hyperparameters: batch size = 64, units = 256, vocab size = 5,000,
epochs = 35. We performed informal grid search on units (128 vs. 256) and batch
size (32 vs. 64) on a single validation fold, selecting the above for optimal validation
accuracy and training time.

**Regularization**
Early stopping acted as implicit regularization to prevent overfitting and training the
model in an effective way.


## 5. Evaluation and Analysis

**Evaluation Metrics:**
We used **token-level masked accuracy** and **cross-entropy loss**. For
human-readable evaluation, we inspected **example translations** and computed
**informal BLEU-style scores** by comparing to reference sentences.
BLEU was chosen because it provides a meaningful measure of translation quality
by evaluating how well the generated sentences match reference translations in
terms of both word choice and word order. It is a widely accepted metric in
machine translation tasks and complements token-level accuracy by capturing the
overall fluency and coherence of the output.

**Validation Results**
French to English :

- **Validation loss:** 1.1 485
- **Validation masked accuracy:** ≈ 80 %

English to French:

- **Validation loss:** 1.
- **Validation masked accuracy:** ≈ 77 %

Both directions yielded similar results, indicating strong generalization. Some

tokens translated better in one direction than the other, e.g., pronouns and

idiomatic expressions.

**Error Analysis**
Typical errors include:

- Presence of **[UNK]** tokens, especially for rare words _(because of the limited_
    _vocabulary)_
- Word order inversions in longer sentences.
- Missing or repeated function words _(e.g., articles, prepositions)._
- Verb conjugation mismatches


**Model Comparison**
Only the described architecture was tested. A baseline “random” model would
expect loss ≈ 8.52 and accuracy ≈ 0.02%; our model vastly outperforms this.


## 6. Interpretability and Insights

**Attention Visualization**
Plotting attention weights for sample sentences shows strong diagonal patterns,
indicating the model aligns source and target tokens effectively for short inputs. For
example, translating “I like red apples.” correctly attends “pommes”→“apples.”

**Insights**

- Even a single attention head captures meaningful alignments.
- The model’s performance plateaus around epoch 27 for both models due to
    small vocab (5000 words) and limited hardware to test with more epochs,
    suggesting no additional benefit from further epochs if the current vocab
    size/dataset.

## 7. Conclusion and Future Directions

**Summary of Contributions**
We built and trained a Seq2Seq translation model with attention, successfully
training a **single model architecture for bidirectional translation**. This validates
the flexibility of the architecture and provides a foundation for further multilingual
models. The project demonstrates practical application of RNNs and attention for
NLP tasks.

**Limitations**

- Limited vocabulary leads to **unknown tokens**.
- RNNs/LSTMs struggle with **very long** sentences.

**Future Work**

- Integrate subword tokenization (Byte-Pair Encoding) to reduce [UNK].
- Integrate a **universal tokenizer** to handle both directions in a single training
    cycle.
- Experiment with **Transformer architectures** for improved long-range
    modeling.
- Apply dropout or layer normalization within RNNs for stronger regularization.
- Explore **multilingual pretraining** to improve zero-shot translation.


## Environment specifications:

- TensorFlow 2.18.0, TensorFlow-Text 2.18.1, Python 3.
- Random seed fixed via np.random.seed(0) and tf.random.set_seed(0) for
    reproducibility.

All code cells include explanations, and instructions at the top of the notebook
guide replication of results.



