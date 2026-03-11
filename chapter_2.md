# 2. Modern Models
## 2.1 Encoder-Only
### 2.1.1 MLM & NSP Model
In the context of Large Language Models—specifically BERT (Bidirectional Encoder Representations from Transformers)—MLM and NSP are the two primary pre-training objectives used to teach the model language and context.

MLM is a "fill-in-the-blank" task. It allows the model to learn a bidirectional representation of a sentence, meaning it looks at both the left and right context of a word simultaneously. To implement Masked Language Modeling (MLM), we treat the problem as a multi-class classification task where the model predicts the identity of "hidden" tokens from the entire vocabulary.

During pre-training, we take a sequence of text and apply a stochastic masking strategy (most famously the BERT 15% rule): 
1. Selection: 15% of the tokens in a sequence are chosen at random.

2. Transformation: For those chosen tokens:
   - 80% are replaced with a special [MASK] token.
   - 10% are replaced with a random token from the vocabulary (this forces the model to maintain a correct representation of the observed tokens under the given context, which is the most critical feature for resolving polysemy).
   - 10% remain unchanged (this biases the representation towards the actual observed word).

3. Forward Pass: The transformer processes the entire sequence (including masks) and produces a hidden vector $h_i$ for every position $i$.

The objective is to enable the model to learn bidirectional context. Unlike standard GPT models that only look at previous words, MLM forces the model to use both the "left" and "right" context to reconstruct the missing information. The goal is to maximize the likelihood of the original tokens $x$ given the corrupted (masked) version of the sequence $\tilde{x}$. The loss function for MLM is the Cross-Entropy Loss, calculated only over the masked positions. If $M$ is the set of indices of the masked tokens, the loss $\mathcal{L}_{\text{MLM}}$ is defined as:

$$\mathcal{L}_{\text{MLM}} = - \sum_{i \in M} \log P(x_i | \tilde{x})$$

In practice, the model outputs a vector $h_i$ for a masked position, which is then multiplied by the word embedding matrix $W$ and passed through a Softmax:

$$P(x_i | \tilde{x}) = \text{softmax}(h_i W + b)$$

To implement Next Sentence Prediction (NSP), the model is trained to understand the logical relationship between two sentences, moving beyond the word-level context provided by MLM. During the data preparation phase, the model is fed pairs of sentences $(A, B)$ sampled from a large corpus:

1. Selection: 50% of the time: Sentence $B$ is the actual sentence that follows $A$ in the original document (labeled as IsNext). 50% of the time: Sentence $B$ is a random sentence chosen from a different part of the corpus (labeled as NotNext).

2. Formatting: The two sentences are concatenated using special tokens: [CLS] Sentence A [SEP] Sentence B [SEP].

3. Encoding: The model uses Segment Embeddings to distinguish between the first and second sentence, helping the attention mechanism track which tokens belong to which part of the pair.

4. Classification: The final hidden state of the first token, the [CLS] token, is treated as the aggregate representation for the entire sequence pair


NSP is treated as a Binary Classification problem. The model passes the [CLS] embedding through a linear layer and a Softmax to predict the probability of the IsNext label. The loss function is the Binary Cross-Entropy (BCE) Loss:

$$\mathcal{L}_{\text{NSP}} = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

Where:
- $y$ is the binary ground truth (1 if IsNext, 0 if NotNext).
- $\hat{y}$ is the model's predicted probability that $B$ follows $A$.

#### 2.1.1.1 BERT
#### 2.1.1.2 RoBERTa
#### 2.1.1.3 ALBERT
#### 2.1.1.4 SpanBERT
#### 2.1.1.5 DeBERTa V1/2
#### 2.1.1.6 DeBERTa V3
#### 2.1.1.7 XLNet

### 2.1.2 Contrastive Learning Model
Standard BERT wasn't designed to produce high-quality sentence embeddings. While convenient, SBERT researchers found the [CLS] token performs poorly for semantic similarity. Since the [CLS] token was trained specifically for the NSP (Next Sentence Prediction) task, it captures "logical follow-up" information rather than the actual semantic meaning of the sentence. Taking the mean or max of all token embeddings in a sentence is generally better than using [CLS]. However, it suffers from anisotropy. In a pre-trained BERT space, word embeddings tend to occupy a very narrow cone. This means even unrelated sentences can have a high cosine similarity, making it difficult to distinguish between them based on distance.

To solve these issues and make Cosine Similarity a meaningful metric, SBERT uses a Siamese Network architecture (fine-tuning with paired sentences) under three specific objective functions:
1. Classification Objective Function. It is used when you have a dataset with discrete labels (e.g., Entailment, Neutral, Contradiction). It concatenates the two sentence embeddings ($u$ and $v$) along with their element-wise difference ($|u - v|$) before the softmax. The difference vector $|u - v|$ is crucial as it highlights the dimensions where the two sentences disagree.

$o = \text{softmax}(W_t(u, v, |u - v|))$

2. Regression Objective Function. It is used to predict a continuous similarity score (e.g., a scale from 0 to 5). It calculates the cosine similarity between the two embeddings directly. It typically uses Mean Squared Error (MSE) to minimize the distance between the predicted similarity and the gold standard label.

3. Triplet Objective Function. It is used to ensure a specific "Anchor" sentence is closer to a "Positive" (similar) sentence than a "Negative" (dissimilar) one.  It forces the distance between the Anchor ($s_a$) and Positive ($s_p$) to be smaller than the distance between the Anchor and Negative ($s_n$) by at least a margin ($\epsilon$).

$o = $\max(\|s_a - s_p\| - \|s_a - s_n\| + \epsilon, 0)$ 

#### 2.1.2.1 Sentence-BERT

## 2.2 Encoder-Decoder
### 2.2.1 BART
### 2.2.2 T5

## 2.3 Non-Causal Decoder-Only
### 2.3.1 UniLM

## 2.4 Causal Decoder-Only
### 2.4.1 GPT
### 2.4.2 LLaMA
### 2.4.3 DeepSeek
### 2.4.4 Qwen
### 2.4.5 Gemini
