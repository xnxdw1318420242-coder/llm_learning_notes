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
   - During the pre-training phase, the model is conditioned to see the [MASK] token. However, during fine-tuning or inference (real-world use), the input text will never contain a [MASK] token. This creates a significant mismatch between the distribution of data the model was trained on and the data it actually sees in production. If the model only learns to extract features for "missing" words, its performance on fully visible sentences will degrade. 

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
BERT is fundamentally constructed as a deep stack of Transformer Encoders. It conceptually borrows the continuous bag-of-words (CBOW) methodology from Word2Vec but applies it through a masked language modeling approach. The architecture is typically deployed in two standard configurations:Base: 12 Transformer layers ($L$), a hidden dimension of 768 ($H$), and 12 self-attention heads ($A$).Large: 24 Transformer layers ($L$), a hidden dimension of 1024 ($H$), and 16 self-attention heads ($A$).The total parameter count for the model is derived from the vocabulary size ($V$), the model dimension ($d_{model}$), and the feed-forward network dimensions ($d_{ff}$):

$$\text{Total Parameters} = V \cdot d_{model} + L(4 \cdot d_{model}^2 + 2 \cdot d_{model} \cdot d_{ff})$$

<p align="center">
<img width="903" height="204" alt="479ede86-8630-449c-8993-9b501cd7a44f" src="https://github.com/user-attachments/assets/d847684f-3bc1-41ba-b9a5-9e8526762a6d" />
</p>

The primary differentiator among the models above is how they route information and extract semantic features:

- ELMo (Shallow Bidirectionality): Built on Bi-LSTM networks, ELMo trains two separate models—one reading left-to-right and the other right-to-left. The final representation is merely a concatenation of these two independent vectors. It lacks true simultaneous integration of the full context.

- GPT (Unidirectionality): Utilizes the Transformer Decoder block. It is strictly autoregressive, predicting the next word based exclusively on preceding tokens, optimizing the objective $P(w_i|w_1,...,w_{i-1})$. While exceptional for generative tasks, this forward-only constraint limits deep contextual understanding.

- BERT (Deep Bidirectionality): Employs the Transformer Encoder to evaluate all tokens simultaneously, optimizing for $P(w_i|w_1,...,w_{i-1}, w_{i+1},...,w_n)$. By sacrificing the ability to generate text sequentially, it deeply fuses global semantics to achieve significantly richer feature extraction.

Transitioning to BERT resolves several historical bottlenecks in language processing:

- Advantage over RNNs/LSTMs: The Transformer foundation natively supports parallel sequence processing, eliminating the sequential bottleneck of recurrent networks. Its multi-head attention mechanism is also far more effective at capturing long-range dependencies within a text.

- Advantage over Word2Vec: Where Word2Vec assigns a single, static vector to each word regardless of its usage, BERT generates dynamic, context-aware embeddings. This effectively resolves the issue of polysemy, allowing the same word to have different representations depending on its surrounding text.

- Advantage over ELMo and GPT: BERT supersedes ELMo's superficial concatenation by deeply fusing semantic information across all layers. Compared to GPT's restricted left-to-right visibility, BERT's unrestricted access to surrounding context optimizes it heavily for comprehensive understanding tasks rather than pure generation.

To prepare a sequence for the Transformer Encoder, BERT uses a composite input representation. Every input token is the sum of three distinct embedding layers, which allow the model to understand the identity, position, and contextual group of each word. For any given input, BERT constructs a vector by element-wise addition of the following:

1. Token Embeddings
This is the core semantic layer. BERT uses WordPiece tokenization, which breaks down words into sub-units (e.g., "embeddings" might become "em", "##bed", "##dings").
Special Tokens:

- [CLS]: Always the first token of every sequence. Its final hidden state is used as the aggregate representation for classification tasks.

- [SEP]: A delimiter token used to separate two different sentences in a single input.

2. Segment Embeddings
Since BERT is often trained on sentence pairs (like Question-Answering or Next Sentence Prediction), it needs to know which tokens belong to "Sentence A" and which belong to "Sentence B." All tokens in the first sentence are assigned a Sentence A embedding. All tokens in the second sentence (after the [SEP] token) are assigned a Sentence B embedding. This helps the model model relationships between sentences.

Tokens that belong to the same sentence share the same Segment Embedding. This shared representation allows the attention mechanism to distinguish between information belonging to different segments, facilitating the learning of cross-sentence relationships. The way these embeddings are used depends entirely on the nature of the NLP task. When the model only processes one sentence at a time (e.g., sentiment analysis), there is only one segment to consider. Consequently, the Segment ID is always 0 for all tokens in the input. When the input consists of two related sentences (a premise and a hypothesis), the model needs to track the boundary between them. In this case, tokens in the first sentence are assigned a Segment ID of 0, and tokens in the second sentence are assigned a Segment ID of 1.

3. Position Embeddings
Unlike Recurrent Neural Networks (RNNs) that process words one by one, Transformers process all tokens in a sequence simultaneously. To retain the sense of word order, BERT learns a unique vector for each absolute position (0, 1, 2, ... up to 512). Without these, the model would treat the sentence "The cat chased the dog" exactly the same as "The dog chased the cat."

The final input representation for any token $t$ is the element-wise sum of three distinct embeddings: 

$$Embedding(t) = TokenEmb[t] + PositionEmb[pos] + SegmentEmb[seg]$$

BERT FFN utilizes GeLU as the activation function.

<p align="center">
<img width="901" height="265" alt="6506d0da-b537-4872-9016-144db09396dd" src="https://github.com/user-attachments/assets/7f0ce64a-4778-4d77-a8a6-7d7ec09f8112" />

</p>
BERT was trained on a massive, unlabeled dataset to capture a broad range of linguistic patterns: BooksCorpus and English Wikipedia, providing the model with exposure to diverse writing styles and complex sentence structures. The pre-training was a computationally expensive process involving specific architectural and optimization choices.

- Optimizer: Adam with specific parameters ($\beta_1 = 0.9, \beta_2 = 0.999$).

- Learning Rate (LR): Set at 1e-4 with a linear warmup for the first 10,000 steps, followed by a linear decay.

- Weight Decay: L2 weight decay of 0.01 to prevent overfitting.

- Dropout: A consistent rate of 0.1 applied across all layers.

- Steps & Batch Size: The model was trained for 1,000,000 steps with a batch size of 256 sequences.

- Two-Phase Training: Phase 1 (90% of steps) trained with a shorter sequence length of 128 to speed up the initial learning, and phase 2 (10% of steps) trained with a sequence length of 512 to help the model learn long-range dependencies and positional embeddings.

BERT is pre-trained using a self-supervised approach. This means it learns directly from massive amounts of raw text without requiring manual labeling, making it highly scalable. At its core, the model learns high-quality vector representations (embeddings) for words. These embeddings are context-aware, allowing the same word to have different mathematical representations based on its surrounding sentence. 

BERT Training utilized Google TPUs (8 to 64 nodes depending on the model size). The standard training cycle for $\text{BERT}_{\text{BASE}}$ took approximately 4 days.

BERT performs fine-tuning by taking the pre-trained "foundation" model and adding a single, task-specific output layer on top. BERT adapts to four primary types of downstream tasks using the following methods.

- Sentence-Level Classification (Pair & Single). 
For tasks involving entire sentences—such as determining if two sentences are similar or classifying the sentiment of a single sentence—BERT leverages the [CLS] token. The final hidden state (vector) of the [CLS] token, which is always the first token in the sequence, is treated as the aggregate representation of the entire input. A Fully Connected (FC) layer and a Softmax layer are added on top of the [CLS] output. For binary classification, it uses the existing structure from the Next Sentence Prediction (NSP) pre-training task. For multi-class tasks, the FC layer ensures the output dimension matches the number of categories, followed by an argmax operation to find the final result.
<p align="center">
<img width="202" height="182" alt="image" src="https://github.com/user-attachments/assets/057bbedc-2564-43c1-b38e-a51d61e6f88b" />
<img width="202" height="182" alt="a57b334c-91ca-40e6-b5af-e8edf9af5978" src="https://github.com/user-attachments/assets/eeeaf464-ceb4-4a2f-afce-cb7fbcb6a56b" />
</p>

- Question Answering (QA). This is a more complex task where the model must identify the specific start and end "span" of an answer within a paragraph (Sentence B) based on a question (Sentence A). BERT introduces two new auxiliary vectors: $s$ (start) and $e$ (end). The final feature vector $T'_i$ for each word in the paragraph is passed through a Fully Connected layer to transform abstract semantics into task-oriented features. The model computes the dot product between these transformed features and the auxiliary vectors $s$ and $e$. A Softmax is applied over all tokens in the paragraph. The token with the highest probability for $s$ is the start of the answer, and the highest for $e$ is the end.
<p align="center">
<img width="207" height="182" alt="a8742204-0747-4ccc-9e4b-5a63ea4781d5" src="https://github.com/user-attachments/assets/6c4a6f0e-bed4-41d9-9fd9-245a6e0043aa" />

</p>

- Single Sentence Labeling (NER). In tasks like Named Entity Recognition (NER), where every word needs a label (e.g., Person, Location, Organization), BERT operates at the token level. Unlike classification tasks that only look at the [CLS] token, labeling tasks use the final hidden state of every individual token. A Fully Connected layer followed by a Softmax is added to the end of each token's feature vector. The output uses the IOBES method. 

<p align="center">
<img width="237" height="202" alt="d9913a1b-7c10-4dd9-8b1a-2ef46e16f615" src="https://github.com/user-attachments/assets/f78aad7e-e53c-4861-ac62-3ac2ed755202" />
</p>

#### 2.1.1.2 RoBERTa

RoBERTa (A Robustly Optimized BERT Pretraining Approach) introduces several key improvements over the original BERT model to enhance training efficiency and performance. RoBERTa scales up the training process through larger batches and modified optimization. While BERT used a batch size of 256 for 1 million steps, RoBERTa experimented with much larger batches, such as 2k and 8k. The final model utilized a batch size of 8k for 500k steps. The Adam optimizer's $\beta_2$ parameter was adjusted from 0.999 (BERT) to 0.98 for RoBERTa.

RoBERTa also utilizes a significantly larger and more diverse corpus than its predecessor. It was trained on 160GB of text, a tenfold increase over BERT's 16GB. In addition to the original BookCorpus and Wikipedia (16GB), RoBERTa incorporated CC-NEWS, OPENWEBTEXT, and STORIES. 

RoBERTa researchers questioned the necessity of the Next Sentence Prediction (NSP) task and tested four different input formats.

- SEGMENT-PAIR + NSP: The original BERT format using two segments (each with multiple sentences) and NSP loss. The input contains two parts, each consisting of a "segment" from either the same document or different documents. A "segment" can contain multiple natural sentences, but the total combined token count for both segments must be fewer than 512. Pre-training includes both the MLM task and the NSP task.

- SENTENCE-PAIR + NSP: Similar to the above, but specifically restricted to single sentences. Each input contains a pair of single sentences sampled from the same or different documents. Because these inputs are significantly shorter than 512 tokens, the batch size is increased so that the total number of tokens per batch remains similar to the SEGMENT-PAIR configuration. Experimental results showed that using single sentences harms performance on downstream tasks, likely because the model cannot learn long-range dependencies.

- FULL-SENTENCES: The input consists of only one part. It contains a continuous stream of full sentences sampled from one or more documents. If the model reaches the end of one document, it continues sampling from the next document until the total token count reaches 512. Pre-training does not include the NSP task.

- DOC-SENTENCES: Similar to FULL-SENTENCES, but with a strict document boundary. Sentences are sampled continuously from a single document only. The input does not cross document boundaries. If the end of a document is reached before hitting 512 tokens, the input may be shorter. To ensure efficiency and maintain a high number of tokens per batch, the batch size is dynamically adjusted for these shorter inputs. This format was found to perform slightly better than FULL-SENTENCES.

The researchers found that when the NSP loss is removed, using DOC-SENTENCES or the original SEGMENT-PAIR input format actually outperforms the original BERT-base results. Ultimately, RoBERTa adopted the FULL-SENTENCES format for subsequent experiments primarily for its implementation convenience.

RoBERTa replaces the "static" masking used in BERT with a more fluid approach. In BERT (Static), masked tokens were fixed during data preprocessing, meaning the model saw the same masks in every epoch. In RoBERTa (Dynamic), the mask is generated dynamically every time a sequence is fed into the model. Dynamic masking was found to be more efficient and slightly more effective than static masking.

RoBERTa uses byte-level BPE with a 50K vocabulary. This allows the model to handle a wider variety of common vocabulary without additional heuristic preprocessing or tokenization. This change added approximately 15 million parameters to the Base model and 20 million to the Large model.

#### 2.1.1.3 ALBERT

ALBERT (A Lite BERT) focus on reducing parameter count and enhancing the model's ability to learn inter-sentence coherence.

In the original BERT, the Token Embedding dimension ($E$) is tied to the Transformer hidden layer dimension ($H$) (e.g., both are 768). ALBERT decouples these two based on different learning objectives. The embedding layer primarily learns context-independent word/surface information, while the Transformer layers focus on context-dependent semantic and syntactic meaning. These two functions do not necessarily require the same dimensionality. ALBERT first projects tokens into a lower-dimensional space $E$, then uses a projection matrix ($E \times H$) to map them into the higher-dimensional hidden space $H$. Total parameters for this section are reduced from $V \times H$ to $(V \times E) + (E \times H)$. When $E \ll H$, this significantly decreases the parameter count without a major loss in performance.

To further reduce model size, ALBERT employs a "weight-sharing" strategy across its internal layers. All Transformer layers (including Attention and Feed-Forward Network parameters) share a single set of weights. This means the model essentially trains one layer and has the input pass through it multiple times. This drastically reduces the number of parameters. For a 12-layer model, you only need to store 1/12th of the original parameters. While there is a slight dip in performance compared to non-sharing models, it reduces overfitting and makes the model much more memory-efficient.

ALBERT replaces BERT's original Next Sentence Prediction (NSP) task with a more challenging objective called SOP. NSP was criticized for being too easy because it often compared a related sentence to a completely random one from a different document. In ALBERT, positive samples (50%) include two consecutive sentences (A, B) from the same document in their original order. Negative samples (50%) include the same two sentences but with their order swapped (B, A). The model must now learn fine-grained coherence and discourse relationships to detect the swap, rather than just identifying topical differences. SOP is a much more rigorous task that leads to better performance on downstream multi-sentence tasks.

#### 2.1.1.4 SpanBERT

SpanBERT focuses on improving how the model represents contiguous spans of text rather than just individual tokens.

<p align="center">
<img width="760" height="306" alt="44bb5556-1337-490a-a020-f1ebc5c4fb22" src="https://github.com/user-attachments/assets/282c3a70-399c-4c27-aecc-72d81c2c2faa" />

</p>

SpanBERT shifts from individual Token-level Masking to Span-level Masking. The model first determines a "span length" and then randomly selects a starting position in the sentence to mask a continuous sequence of tokens. This forces the model to rely more heavily on the surrounding global context to predict the entire missing segment. It reduces the randomness associated with masking isolated local positions and better aligns with tasks requiring a deep understanding of entire phrases or coreference resolution.

In addition to the standard Masked Language Model (MLM) task, SpanBERT introduces the Span Boundary Objective. The model uses the tokens at the boundaries (the token immediately before the span, $x_{s-1}$, and the token immediately after, $x_{e+1}$) to predict every masked token within that span. $SBO(p_i) = f(x_{s-1}, x_{e+1}, p_i)$, where $p_i$ represents the $i$-th masked position within the span. By using boundary information to reconstruct the missing content, the model strengthens its ability to model "chunks" or "blocks" of text.

Following the findings of models like RoBERTa and ALBERT, SpanBERT also removes the NSP task. SpanBERT is particularly effective for extractive Question Answering (QA) and coreference resolution, tasks where representing a specific "span" of text is critical for accuracy.

#### 2.1.1.5 DeBERTa V1/2

DeBERTa (Decoding-enhanced BERT with Disentangled Attention) is a model introduced by Microsoft in 2021 that achieved superhuman performance on the SuperGLUE benchmark. It is widely used today as a foundation for challenging NLP and even some NLG tasks. DeBERTa's core purpose is to improve BERT’s self-attention mechanism and masking strategies to enhance the model's linguistic understanding and generalization capabilities.

In traditional BERT, each token is represented by a single vector that sums its content and absolute position. DeBERTa argues that word dependencies depend heavily on relative rather than absolute distance (e.g., words are more likely to be related if they are adjacent). To model this, DeBERTa represents each token $i$ using two separate vectors:

- $\{H_i\}$: Representing the word's Content.
- $\{P_{i|j}\}$: Representing the Relative Position of token $i$ with respect to token $j$.

The cross-product of the Query $Q_i$ and Key $K_j$ is expanded as follows:

$$A_{i,j} = \{H_i, P_{i|j}\} \times \{H_j, P_{j|i}\}^T$$

The total attention weight between token $i$ and token $j$ is calculated by summing four separate types of interactions:

- Content-to-Content ($H_i H_j^T$): How much the meaning of word $i$ relates to the meaning of word $j$.
- Content-to-Position ($H_i P_{j|i}^T$): How much the meaning of word $i$ relates to the relative position of word $j$.
- Position-to-Content ($P_{i|j} H_j^T$): How much the relative position of word $i$ relates to the meaning of word $j$.
- Position-to-Position ($P_{i|j} P_{j|i}^T$): The document notes that because relative position embeddings are already used, this term provides little additional information and is removed in the final implementation to save computation.

The simplified attention score $A_{i,j}$ used in the model is:

$$A_{i,j} = \underbrace{Q_i^c K_j^{c\intercal}}_{\text{Content-to-Content}} + \underbrace{Q_i^c K_{\delta(i,j)}^{r\intercal}}_{\text{Content-to-Position}} + \underbrace{K_j^c Q_{\delta(j,i)}^{r\intercal}}_{\text{Position-to-Content}}$$


The model defines relative distance within a maximum range $k$. The relative distance $\delta(i, j)$ is defined by the following mapping:

$$\delta(i, j) = 
\begin{cases} 
0 & \text{if } i - j \leq -k \\
2k - 1 & \text{if } i - j \geq k \\
i - j + k & \text{otherwise}
\end{cases}$$


This mapping ensures the resulting indices fall within the range $[0, 2k-1]$, allowing the model to distinguish between "left" (past) and "right" (future) contexts relative to the current token.

DeBERTa uses separate projection matrices ($W_{q,c}, W_{k,c}, W_{v,c}$) for content and ($W_{q,r}, W_{k,r}$) for relative positions to generate query and key vectors. Because the attention score is the sum of three distinct components (after removing position-to-position), the final attention matrix is scaled by $\sqrt{3d}$ instead of the standard $\sqrt{d}$ to maintain stability. This approach allows the model to understand that the relationship between "deep" and "learning" is strong because they are adjacent, regardless of where they appear in a sentence.

DeBERTa has better information separation because processing content and position independently reduces mutual interference between these two distinct types of information. DeBERTa achieves a notable increase in performance across various Natural Language Understanding (NLU) tasks compared to the original BERT model.

<p align="center">


</p>

The Enhanced Mask Decoder (EMD) is a specialized mechanism in DeBERTa designed to address a critical limitation of the standard BERT model: the reliance on relative positions alone during the pre-training phase. While relative positions help capture local dependencies, certain tasks (like predicting a masked word) require absolute position information to fully understand the sentence structure. For example, in the phrase "a new store opened beside the new mall," both "store" and "mall" follow the word "new," making them indistinguishable if the model only looks at local relative context.

Instead of merging absolute positions at the very first input layer (as BERT does), DeBERTa incorporates them right before the final prediction head. The EMD typically consists of $n$ layers (where $n=2$) that share weights to remain parameter-efficient. It takes two primary inputs:

- $H$: The hidden states (contextual embeddings) from the final Transformer encoder layer.
- $I$: The specific information needed for decoding. For the first EMD layer, $I$ is the absolute position embedding; for subsequent layers, $I$ is the output from the previous EMD layer.



#### 2.1.1.6 DeBERTa V3
#### 2.1.1.7 XLNet

### 2.1.2 Contrastive Learning Model
Standard BERT wasn't designed to produce high-quality sentence embeddings. While convenient, SBERT researchers found the [CLS] token performs poorly for semantic similarity. Since the [CLS] token was trained specifically for the NSP (Next Sentence Prediction) task, it captures "logical follow-up" information rather than the actual semantic meaning of the sentence. Taking the mean or max of all token embeddings in a sentence is generally better than using [CLS]. However, it suffers from anisotropy. In a pre-trained BERT space, word embeddings tend to occupy a very narrow cone. This means even unrelated sentences can have a high cosine similarity, making it difficult to distinguish between them based on distance.

To solve these issues and make Cosine Similarity a meaningful metric, SBERT uses a Siamese Network architecture (fine-tuning with paired sentences) under three specific objective functions:
1. Classification Objective Function. It is used when you have a dataset with discrete labels (e.g., Entailment, Neutral, Contradiction). It concatenates the two sentence embeddings ($u$ and $v$) along with their element-wise difference ($|u - v|$) before the softmax. The difference vector $|u - v|$ is crucial as it highlights the dimensions where the two sentences disagree.

$$o = \text{softmax}(W_t(u, v, |u - v|))$$

2. Regression Objective Function. It is used to predict a continuous similarity score (e.g., a scale from 0 to 5). It calculates the cosine similarity between the two embeddings directly. It typically uses Mean Squared Error (MSE) to minimize the distance between the predicted similarity and the gold standard label.

3. Triplet Objective Function. It is used to ensure a specific "Anchor" sentence is closer to a "Positive" (similar) sentence than a "Negative" (dissimilar) one.  It forces the distance between the Anchor ($s_a$) and Positive ($s_p$) to be smaller than the distance between the Anchor and Negative ($s_n$) by at least a margin ($\epsilon$).

$$o = \max(\|s_a - s_p\| - \|s_a - s_n\| + \epsilon, 0)$$

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
