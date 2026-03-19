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

#### 2.1.1.5 DeBERTa 

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

The Enhanced Mask Decoder (EMD) is a specialized mechanism in DeBERTa designed to address a critical limitation of the standard BERT model: the reliance on relative positions alone during the pre-training phase. While relative positions help capture local dependencies, certain tasks (like predicting a masked word) require absolute position information to fully understand the sentence structure. For example, in the phrase "a new store opened beside the new mall," both "store" and "mall" follow the word "new," making them indistinguishable if the model only looks at local relative context.

Instead of merging absolute positions at the very first input layer (as BERT does), DeBERTa incorporates them right before the final prediction head. The EMD typically consists of $n$ layers (where $n=2$) that share weights to remain parameter-efficient. It takes two primary inputs:

- $H$: The hidden states (contextual embeddings) from the final Transformer encoder layer.
- $I$: The specific information needed for decoding. For the first EMD layer, $I$ is the absolute position embedding; for subsequent layers, $I$ is the output from the previous EMD layer.

By introducing absolute positions at the end of the stack, the model can use the rich relative-position-aware context it has already learned to "anchor" itself to specific points in the sequence. This is visually represented in the architecture where absolute position embeddings are added only at the decoder stage.

The EMD functions as a flexible decoder that can utilize various types of input information.
In general case, each EMD layer output becomes the input $I$ for the next:

$$I_{next} = \text{EMD\_{Layer}}(I, H)$$

If we set $n=1$ and the input $I=H$, the EMD mathematically simplifies to a standard BERT-style decoder. However, by setting $n=2$ and using absolute positions as the initial $I$, DeBERTa gains superior flexibility in modeling complex syntax.

SiFT (Scale-invariant Fine-Tuning) is a robust training technique introduced alongside DeBERTa to improve the model's stability and generalization during the fine-tuning phase. Standard fine-tuning can be unstable because small perturbations in the input can lead to large changes in the model's output. While Virtual Adversarial Training (VAT) helps by adding noise to inputs to make the model more robust, it often suffers from "unstable gradients," especially when dealing with different model scales (like BERT-base vs. BERT-large). SiFT improves upon VAT by applying perturbations to normalized word embeddings rather than the raw embeddings.

Before adding any noise, SiFT normalizes the word embeddings. If $x$ is the word embedding, the normalized version $\hat{x}$ is calculated as:

$$\hat{x} = \text{LayerNorm}(x)$$

This ensures that the embeddings are on a consistent scale, making the subsequent noise application more predictable across different models and tasks. Once normalized, SiFT adds a small amount of adversarial noise ($\delta$) to the embeddings:

$$\tilde{x} = \hat{x} + \delta$$

The goal is to train the model such that the output for the "noisy" input $\tilde{x}$ remains as close as possible to the output for the original input $x$.

The training process uses a specific loss function to enforce this stability:

- Standard Loss: The model is trained to minimize the standard cross-entropy loss for the specific task.
- Robustness Loss (Kullback-Leibler Divergence): The model also minimizes the KL divergence between the probability distribution of the original input and the perturbed input. This forces the model to be "smooth"—meaning small changes in the embedding space won't cause the model to jump to a different classification.
  
$$\min_{\theta} LD(P(y|x; \theta), P(y|x+\delta; \theta))$$

DeBERTa utilizes two main objectives during its pre-training phase:
- Enhanced Masked Language Model (MLM): Like BERT, DeBERTa is trained to predict masked tokens in a sequence. However, it uses an "Enhanced" version that incorporates the Disentangled Attention mechanism—modeling content and relative positions separately—and the Enhanced Mask Decoder (EMD), which introduces absolute position information at the decoding stage to better distinguish between identical words in different positions.
- Sentence Order Prediction (SOP): DeBERTa replaces BERT’s Next Sentence Prediction (NSP) task with SOP. This task requires the model to determine if two sentences are in their original order or have been swapped. SOP is considered more challenging than NSP, encouraging the model to learn deeper semantic relationships and cross-sentence coherence.

DeBERTa-v2 shifts its tokenization strategy to handle text more efficiently. It moves away from the previous tokenizer in favor of SentencePiece. The vocabulary size is increased from 50k to 128k. This larger vocabulary allows the model to better represent diverse datasets and reduces the fragmentation of rare words into too many sub-tokens. 

Besides, in the original DeBERTa, the disentangled attention used three terms: Content-to-Content, Content-to-Position, and Position-to-Content. DeBERTa-v2 integrates an additional interaction term: Position-to-Content (often referred to in technical shorthand as n2c or neighbor-to-content). While the original model already had position-aware components, v2 refines how positional information "queries" content, allowing for a more nuanced understanding of syntactic dependencies based on relative distance.

DeBERTa-v2 also introduces significantly larger model variants to push the boundaries of NLU performance: The architecture for the largest version includes 48 layers with a hidden dimension ($H$) of 1536. Unlike BERT or DeBERTa-v1, which often use a simple linear projection for the input, DeBERTa-v2 adds a convolutional layer to the input embedding stage. This "Conv-Stem" acts as a local feature extractor before the global self-attention layers take over. This allows the model to capture immediate n-gram relationships more effectively at the very start of the network, which has been shown to improve the stability of deep transformer stacks.

DeBERTa-v3 is an evolution of the model that changes the core training philosophy while keeping the architectural breakthroughs (like Disentangled Attention) from previous versions. The biggest shift in v3 is moving from a "Guess the Missing Word" task to a "Spot the Fake Word" task. Instead of using the standard Masked Language Model (MLM) where the model predicts a hidden token, DeBERTa-v3 uses RTD, similar to the ELECTRA model. It works like a GAN (Generative Adversarial Network) with two parts:

- The Generator: A small model that performs MLM. It takes a sentence with masked words and fills them in with plausible (but potentially "fake") words.

$$L_{MLM} = \mathbb{E} \left( - \sum_{i \in C} \log p_{\theta_G} (\tilde{x}_{i,G} = x_i | \tilde{X}_G) \right)$$

- The Discriminator: This is the main DeBERTa model. Instead of predicting what the original word was, it must decide for every word in the sentence: "Is this the original word, or was it replaced by the generator?".

$$L_{RTD} = \mathbb{E} \left( - \sum_{i} \log p_{\theta_D} (\mathbb{1}(\tilde{x}_{i,D} = x_i) | \tilde{X}_{D,i}) \right)$$

The model is trained using a multi-task loss function that combines the Generator's attempt to learn language and the Discriminator's attempt to catch fakes:

$$L = L_{MLM} + \lambda L_{RTD}$$

In standard MLM, the model only learns from the ~15% of tokens that are masked. In RTD, the model must make a binary choice for 100% of the tokens in the input, which is a much more efficient use of data. The generator is trained to produce "ambiguous" tokens—fake words that actually fit the context—forcing the discriminator to learn very fine-grained semantic differences to tell them apart.

A major technical challenge in v3 was how the Generator and Discriminator should share their "vocabulary" (embeddings). The authors explored three ways:

- Embedding Sharing (ES): They share the same table, but the two tasks pull the meanings in opposite directions (MLM wants similar words together; RTD wants to separate them).
- No Sharing (NES): They have completely separate tables. This is faster but hurts overall performance.
- GDES (The Winner): This is a clever compromise used in v3. The Discriminator "borrows" the embeddings from the Generator but is not allowed to change them directly. Instead, it uses a separate "delta" ($E_\Delta$) to adapt those embeddings for its own needs.

$$E_D = sg(E_G) + E_{\Delta}$$

The Discriminator "borrows" the Generator's knowledge but isn't allowed to change the Generator's weights. $E_{\Delta}$ is a specific "delta" or adjustment layer that the Discriminator learns on its own to fine-tune the borrowed embeddings for the "spot the fake" task.

#### 2.1.1.6 XLNet

The primary goal of XLNet: to combine the strengths of two main language modeling approaches.

- AR LM (Auto-Regressive, like GPT): These models predict tokens from left to right. While they are naturally suited for generative tasks, they suffer from a "context gap" because they can only see information from one direction.

- AE LM (Auto-Encoding, like BERT): These models use bi-directional context by randomly masking tokens and predicting them. However, the use of the [MASK] token creates a mismatch between pre-training (where masks exist) and inference (where they do not).

- The XLNet Solution: It aims to keep the consistency of AR modeling (no masks) while capturing the bi-directional context of AE models through a technique called Permutation Language Modeling (PLM).

Instead of predicting tokens in their natural linear order, XLNet randomly generates a permutation $\pi$ of the sequence indices. The model performs predictions based on the order of the permutation.  At the attention layer level, a mask ensures that the token at position $\pi(j)$ can only access the "prior" tokens $\pi(k)$ where $k < j$ in the permutation. Since tokens appear in different positions across various random permutations, every token eventually gets to "see" every other token in the sequence, achieving a pseudo-bi-directional understanding without using masks.

In standard AR modeling, a prediction at position $z$ depends on both the context and the content of the token at $z$. In XLNet's permutation approach, this creates a paradox: to predict the word at position $z$, the model needs to know the position but must not know the content of that word. To solve this, XLNet uses two separate hidden representations (streams):

- Content Stream ($h_z$): Similar to standard Transformer states, this encodes both the context and the content of token $z$. It is used as context for other tokens.
  
- Query Stream ($g_z$): This encodes the context and the target position $z$, but specifically excludes the content of token $z$. This is used to predict the actual word at that position.


To handle the massive computational requirements of permutations and long-range dependencies, XLNet integrates two more techniques.

1. Predicting every token in every permutation is extremely expensive. XLNet only predicts the last $1/K$ tokens in a given permutation sequence. This focuses the model's learning on tokens that already have a significant amount of context available, which improves training efficiency.
2. XLNet inherits the "Extra-Long" architecture from Transformer-XL to handle long sequences. Instead of treating segments independently (which causes "context fragmentation"), the model reuses hidden states from previous segments as a memory cache. Since absolute positions (1, 2, 3...) lose meaning when segments are reused, XLNet uses relative distances between tokens to maintain spatial coherence across long texts.

### 2.1.2 Contrastive Learning Model
Standard BERT wasn't designed to produce high-quality sentence embeddings. While convenient, SBERT researchers found the [CLS] token performs poorly for semantic similarity. Since the [CLS] token was trained specifically for the NSP (Next Sentence Prediction) task, it captures "logical follow-up" information rather than the actual semantic meaning of the sentence. Taking the mean or max of all token embeddings in a sentence is generally better than using [CLS]. However, it suffers from anisotropy. In a pre-trained BERT space, word embeddings tend to occupy a very narrow cone. This means even unrelated sentences can have a high cosine similarity, making it difficult to distinguish between them based on distance.

To solve these issues and make Cosine Similarity a meaningful metric, SBERT uses a Siamese Network architecture (fine-tuning with paired sentences) under three specific objective functions:
1. Classification Objective Function. It is used when you have a dataset with discrete labels (e.g., Entailment, Neutral, Contradiction). It concatenates the two sentence embeddings ($u$ and $v$) along with their element-wise difference ($|u - v|$) before the softmax. The difference vector $|u - v|$ is crucial as it highlights the dimensions where the two sentences disagree.

$$o = \text{softmax}(W_t(u, v, |u - v|))$$

2. Regression Objective Function. It is used to predict a continuous similarity score (e.g., a scale from 0 to 5). It calculates the cosine similarity between the two embeddings directly. It typically uses Mean Squared Error (MSE) to minimize the distance between the predicted similarity and the gold standard label.

3. Triplet Objective Function. It is used to ensure a specific "Anchor" sentence is closer to a "Positive" (similar) sentence than a "Negative" (dissimilar) one.  It forces the distance between the Anchor ($s_a$) and Positive ($s_p$) to be smaller than the distance between the Anchor and Negative ($s_n$) by at least a margin ($\epsilon$).

$$o = \max(\|s_a - s_p\| - \|s_a - s_n\| + \epsilon, 0)$$

#### 2.1.2.1 Sentence-BERT

while BERT is a powerful language model, it has significant limitations when it comes to representing entire sentences for similarity or retrieval tasks. Sentence-BERT (SBERT) was developed to overcome these efficiency and performance hurdles.

Using a standard BERT model to find similar sentences is problematic for several reasons. BERT is primarily trained on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). It is not specifically optimized for sentence-level similarity, meaning its standard output (like the [CLS] token) often yields poor semantic distance measurements.  To compare two sentences, BERT requires them to be concatenated into a single input for a forward pass. If you want to find the most similar pairs among $N$ sentences, you must perform $N(N-1)/2$ forward passes. Because BERT needs to see both sentences at once to calculate their relationship, you cannot pre-calculate and "cache" individual sentence embeddings for quick lookup.

SBERT modifies the BERT architecture using a Siamese or Triplet network structure. This allows the model to process sentences independently but in a way that maps them into a shared vector space where similar sentences are physically close to each other. SBERT encodes each sentence into a fixed-sized vector in a single forward pass. Once encoded, similarity can be calculated instantly using simple math like Cosine Similarity or Dot Product. During training, SBERT is supervised with tasks that explicitly group similar sentence pairs together and push dissimilar ones apart. This results in much higher-quality semantic embeddings compared to vanilla BERT's [CLS] token. Because sentences are encoded individually, you can compute embeddings for a large database once, index them, and perform massive searches in milliseconds.

Because SBERT provides high-quality, efficient sentence vectors, it is widely used for semantic search, clustering and retrieval systems.

SBERT uses a Siamese network (or twin network) architecture. This means it consists of two identical BERT models that share the same weights. During inference or comparison, each sentence is fed into its own BERT branch separately. Because the weights are tied, the model maps both sentences into the same vector space using the exact same transformation logic.

Standard BERT produces an embedding for every token in a sentence. SBERT adds a pooling operation after the BERT output to derive a single, fixed-sized "sentence embedding" ($u$ and $v$). Common pooling strategies include:

- MEAN-strategy (Default): Taking the average of all contextualized word embeddings.
- MAX-strategy: Taking the maximum value across each dimension of the token embeddings.
- CLS-token: Using the output of the special [CLS] token (though this is often less effective for similarity).

To ensure the embeddings are semantically meaningful, SBERT is fine-tuned on specific tasks using different objective functions. If the model is trained on labeled pairs (like "Entailment" vs. "Contradiction"), it concatenates the embeddings $u$ and $v$ with their element-wise difference $|u - v|$ and multiplies them by a trainable weight matrix $W$:

$$\text{softmax}(W(u, v, |u - v|))$$

In regression objective, the model calculates the Cosine Similarity between $u$ and $v$. The loss (typically Mean Squared Error) is then computed against a gold standard similarity score (e.g., from 0 to 5).

For triplet objective, the model is given an Anchor sentence ($a$), a Positive sentence ($p$), and a Negative sentence ($n$). It is trained to minimize the distance between $a$ and $p$ while maximizing the distance between $a$ and $n$.

By encoding sentences into vectors once and caching them, SBERT avoids the $O(N^2)$ complexity of standard BERT cross-encoders. Finding the most similar pair among 10,000 sentences is reduced from ~65 hours with BERT to ~5 milliseconds with SBERT.

The improvements and alternatives to SBERT focus on more flexible architectures, contrastive learning techniques, and specialized retrieval strategies.

1. Sentence-T5 / ST5. Unlike SBERT's encoder-only setup, ST5 uses the T5 (Text-To-Text Transfer Transformer) Encoder-Decoder structure. It treats the entire sentence as input and guides the output side to a fixed token form to extract the sentence vector. The Encoder-Decoder structure offers more flexibility for generation-based tasks, though it may not always outperform SBERT in simple similarity scenarios.

2. SimCSE. SimCSE introduces a more lightweight contrastive learning approach compared to SBERT's reliance on labeled classification or regression data. It uses different random dropout masks on the same input sentence to create "positive pairs," training the model to pull these closer while pushing different sentences further apart. When fine-tuned on STS (Semantic Textual Similarity) data, it demonstrates strong performance and is considered a mainstream alternative to SBERT.

3. E5 and Contriever. E5 (Embedding from an Extensible Encoder Effort) utilizes large-scale contrastive learning specifically optimized for retrieval and semantic search tasks, achieving superior results in these domains. Similar to E5, Contriever employs a contrastive learning philosophy with varied pre-training strategies aimed specifically at text-to-text retrieval, showing significant effectiveness.

## 2.2 Encoder-Decoder
The core pre-training logic for Encoder-Decoder architectures generally revolves around the concept of Denoising Auto-encoding. In this approach, the model is given a "noisy" or corrupted version of an input sequence and is tasked with reconstructing the original, clean text. 

1. Span Corruption (Text Infilling). This task focuses on teaching the model to predict missing chunks of text. Several spans (usually continuous sequences of tokens) are randomly selected from the input and replaced with a special sentinel or [MASK] token. The Encoder processes the corrupted sequence, and the Decoder is responsible for reconstructing the missing parts or the entire original sequence. This includes T5 and MASS (which specifically focuses on generating only the masked portions).
2. Denoising Auto-Encoder. This task introduces more varied structural noise to make the model robust to different types of corruption. A variety of noise operations are applied to the input, such as randomly deleting tokens, replacing tokens with random ones, or shuffling the order of sentences. The Encoder reads the noisy "shuffled" sequence, and the Decoder must "denoise" the data to restore the text to its original, coherent form. BART heavily utilizes shuffling and token deletion during its pre-training.
3. Multi-Task Multi-Target Masking. This approach unifies various natural language tasks into a single framework. It frames all NLP tasks (e.g., translation, summarization, Q&A) as a "text-to-text" problem. Special instruction tags or prefixes are added before the input to guide the model. This teaches the model a universal way to handle different outputs based on the provided command. Typical models include T5, Flan-T5, and mT5 (the multilingual version).
   
### 2.2.1 BART

The model is trained to reconstruct an original document from a version that has been structurally corrupted by random noise functions. Unlike BERT, which is limited to specific masking, BART's architecture allows it to be trained using any form of document corruption. The model uses a Reconstruction Loss, specifically the cross-entropy loss between the Decoder’s predictions and the original document labels.

BART employs five main methods to corrupt text during pre-training:
1. Token Masking: Similar to BERT, random tokens are sampled and replaced with a [MASK] token.

2. Token Deletion: Tokens are randomly deleted from the sequence. Unlike masking, the model does not know where the missing tokens were, forcing it to learn both the content and the position of the missing data.

3. Text Infilling (Span Masking): Spans of text are sampled using a Poisson distribution (typically with $\lambda=3$). Each entire span is replaced by a single [MASK] token. This is more challenging than BERT (which masks single words) because the model must predict how many tokens were in the original span. While similar, BART's text infilling is considered stricter because it uses a single mask for any span length, whereas SpanBERT often uses multiple masks corresponding to the span length.

4. Sentence Permutation: The order of complete sentences within a document is randomly shuffled. The model must learn the logical flow to restore the sentences to their original positions.

5. Document Rotation: A random token is chosen as the new starting point, and the document is rotated around it. The model must identify the true original starting point of the text.

By learning to reconstruct these various types of structural noise, the Encoder learns deep feature representations while the Decoder learns the generation logic required for tasks like summarization and translation.

BART is built on a standard Encoder-Decoder framework that integrates the bidirectional encoding capabilities of models like BERT with the autoregressive decoding strengths of GPT. The encoder component receives input—often structurally corrupted text during pre-training—and processes it using bidirectional encoding. This approach is similar to the architecture used in BERT. The decoder takes the output representations from the encoder and uses an autoregressive process to predict the original sequence. It functions similarly to GPT, relying on maximum likelihood probabilities similar to those found in N-Gram or NNLM models.

The model is available in two primary sizes: BART-base, which contains 6 layers for both the encoder and decoder, and BART-large, which scales to 12 layers for each. BART utilizes GeLUs for its activation function, opting for it over the common ReLU. Weights are randomly initialized following a normal distribution with a mean of 0 and a variance of 0.02. In contrast to BERT, BART removes one feed-forward network layer before the final output prediction. There is no requirement for strict alignment between the input length of the encoder and the output length of the decoder. When representing an entire document, BART utilizes the final-dimension output of the decoder. This differs from BERT, which uses the final hidden state of the encoder for such representations. During the fine-tuning phase, both the encoder and decoder receive identical, uncorrupted input text. 

BART (Bidirectional and Auto-Regressive Transformers) fine-tuning is designed to handle a variety of NLP tasks by leveraging its unique Encoder-Decoder architecture. The process varies significantly depending on the specific task type.

1. Sequence Classification Tasks. For sentence-level classification (similar to BERT's approach), BART processes the input through both the Encoder and Decoder. The same input is fed into both the Encoder and the Decoder. Unlike BERT, which uses the first token ([CLS]), BART uses the hidden state of the very last token in the Decoder. The authors suggest that using the last token allows its representation to incorporate the semantic information of all preceding tokens. This representation is then fed into a new multi-class linear classifier.

2. Token Classification Tasks. This involves identifying specific labels for individual tokens, such as Named Entity Recognition (NER) or extractive Question Answering. The complete document is fed into the Encoder and Decoder. The model uses the top hidden state of every token from the Decoder as a feature vector to classify each individual word.

3. Sequence Generation Tasks. BART is natively suited for generative tasks like abstractive summarization or dialogue systems because its Decoder is autoregressive. The Encoder receives the source text, and the Decoder generates the target sequence. Fine-tuning is straightforward; the model is provided with the original text and the target text, and the Decoder is trained to minimize the cross-entropy loss between its output and the target.

4. Machine Translation. BART can be adapted for translation by replacing the initial embedding layer with a new, randomly initialized Encoder. The new Encoder allows the model to map a source language (e.g., Chinese) into a space that the pre-trained BART (trained on English) can understand. It has a two-step training strategy. First during freeze and warm-up, most of the BART parameters are frozen. Only the new Encoder, BART's positional embeddings, and the first layer's self-attention projection matrix are updated. Then during joint-finetuning, all parameters are then trained together for a small number of iterations to refine the entire system.

### 2.2.2 T5
T5 (Text-To-Text Transfer Transformer) processes datasets using a unified "Text-to-Text" framework. This means that every NLP task—regardless of whether it is traditionally a classification, regression, or generation task—is converted into a string-to-string format. 

Instead of having different output layers for different tasks (like BERT does for classification vs. NER), T5 uses the same model and loss function for everything. The input is a text string containing a Task Prefix. The output is a target text string representing the solution. To tell the model which task it is currently performing, a specific natural language prefix is prepended to the input data. This allows a single model to handle multiple datasets simultaneously. To translate a sentence, the input becomes: "translate English to German: That is good." $\rightarrow$ Target: "Das ist gut." For Linguistic Acceptability (CoLA), instead of outputting a class ID (0 or 1), the model is trained to generate the literal strings "acceptable" or "not_acceptable". In summarization, input is "summarize: [Document text]" $\rightarrow$ Target: [Summary text]. For tasks like the Semantic Textual Similarity Benchmark (STSB), which usually requires a floating-point number (e.g., 3.8 out of 5.0), T5 processes the score as a string representation of the number. It generates the digits as text, and during inference, these strings are converted back into numerical values for evaluation.

C4 is the massive dataset Google created to train T5. The source is years of web crawl data from Common Crawl. T5 applies heavy cleaning to this data. To ensure high quality, the following rules were applied to filter the raw web crawl:

- Line-Level Filtering: Only lines ending with terminal punctuation (e.g., periods, exclamation marks, or question marks) are kept.

- Content Removal: Pages with too few words or those containing "lorem ipsum" placeholder text are discarded.

- Code and Scripting: Lines containing Javascript or pages containing curly braces (common in programming languages) are removed.

- Quality Control: Any website containing "dirty" words is filtered out, and duplicate data entries are deleted.

- Language Detection: Only English-language pages are retained, verified using the langdetect tool.

While training on specialized data improves performance on downstream tasks within that same domain, it limits the model's multi-domain adaptability. The author recommends a three-stage approach: pre-train on the rich C4 dataset first, continue pre-training on domain-specific data, and then perform final fine-tuning. Because the C4 dataset is so vast, most models only see each sample once; however, the authors emphasize that "more data is better" even if the model cannot cover the entire set.

The authors focused on how parameters and computation are distributed across the Encoder and Decoder:
1. Model 1 (Standard Encoder-Decoder): This is the baseline. It has $L$ layers in the Encoder and $L$ layers in the Decoder. The total parameters are $2P$ and the computation cost is $M$.
2. Model 2 (Shared Parameter Encoder-Decoder): Similar to Model 1, but the Encoder and Decoder share the same weights. This reduces the parameters to $P$ while keeping computation at $M$.
3. Model 3 (Mini Encoder-Decoder): Each component has only $L/2$ layers. This keeps parameters at $P$ but halves the computation cost to $M/2$.
4. Model 4 (Language Model / Decoder-only): A single stack of $L$ layers using a causal mask (similar to GPT). Parameters are $P$ and computation is $M$.
5. Model 5 (Prefix Language Model): A single stack of $L$ layers that uses a hybrid mask (bidirectional for the input prefix, causal for the target). Parameters are $P$ and computation is $M$.

The experiments showed that the Encoder-Decoder structure (Models 1, 2, and 3) consistently achieved the best results compared to the Decoder-only or Prefix LM variants. They discovered that sharing parameters between the Encoder and Decoder (Model 2) could reduce the model's memory footprint by nearly 50% without a significant drop in performance. They defined the final "T5" as a standard Transformer Encoder-Decoder because it provided the best balance of quality and flexibility for handling diverse tasks (translation, summarization, etc.) through the same interface.

In the context of T5, Prefix LM refers to both a specific attention masking pattern and a pre-training objective. It bridges the gap between bidirectional models (like BERT) and autoregressive models (like GPT). In the first part of the sequence (the "prefix"), the model uses a fully-visible mask. Every token can attend to every other token in this section, allowing for deep, context-aware representations. After the prefix, the model switches to a causal mask (standard autoregressive masking). For these tokens, the $i$-th entry can only attend to the prefix and the tokens that came before it in the output sequence. This is technically a "non-causal decoder." It avoids the "information bottleneck" of standard causal LMs (where the first word can't see the rest of the sentence) while still allowing the model to generate text token-by-token. 

While T5 is an Encoder-Decoder, the author chose to modify it from the original Transformer in three specific ways to ensure it was the "best" version:
1. Simplified Layer Norm: Removing the bias term ($y = w \cdot \frac{x}{RMS(x)}$) for better stability.
2. Relative Position Bucketing: Replacing absolute position encodings with a bucketing system to handle various sequence lengths more effectively.
3. Post-Normalization: Moving the residual connection to occur after the layer normalization.

The author compared three broad categories of self-supervised methods to see which fundamental style of learning worked best:
- Language Modeling: Predicting the next token from left to right.
- BERT-style (Denoising): Corrupting parts of the text and then attempting to reconstruct or restore them.
- Deshuffling: Taking a shuffled sentence and trying to restore its original order.
  
BERT-style (Denoising) was found to be the most effective, outperforming both standard language modeling and deshuffling.

Once BERT-style denoising was chosen, the authors explored three specific ways to "corrupt" the input text:
- Masking: Replacing individual corrupted tokens with a special mask symbol ($[M]$).
- Replace Spans (Text Infilling): Grouping adjacent masked tokens into a single special sentinel token. Instead of predicting the whole sentence, the model only predicts the missing "spans".
- Drop Tokens: Simply deleting random tokens from the input without using any replacement symbols.
  
Replace Spans was selected as the T5 pre-training objective. While it performed similarly to masking, it was computationally more efficient because the decoder only had to generate the missing spans rather than the entire sentence. 15% was found to be the most effective corruption rate, mirroring the rate famously used by BERT. An average span length of 3 tokens yielded the best results for general language understanding.

The authors of T5 explored several fine-tuning strategies to determine how to best transition a model from general pre-training to specific downstream tasks. They focused on two main areas: the fine-tuning method (which parameters to update) and the multi-task strategy (how to mix different tasks during training).

The authors compared three ways to adapt the pre-trained model to a new task:
- All-Parameter Fine-Tuning (The Baseline): Updating every single parameter in the model for the new task. Whil
- Adapter Layers: Instead of updating the whole model, small "adapter" blocks are inserted after each Transformer layer. Only these new blocks are trained, while the original weights remain frozen. This is more parameter-efficient but can increase computational overhead during inference.
- Gradual Unfreezing: Starting by training only the final layers and slowly "unfreezing" earlier layers until the entire model is being updated. This helped slightly with stability but didn't show massive gains over standard fine-tuning.

T5 is unique because it treats all tasks as "text-to-text." This allowed the authors to experiment with mixing unsupervised (pre-training) and supervised (fine-tuning) tasks together. By a Examples-Proportional Mixing strategy, tasks are sampled based on the size of their dataset. Larger datasets appear more often during training. Larger datasets dominated the training, causing the model to perform poorly on smaller, specialized tasks. In Equal Mixing strategy, every task is sampled with equal frequency, regardless of dataset size. The model overfits on small datasets quickly and underperforms on large ones. The Temperature-Scaled Mixing strategy is a middle ground that uses a "temperature" $(T)$ to flatten the distribution. Higher $T$ makes the sampling more equal; lower $T$ makes it more proportional. This was the most balanced approach for multi-task pre-training

The most successful strategy identified in the paper wasn't just doing one or the other, but a specific sequence:
1. Multi-Task Pre-training: Train the model on a mix of the C4 unsupervised task and many supervised tasks (using temperature-scaled mixing).
2. Task-Specific Fine-tuning: After the broad multi-task phase, perform a final round of fine-tuning on a single target task.
The multi-task phase gives the model a broad "education," while the final fine-tuning phase allows it to specialize and achieve peak performance on a specific benchmark.

FLAN (Fine-tuned Language Net) is a technique developed by Google to make Large Language Models (LLMs) better at following instructions. The core innovation of FLAN is Instruction Tuning. Instead of fine-tuning a model for just one task (like only translation), researchers fine-tuned it on a massive "mixture" of over 1,800 different tasks simultaneously, all formatted as natural language instructions.

The development of Flan-T5 represents the ultimate "scaling up" of the T5 framework. Flan-T5 does not change the core architecture of T5; it uses the standard Transformer Encoder-Decoder (specifically T5 version 1.1). It features GeLU activations, lacks parameter sharing between encoder and decoder, and uses Relative Position Bucketing. It is pre-trained on the C4 dataset using the "Span Corruption" objective—masking 15% of the text and training the model to fill in the missing 3-token spans.

The "Flan" part refers to the instruction-tuning phase. This is what separates Flan-T5 from a regular T5 model. The model is fine-tuned on 1,836 tasks simultaneously. This is a massive jump from the dozens of tasks used in the original T5 paper. For every task, researchers created multiple natural language templates. Instead of just learning [Input] -> [Output], the model learns to follow commands like "Summarize this for a second grader". There are three primary benefits that Flan brings to the T5 model:
- Downstream Instruction Execution: It allows the model to better parse and execute specific commands (like "Translate this sentence to French") with more accurate and targeted results.
- Generalization to New Tasks: It significantly improves the model's performance on zero-shot or few-shot tasks—tasks the model was never explicitly trained on—by helping it understand the "logic" of following directions.
- Retained Flexibility: It maintains the original T5 "text-to-text" flexibility, allowing it to handle any variety of input and output formats.

For generation tasks, the Flan-T5 loss function is:

$$\mathcal{L}_{FlanT5} = \mathbb{E}_{(instr, x, y) \in D} [-\log P_\theta(y \mid instr, x)]$$

The model uses cross-entropy loss calculated at each token timestep to ensure the generated output $y$ matches the target given the instruction and input.

## 2.3 Non-Causal Decoder-Only
### 2.3.1 UniLM
UniLM (Unified Language Model) is a versatile Transformer-based model designed by Microsoft Research to excel in both Natural Language Understanding (NLU) and Natural Language Generation (NLG). While BERT is great at understanding text (bidirectional) and GPT is great at generating it (unidirectional), UniLM unifies these capabilities into a single set of parameters.

UniLM (Unified pre-trained Language Model) is designed as a single Transformer model that can be configured to act like BERT, GPT, or a Sequence-to-Sequence model simply by changing its Self-Attention Masking. In Bidirectional LM (Understanding), the Mask is a zero matrix ($M=0$), meaning no tokens are blocked. Every token in the sequence can attend to every other token (left and right). It functions like BERT, making it ideal for Natural Language Understanding (NLU) tasks like classification. In Unidirectional LM (Generation), the Mask is a triangular matrix. A token can only attend to itself and tokens to its left. In Sequence-to-Sequence LM (Hybrid), the Mask is a hybrid pattern where the input is split into two segments ($S1$ and $S2$). $S1$ tokens can see each other bidirectionally, while $S2$ tokens can see all of $S1$ but only preceding tokens within $S2$. This allows the model to act like an Encoder-Decoder (T5/BART style) for tasks like summarization and translation. 

UniLM uses a Unified Masked Language Modeling (Masked LM) framework for pre-training, which incorporates four distinct tasks. UniLM cycles through different objectives by adjusting its Mask Matrix within a shared Transformer network: Bidirectional LM, Unidirectional LM, Sequence-to-Sequence (Seq-to-Seq) LM, and Next Sentence Prediction (NSP). 1/3 of the time is dedicated to the Bidirectional LM task. 1/3 of the time is dedicated to the Unidirectional LM task (split equally between left-to-right and right-to-left). 1/3 of the time is dedicated to the Seq-to-Seq LM task. UniLM utilizes the WordPiece tokenization method for its vocabulary.

UniLM adapts its single Transformer stack for two primary types of finetuning tasks:
- Natural Language Understanding (NLU): The model is treated as an encoder. A [SOS] token is used to extract a sentence-level feature vector, which is then fed into a newly added classification layer.
- Natural Language Generation (NLG): The task is formatted as a sequence: [SOS] S1 [EOS] S2 [EOS], where $S1$ is the source and $S2$ is the target. For NLG, the model randomly masks parts of the target sequence ($S2$) and minimizes cross-entropy loss using a teacher-forcing approach with random sampling.
### 2.3.2 GLM

GLM (General Language Model) is a versatile Transformer-based architecture designed to unify the strengths of BERT (understanding) and GPT (generation) into a single model. While most early models specialized in either understanding or generation, GLM uses a unique autoregressive blank infilling objective that allows it to perform competitively across Natural Language Understanding (NLU), unconditional generation, and conditional generation.

Unlike BERT, which predicts masked tokens independently, or GPT, which only looks at previous tokens, GLM handles "blanks" (masked spans of text) as follows:
1. Blanking Out Spans: Random spans of tokens are removed from the input and replaced with a single [MASK] token.
2. Part A and Part B: The model splits the task into two parts. Part A is the corrupted text (the original sentence with masks), and Part B consists of the missing spans.
3. Autoregressive Generation: The model generates the spans in Part B one by one. Crucially, it can see the entire context of Part A but generates the missing parts in an autoregressive (step-by-step) manner.
   
## 2.4 Causal Decoder-Only
### 2.4.1 GPT
GPT is the Transformer Decoder with the "Encoder-Decoder Attention" layer removed. It is also equivalent to a Transformer Encoder layer, but with the standard Multi-Head Attention replaced by Masked Multi-Head Attention (also known as Masked Self-Attention).
#### 2.4.1.1 GPT-1
The pre-training of GPT-1 is a classic example of Generative Pre-training using a causal language modeling objective. The primary goal of GPT-1 pre-training is to maximize the likelihood of the next token given its preceding context. This is mathematically expressed as:

$$L_1(\mathcal{U}) = \sum_{i} \log P(u_i | u_{i-k}, \dots, u_{i-1}; \Theta)$$

- $\mathcal{U} = \{u_1, \dots, u_n\}$: The unlabeled corpus of tokens.
- $k$: The size of the context window.
- $P$: The conditional probability modeled by a neural network with parameters $\Theta$.

The model achieves Autoregression through a Causal Mask, which ensures that token $t$ can only attend to tokens at positions $< t$. The mask is defined as:

$$
\text{Casual Mask}(i, j) = 
\begin{cases} 
0, & j \leq i \\ 
-\infty, & j > i 
\end{cases}
$$

GPT-1 utilizes a multi-layer Transformer Decoder (specifically 12 layers). Unlike the original Transformer, it only uses the decoder block and removes the encoder-decoder attention layer. The initial hidden state $h_0$ in the input embedding is formed by combining token embeddings and learned positional embeddings:

$$h_0 = UW_e + W_p$$

Where $U$ is the context vector of tokens, $W_e$ is the token embedding matrix, and $W_p$ is the position embedding matrix.

In the Transformer blocks, the hidden state is processed through $n$ layers of Transformer blocks. The final distribution over the vocabulary is calculated via a Softmax layer:

$$P(u) = \text{Softmax}(h_n W_e^T)$$

The model was trained with a focus on stability and performance scaling, including 12 layers, 768-dimensional hidden states, and 12 attention heads. In the feed-forward layer there are 3072 hidden units. It uses the Adam optimizer with a max learning rate of $2.5 \times 10^{-4}$. The learning rate followed a linear warmup for the first 2,000 steps, followed by a cosine annealing decay to 0. For regularization, it is dropout with a rate of 0.1 and a modified version of $L_2$ regularization ($w=0.01$). GELU (Gaussian Error Linear Unit) was used instead of standard ReLU. Byte Pair Encoding (BPE) with a vocabulary size of 40,000 was used for tokenization. GPT-1 also used post-normalization.

Training stops based on accuracy. The model compares its generated text with the original text; if the accuracy reaches the expected threshold or begins to fluctuate without improvement (plateauing), the pre-training phase is concluded.

The primary dataset BooksCorpus containing over 7,000 unpublished books (approx. 800 million words). This corpus was chosen specifically because it contains long sequences of continuous text, which allows the model to learn long-range linguistic dependencies. A secondary dataset (1B Word Benchmark) used for comparison (though it is noted that this dataset is shuffled at the sentence level, destroying long-range structures).

Fine-tuning adapts the pre-trained model to a labeled dataset $\mathcal{C}$, where each instance consists of an input sequence $x^1, \dots, x^m$ and a corresponding label $y$. The input sequence is passed through the pre-trained Transformer layers to obtain the final hidden state (embedding) of the last token, denoted as $h_l^m$. This embedding is fed into an added Linear Output Layer with parameters $W_y$:

$$P(y | x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)$$

The goal is to maximize the likelihood of the correct labels across the dataset:

$$L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y | x^1, \dots, x^m)$$

A key finding in GPT-1 is that including the original language modeling objective as an auxiliary task during fine-tuning improves generalization and accelerates convergence. The final optimization target $L_3(\mathcal{C})$ is a weighted sum:

$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C})$$

Where $\lambda = 0.5$.

To handle different NLP tasks without changing the model architecture, GPT-1 uses specific input formats, often utilizing "Delimiters" to separate text segments. 
- Classification: The input is standard text; the vector of the last token is used for the final classification.

- Inference (Entailment): The input is formatted as Premise + Delimiter + Hypothesis. The last token’s vector determines if the relationship holds (binary classification).

- Similarity: Since similarity is symmetric, the model processes two versions: Sentence A + Delimiter + Sentence B and Sentence B + Delimiter + Sentence A. The resulting vectors are added before passing through the linear layer.

- Question Answering (QA): The context and question are concatenated with each possible answer using delimiters (e.g., [Context; Question; Answer i]). Each sequence is processed independently, and a softmax is applied across all answer scores to find the most probable result.

Typically 3 epochs of training are sufficient. Linear learning rate decay with a warmup over the first 0.2% of training. Dropout is added to the classifier with a rate of 0.1. The learning rate is $6.25 \times 10^{-5}$. Batch size is 32. 

GPT-1 outperformed state-of-the-art models in 9 out of 12 benchmark supervised tasks. 

**Advantages**
- Two-Stage Training Framework: By combining large-scale unsupervised pre-training with supervised fine-tuning, the model significantly improved performance across a wide variety of NLP tasks.

- Strong Capture of Long-Range Dependencies: Utilizing the Transformer architecture's self-attention mechanism allowed the model to capture complex linguistic dependencies over long distances more effectively than previous RNN-based models.

- Excellent Transfer Learning: Once pre-trained, the model demonstrated high adaptability, requiring only task-specific fine-tuning to excel in various downstream applications.

**Disadvantages**
- Unidirectional Constraint: As a unidirectional language model, it only considers context from left to right. This limits its ability to fully understand information that relies on bidirectional context (unlike BERT).

- Limited Model Capacity: With 117 million parameters, the model scale is relatively small by modern standards. This limited capacity affects its ability to model highly complex or nuanced linguistic phenomena.

- Heavy Reliance on Fine-tuning: GPT-1 is not a "universal linguist" out of the box. It requires specific fine-tuning for every individual downstream task, which increases the complexity and computational cost of deployment.

#### 2.4.1.2 GPT-2

GPT-2 represents a significant shift from GPT-1 by moving toward a Zero-shot learning paradigm, predicated on the idea that supervised learning is a subset of language modeling when the model capacity is large enough. While GPT-1 relied on a two-stage process of pre-training and fine-tuning, GPT-2 focuses on Zero-shot capabilities. It removes the task-specific fine-tuning layer entirely. GPT-2 uses natural language prompts to identify tasks automatically. For example, if the model is trained on text like "Michael Jordan is the best basketball player in history," it naturally learns to answer the question "Who is the best basketball player in history?" without being explicitly fine-tuned for Q&A. 

GPT-2 increased the number of Transformer layers to 48 and the hidden layer dimension to 1600. The total parameter count reached 1.5 billion, a 15x increase over GPT-1's 117 million. GPT-2 moved from the 5GB BooksCorpus to a much larger 40GB dataset called WebText, consisting of approximately 8 million high-quality articles scraped from Reddit. To avoid data leakage, all Wikipedia articles were removed from WebText. The Byte Pair Encoding (BPE) vocabulary size was expanded from 40,000 to 50,257.

GPT-2 moved Layer Normalization to the start of each sub-block (Pre-norm) rather than the end (Post-norm) to stabilize and simplify the training process. An additional Layer Normalization was added after the final Self-Attention block. The initial values of the residual layers are scaled by a factor of $\frac{1}{\sqrt{N}}$, where $N$ is the number of residual layers. The batch size was increased from 64 to 512, and the context window (maximum sequence length) was doubled from 512 to 1024 tokens.

GPT-2 pre-training follows a traditional language modeling objective scaled to a larger, more diverse dataset with specific handling for document sequences. GPT-2 uses the same fundamental objective as traditional language models: predicting the next token in a sequence. The training objective is to minimize the negative log-likelihood of the next token given all previous tokens in the sequence:

$$\mathcal{L}_{\text{LM}} = \mathbb{E}_{\mathbf{x} \sim \text{Data}} \left[ -\sum_{t=1}^{|\mathbf{x}|} \log P(x_t \mid x_{1..t-1}) \right]$$

To ensure the model only looks at past information and not future tokens, it utilizes a causal mask. This mask (inherited from GPT-1) allows a token at position $i$ to attend to positions $j \leq i$, while setting attention to $-\infty$ for any $j > i$.

To manage long documents within the model's architectural limits, GPT-2 employs a specific block-based strategy. Documents are sliced into fixed-length "blocks" of 1024 tokens. If a document is shorter than the 1024-token limit, it is either padded or merged with other documents to fill the block. The model performs the standard left-to-right Language Modeling objective on these randomized blocks during training.

**Advantages**
- Powerful Generation Capabilities: GPT-2 excels at generating coherent, smooth text, including stories and even code snippets.

- Contextual Understanding: By learning from massive amounts of text data, the model can understand context and generate logically consistent responses.

- Multi-domain Application: The model shows good suitability for tasks in various fields, such as machine translation, summarization, and dialogue systems.

- Proof of Concept for LLMs: Its Zero-shot training approach effectively proved the possibility of a universal model, marking the early stages of the LLM and AIGC era.

**Disadvantages**
- Potential for Inappropriate Content: Because the training data is scraped from the internet, the model may generate content that contains bias or is inappropriate.
  
- High Computational Resource Demand: Due to its large model size, both the training and inference processes require significant computational resources.

- Lack of Common Sense Reasoning: GPT-2’s performance can be unsatisfactory when handling tasks that require deep common-sense reasoning.

#### 2.4.1.3 GPT-3

GPT-3 introduces several massive leaps in scale and capability over GPT-2, primarily focusing on model size, data diversity, and the emergence of in-context learning. The most apparent improvement is the exponential increase in parameters. Parameter counts in GPT-2 ranged from 150 million to 1.5 billion (Small to XL). GPT-3 reached a maximum scale of 175 billion parameters, representing a massive expansion in model capacity. GPT-3 was trained on a much broader and larger corpus compared to the WebText used for its predecessor. It used hundreds of gigabytes of multi-source data, including CommonCrawl, WebText2, Books, and Wikipedia, providing much wider domain coverage. For GPT-3, the scale is so vast that training requires extreme computational power and memory resources that only a very few organizations can afford; inference costs are similarly high.

While GPT-2 showed some potential in zero-shot scenarios, it often required fine-tuning to be truly effective. GPT-3 revolutionized this with training-free capabilities. GPT-3 can perform tasks like translation, Q&A, and summarization simply by providing a few examples or instructions in the prompt, without any gradient updates or fine-tuning. This in-context learning happens entirely during the inference stage. Instead of using backpropagation to update weights (as in fine-tuning), the model uses its existing self-attention mechanism to "locate" the relevant task within its pre-trained latent space. You provide the model with a task description followed by a few demonstration pairs (Input: $x$, Output: $y$) and a final query $x'$. The model reads these examples like a sequence, identifying consistent relationships in structure, meaning, or logic. Then it uses the examples as a semantic prior. It calculates the probability of the next token by conditioning on the entire history of the prompt, effectively "copying" the demonstrated pattern for the new input.

Some researchers argue that the Transformer's attention mechanism acts as an "inner loop" of an optimization process. In this view, the model produces "meta-gradients" during the forward pass that behave similarly to the explicit gradients calculated during fine-tuning. Another view suggests the model uses the prompt to narrow down which "latent concept" (learned during its massive pre-training on the web) it should use for the current task.

GPT-3 introduces a specific modification to the standard Transformer attention mechanism to handle its massive scale and long context window. The most significant change in GPT-3's attention compared to GPT-2 is the use of Alternating Dense and Sparse Attention patterns.

- Standard (Dense) Attention. Every token attends to every single previous token in the sequence. This is represented by a fully "filled" or dark lower-triangular matrix. It provides the most comprehensive "Global View," allowing the model to find relationships between any two words, no matter how far apart. To calculate the correlations between any two vectors, a matrix of size $n^2$ is required. Theoretically, both the calculation time and memory occupancy grow at a rate of $O(n^2)$, where $n$ is the sequence length.

- Dilated Self-Attention. Inspired by dilated convolutions, this type restricts correlation so that each element only connects to other elements at specific relative distances (e.g., $k, 2k, 3k, \dots$). $k > 1$ is a pre-set hyperparameter. Each element only calculates correlations with $n/k$ other elements. In ideal conditions, the operational efficiency and memory occupancy are reduced to $O(n^2/k)$, which is $1/k$ of the original requirement.

- Local Self-Attention. This introduces local correlations by restricting each element to attend only to its immediate $k$ neighbors before and after it. It maintains a window of size $2k + 1$. Each element only interacts with $2k + 1$ other elements, causing complexity to grow linearly ($O(kn)$) with sequence length $n$. While ideal for efficiency, this approach sacrifices the ability to capture long-range dependencies.

- Sparse Self-Attention. This is a hybrid approach that merges the characteristics of Dilated and Local self-attention. In the attention matrix, all correlations are set to 0 except for those within a relative distance of $k$ (Local) or at specific intervals like $k, 2k, 3k, \dots$ (Dilated). It combines "locally dense" and "remotely sparse" correlations. This reduces computational complexity, saves memory/power, and allows the model to process much longer input sequences while focusing most heavily on nearby context. Because it relies on the assumption that truly dense long-range dependencies are rare, it may perform poorly on long-text modeling tasks where those global relationships are actually essential.

GPT-3 significantly increases the depth and width of the Transformer architecture compared to its predecessors. It utilizes 96 layers of multi-head self-attention. Each layer contains 96 attention heads. The embedding dimension features a word vector length of 12,888. The context window is expanded to 2,048 tokens, double that of GPT-2.

The model was trained on a massive, diverse collection of text totaling hundreds of gigabytes, with specific weights assigned to different sources based on quality.
- C4 (Common Crawl): Filtered version of Common Crawl (570GB) making up 60% of the training data.

- WebText2: High-quality web content making up 22%.

- Books1 & Books2: Two internet-based book corpora accounting for 16%.

- Wikipedia: English Wikipedia content making up 3%.

Like previous GPT models, GPT-3 uses an unsupervised, self-supervised learning objective. The model is trained to predict the next token given all previous tokens in a sequence. It uses a causal mask to ensure that during training, each token can only "attend" to current and previous positions, preventing it from seeing future information. To handle the large context window efficiently, it alternates between dense self-attention and Sparse Self-Attention patterns (combining local and dilated attention) to reduce computational complexity.

The primary goal of GPT-3's massive pre-training was to achieve In-context Learning, a form of meta-learning where the model adapts to new tasks without updating its parameters.

- Zero-shot: The model performs a task based only on a natural language description (e.g., "Translate English to French:").

- One-shot: The model is given the description plus a single example.

- Few-shot: The model is given a description and several examples within the prompt.

ICL is described as a form of "learning how to learn". During massive unsupervised pre-training, the model develops broad strategies to adapt to unseen tasks quickly. As model size increases, the ability to learn from context becomes more efficient. Large models show a "steeper" learning curve, meaning they use the information in the prompt much more effectively than smaller models. There is a smooth, positive correlation between model capacity and performance across Zero, One, and Few-shot scenarios. Larger models are fundamentally better at "Meta-learning". 

**Advantages**
- Powerful Language Generation Capabilities: GPT-3 can generate coherent and creative text, making it suitable for writing assistance and content creation.

- Few-shot Learning Ability: It demonstrates excellent performance in few-shot scenarios, where it can perform specific tasks provided with only a small number of examples.

- Wide Range of Application Scenarios: The model can be applied to diverse Natural Language Processing (NLP) tasks, including text classification, sentiment analysis, and code generation.

- Superior Contextual Understanding: Compared to GPT-2, GPT-3 maintains higher coherence over long contexts and provides more consistent answers.

**Disadvantages**
- High Computational Resource Requirements: Due to its massive parameter count (175 billion), training and deploying GPT-3 requires significant computing power and storage space.
  
- Potential for Inappropriate Content: The model may generate text containing biases or inappropriate content, reflecting the biases present in its original training data.

- Lack of Common Sense Reasoning: Like its predecessors, GPT-3 may still perform unsatisfactorily on tasks that require deep common sense reasoning.

- High Costs: The extreme scale of the model means that only a few organizations can afford the immense computational and memory resources required for its training.
  
#### 2.4.1.4 GPT-3.5

While GPT-3 has strong zero-shot understanding, it struggles specifically in conversational settings. The model does not inherently possess human values, making its answers sometimes lack diplomacy or "smoothness". The way the model generates text often deviates from what humans naturally expect in a conversation. Also, the model has a tendency to make things up. The fundamental reason for these flaws is that GPT-3 is purely a language model. While it has mastered grammar rules and built vast knowledge networks from massive datasets, it fundamentally does not understand human preferences. It was trained to predict the next word, not to please a human user. GPT-3.5 introduces RLHF as a fix.

Because the base model is a generative model based on prompt learning, the GPT SFT dataset is constructed entirely of prompt-response pairs. The data comes from two main sources: users of OpenAI's Playground and 40 specifically hired and trained human labelers. During this SFT phase, the human labelers annotated a total of 13k (prompt, completion) pairs. The labelers were tasked with writing instructions that fulfilled three specific criteria to ensure the model learned to handle a variety of requests:

1. Simple Tasks: Labelers were asked to come up with arbitrary, simple tasks while ensuring a wide diversity of task types.

2. Few-shot Tasks: Labelers provided an instruction followed by multiple "query-response" pairs to demonstrate how to complete that specific instruction.

3. User-related Tasks: Labelers were given real use cases pulled from OpenAI's API and wrote instructions based on those actual user scenarios.

The SFT phase uses the exact same network architecture as the initial pre-training phase. The model is trained by maximizing the likelihood function. The mathematical objective shown in the text is:

$$L(\theta) = \prod_{i=1}^{N} \pi^{SFT}(y_i|x_i; \theta)$$

The core goal of the Reward Model is to act as a mathematical proxy for human preference. It learns to score text based on what humans find helpful, safe, and accurate, penalizing things like bias or toxic content. Instead of just asking the model for one answer, the system takes a random prompt and asks the previously trained Supervised Fine-Tuned (SFT) model to generate several different responses. Specifically, it generates $K$ answers (where $K$ is usually between 4 and 9). Human labelers step in and review these $K$ answers. Their job isn't to write a better answer, but simply to rank the generated answers from best to worst based on quality and alignment guidelines. 

To train the model, the system breaks these rankings down into pairs. From the $K$ ranked answers, it creates every possible combination of two answers, resulting in $\binom{K}{2}$ pairs. For example, if the model generated answers A, B, and C, and the human ranked them A > B > C, the training data becomes three pairs:
- (Prompt, A is better than B)
- (Prompt, A is better than C)
- (Prompt, B is better than C)

The Reward Model itself is a neural network that takes in a prompt and a response, and outputs a single scalar number (a score). The training objective is to minimize the Pairwise Ranking Loss. The mathematical formula shown in the image is:

$$L(\theta) = -\frac{1}{\binom{K}{2}} E_{(x, y_w, y_l) \sim D} [\log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))]$$

$r_\theta(x, y_w)$ is the score the model gives to the "winning" (better) response. $r_\theta(x, y_l)$ is the score the model gives to the "losing" (worse) response. The model learns by trying to make the difference between the winning score and the losing score as large as possible. The $-\frac{1}{\binom{K}{2}}$ at the beginning simply averages the loss across all the pairs for that specific prompt. This ensures that a prompt which happened to have 9 generated answers doesn't overpower a prompt that only had 4 generated answers during training.

Instead of feeding each pair into the model one at a time, all pairs associated with a single prompt are fed into the model simultaneously as a single batch. This solves two major problems:

- Prevents Overfitting: If processed sequentially, a single response would be used in multiple separate gradient updates ($K-1$ times), which risks the model overfitting to that specific text. By doing it in one batch, it only participates in one combined gradient calculation.

- Massive Speed Increase: Calculating the score $r_\theta(x, y)$ is computationally heavy. If done sequentially (updating the model's parameters after every pair), you have to recalculate the score for the exact same text over and over—taking $K(K-1)$ calculations. By batching them, the model parameters don't change during the batch, so the model only has to calculate the score for each answer exactly once ($K$ times).

Once this Reward Model is fully trained, it can instantly look at any new text GPT generates and grade it like a human would.

In this third step of RLHF, the system uses the Reward Model (RM) to continuously train and fine-tune the Supervised Fine-Tuned (SFT) model so it better aligns with human intent. Unlike previous steps, the data used here does not rely on human labelers. Instead, it uses a massive dataset of real, unlabelled prompts submitted by users through the GPT-3 API (broken down into generative tasks, Q&A, brainstorming, etc.). It relies on PPO (Proximal Policy Optimization). Standard Policy Gradient algorithms are highly sensitive to step size, making training volatile. PPO solves this by enabling small, stable batch updates over multiple training steps. The ultimate goal of this phase is to maximize the following objective function. The formula balances three competing goals to ensure the model gets smarter without breaking:

$$\text{objective}(\phi) = E_{(x,y)\sim D_{\pi_\phi^{\text{RL}}}} \left[ r_\theta(x, y) - \beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right) \right] + \gamma E_{x \sim D_{\text{pretrain}}} \left[ \log (\pi_\phi^{\text{RL}}(x)) \right]$$

The current RL model ($\pi_\phi^{\text{RL}}$) generates a response ($y$) for a given prompt ($x$). The Reward Model ($r_\theta$) evaluates this and assigns a score. The primary objective is to push the model to generate text that earns the highest possible score. If the model only chases the highest reward, it will eventually "game" the system, finding weird linguistic shortcuts that trick the Reward Model into giving high scores but result in unnatural text for humans. To prevent this, the formula includes a KL Divergence penalty ($\beta \log(\dots)$). It constantly compares the output distribution of the new RL model with the output distribution of the original SFT model ($\pi^{\text{SFT}}$). If the new model's behavior deviates too far from the original, safe SFT model, it gets mathematically penalized. The parameter $\beta$ controls the strictness of this penalty. (Using just Goal 1 and Goal 2 is standard PPO). To further ensure the model doesn't suffer from "mode collapse" (forgetting its general language understanding while hyper-focusing on human alignment), the system feeds it original GPT-3 pre-training data ($D_{\text{pretrain}}$). The term $\gamma E_{x \sim D_{\text{pretrain}}} \dots$ evaluates how well the RL model still performs on these foundational language tasks. If its foundational performance drops, it faces another penalty, controlled by the parameter $\gamma$.


**Advantages**
- Enhanced Truthfulness and Value Alignment: Because of the human annotation introduced during the training process, GPT-3.5 generates outputs that are significantly more factual and better aligned with human values and expectations.

- More Natural Responses: By building on GPT-3’s strong foundation for generalization and text generation, and pairing it with human-written prompts and ranked results, the model produces responses that feel much more natural and conversational.
  
- Improved Harmlessness: While the improvement is noted as somewhat limited, GPT-3.5 is still measurably better at reducing the amount of harmful or toxic content it generates compared to its predecessors.

**Disadvantages**
- Performance Drop on General NLP Tasks: This is often referred to as the "alignment tax." Because the model was heavily optimized for specific conversational and instruction-following tasks, its performance on other, more general Natural Language Processing (NLP) benchmarks actually decreased.
  
- Potential for Absurd Outputs: Even with human feedback (RLHF), the model isn't perfect. It can still "hallucinate" and generate content that is completely untrue, illogical, or nonsensical.

- Over-sensitivity to Instructions: The model is highly sensitive to the exact phrasing of input prompts. Because of this, it can sometimes overthink or misinterpret very simple concepts, leading to answers that completely miss what the user actually wanted.

#### 2.4.1.5 GPT-4
GPT-4 is a multimodal model. Compared to GPT-3.5 or ChatGPT, it can process both image and text inputs to generate text outputs. Despite its ability to handle multiple media types, its output mechanism remains an autoregressive next-token prediction task.

GPT-4 utilizes a Mixture of Experts (MoE) architecture rather than a standard dense model. This means that instead of every part of the model processing every piece of data, different specialized components (experts) work together to produce the final output. The model contains approximately 1.76 trillion parameters overall. Its basic width and depth are roughly the same as GPT-3 (which had 175B parameters), with the primary difference being that GPT-4 has 16 times as many Multi-Layer Perceptrons (MLPs). It is built on 120 Transformer layers.

Inside each Transformer layer, the parameter distribution is split between two main mechanisms. 55 Billion (55B) parameters are in the attention mechanism. GPT-4 consists of 16 distinct expert models. Each individual expert has 111 Billion (111B) parameters. (Totaling $111\text{B} \times 16$). When text (tokens) passes through the model, it does not activate all 16 experts at once. A routing algorithm evaluates each token and sends it to exactly two of the 16 MLPs for computation. The sequence length (seq_len) is 8,000 (8k) tokens, and the routing is distributed so that each individual MLP ends up handling roughly 1,000 (1k) tokens during the process. 

The pretraining of GPT-4 requires an immense amount of computational power and complex engineering to manage its massive size. The model was trained using an estimated 3,125 machines, which totals around 25,000 A100 GPUs. It uses a massive batch size of 60 Million tokens per step, with a sequence length (seq_len) of 8k. To physically fit the model into memory and compute it efficiently, the training uses a combination of three standard parallelism methods:

- Tensor Parallelism (8-way): The matrix operations for a single layer are split across 8 GPUs simultaneously. Communication overhead for this is kept under 15%.

- Pipeline Parallelism (16-way): The model's layers are sliced vertically across 16 different machines. For instance, Machine 0 computes Layers 0-7, passes the output down to Machine 1 for Layers 8-15, all the way to Machine 15 for the final layers (120-127).

- Data Parallelism (196-way): This entire pipeline and tensor setup is replicated 196 times so that the system can process different chunks of the 60M token batch at the same time.

Because GPT-4 uses a Mixture of Experts (MoE) architecture, standard 3D parallelism isn't enough to prevent out-of-memory errors. Even when split across 8-way Tensor and 16-way Pipeline parallelism, a single GPU still needs to manage about 14 Billion parameters. If calculating gradients in standard FP32 precision, this requires 84 GB of VRAM per GPU. A standard A100 GPU cannot hold this. To solve this memory crisis, the training implements Expert Parallelism. Instead of every GPU holding all the MoE Multi-Layer Perceptrons (MLPs), the individual "experts" are distributed across different GPUs. During training, an Attentive Gating Network determines which expert needs to process which token. The system then uses "all-to-all" communication networks to physically route the data from the attention layers on one GPU to the specific MLP expert sitting on a different GPU, allowing the massive model to train without crashing the hardware.

**Advantages**
- Stronger Reasoning Ability: GPT-4 shows significant improvement over its predecessors in logical reasoning, grasping complex contexts, and managing multi-turn conversations. This allows it to tackle much harder problems and provide more accurate answers.
  
- Multimodal Support: It accepts both text and image inputs. This drastically broadens its real-world application, allowing it to perform tasks like describing images or answering questions based on a combination of visual and text data.
  
- Better Knowledge Integration and Accuracy: When handling factual questions, GPT-4 is noticeably more accurate than GPT-3 or GPT-3.5. It is better at synthesizing information from various sources and is less prone to generating false information (hallucinations).

**Disadvantages**
- High Computational Resource Consumption: Because the model is incredibly massive, the computing power and storage required to train and deploy it are far higher than previous generations. This creates major financial and hardware challenges for researchers and developers.
  
- Still Produces Biased and Harmful Output: While it has improved guardrails, GPT-4 is not immune to generating biased, inappropriate, or harmful content. This issue becomes even more complicated to manage with the addition of multimodal (image) generation.

- Reasoning is Still Limited: Despite its massive upgrades, it is not perfect. It can still make mistakes when faced with highly complex logical problems, particularly if it lacks sufficient context or niche knowledge, which lowers its reliability in those specific edge cases.
  
#### 2.4.1.6 OpenAI o1

Unlike previous models that immediately start predicting the next word to form an answer, o1 is designed to "think" first. It generates a chain of logical reasoning before producing the final output. This significantly boosts its performance on complex reasoning tasks and makes its outputs more safe and consistent.

The training process has been overhauled using Reinforcement Learning. Rather than just learning to mimic data, o1 is actively trained to try out different strategies, identify and correct its own mistakes, and improve its overall decision-making quality. This is paired with highly diverse datasets and a very strict data filtering pipeline to remove personal information and ensure content safety.

Older models often suffered from "over-refusal" (accidentally blocking safe, harmless requests just to be safe). o1 introduces a new "Deliberative Alignment" technique. Because the model can reason, it is trained to explicitly evaluate whether its planned answer matches safety policies before it speaks. This optimizes the model to successfully block actual policy violations while drastically reducing the false-positive rejection of harmless prompts.

#### 2.4.1.7 GPT-OSS
GPT-OSS is an Autoregressive Mixture of Experts (MoE) Transformer model built upon the foundational designs of GPT-2 and GPT-3. OpenAI released two different sizes for this model:
- GPT-OSS-120B: 36 layers, 116.8 Billion total parameters, with 5.1 Billion active parameters during generation.
- GPT-OSS-20B: 24 layers, 20.9 Billion total parameters, with 3.6 Billion active parameters.
Both model sizes share a hidden layer dimension of 2880. They use RMSNorm on the activations before every Attention and MoE module, following the Pre-LN (Pre-Layer Normalization) setup used in GPT-2. The MoE modules utilize a gated SwiGLU activation function.

The attention mechanism inherits GPT-3's attention design, alternating between a banded window mode (with a window width of 128 tokens) and a fully connected mode. Each layer contains 64 Query Heads (each with a dimension of 64) paired with 8 Key-Value Heads. Additionally, every attention head has a learnable bias. It uses RoPE (Rotary Position Embedding) and leverages YaRN to extend the dense context window up to 131,072 tokens. The core purpose of this is to ensure numerical stability and aggregate information. In standard attention mechanisms, the softmax function mathematically forces the total sum of attention weights across a sequence to equal exactly 1. The problem arises when the model decides it does not want to reference any of the tokens in the current sequence. Because the weights must still add up to 1, a conventional setup forces the model to arbitrarily distribute that remaining weight across random tokens, which introduces unwanted noise into the data. By adding a specific learnable scalar value (the "sink") to each Query Head, the model is given a safe place to "dump" its attention. When the model wants to ignore all the actual tokens, it simply shifts the bulk of the attention weight onto this sink. This naturally compresses the attention weights on the real tokens to near 0, completely avoiding the noise issue. Beyond just absorbing unwanted attention, the sink serves a secondary role for information aggregation. It acts as a "global pooling" mechanism, gathering and summarizing the broader contextual information from the sequence before passing it efficiently down to the downstream layers.

In GPT-OSS, each MoE module contains a fixed number of experts and uses a standard linear routing projection layer to score the activations. For every single token, the router selects the top 4 scoring experts. It then applies softmax weighting exclusively to the outputs of those 4 selected experts. The routing mechanism decides which "expert" handles which token. In GPT-OSS, this is handled simply but effectively at the token level using a linear layer with a bias. The experts themselves are built using SiGLU (Sigmoid-gated Linear Unit) networks. To balance VRAM consumption, the calculation process changes depending on what the model is doing. To save memory, the model processes tokens sequentially, expert by expert. It iterates through the experts and accumulates the activations for the tokens that hit each one. When generating text (where speed is critical and memory pressure is slightly different), the model copies the inputs $E$ times (where $E$ is the number of experts) to compute everything in parallel, trading VRAM for speed. A common problem in MoE models is that the router might start favoring one or two experts for everything, leaving the others unused. To prevent this, GPT-OSS uses an Auxiliary Loss function, heavily inspired by the Switch Transformer design. The total loss is calculated as:

$$Loss = KL\_Loss + \alpha \cdot Aux\_Loss$$

The auxiliary loss ($\mathcal{L}_{\text{load}}$) is defined as:

$$\mathcal{L}_{\text{load}} = \alpha N \sum_{i=1}^{N} f_i P_i$$

- $f_i$: The actual proportion of tokens in the batch that were routed to expert $i$.
- $P_i$: The average probability assigned by the router to expert $i$ across the whole batch.

This mathematical setup acts as a penalty. The auxiliary loss hits its absolute minimum when both $f_i$ (the actual token distribution) and $P_i$ (the predicted probability) are perfectly uniform (meaning ideally, every expert gets $1/N$ of the traffic). By adding this to the total loss, the training process is mathematically forced to distribute tokens evenly across all experts, ensuring no single expert becomes overloaded while others sit idle.

GPT-OSS uses an open-source BPE tokenizer called o200k_harmony (available in the TikToken library). It expands upon the o200k tokenizer used in GPT-4o and OpenAI o4-mini by adding special tokens dedicated to the "harmony" chat format, resulting in a total vocabulary size of 201,088.
### 2.4.2 LLaMA
#### 2.4.2.1 LLaMA-1
LLaMA's architecture fundamentally follows the Decoder-only structure used by the GPT series. However, it incorporates three major architectural improvements borrowed from other advanced models (GPT-3, PaLM, and GPTNeo) to boost training stability, performance, and computational speed.

Instead of normalizing the output of the attention layers, LLaMA borrows a technique from GPT-3 and applies normalization to the input of each Transformer sub-layer (Pre-normalization) to improve training stability. Furthermore, it replaces standard LayerNorm with RMSNorm (Root Mean Square Layer Normalization). By skipping the mean calculation, the RMSNorm math is simpler. This saves between 7% and 64% of the calculation workload for that step, resulting in an overall speedup of about 40% without losing training stability. 

While Layer Norm helps prevent vanishing or exploding gradients, its reliance on calculating the mean makes it susceptible to noise and input shifts, which can sometimes lead to unstable gradient propagation. By omitting the mean and relying only on RMS, LLaMa-1 provides a much smoother gradient flow. This minimizes the negative impact that mean fluctuations can have on gradients, ultimately enhancing both the stability and speed of the training process, particularly in deeper network layers.

LLaMA overhauls the traditional Feed-Forward Network (FFN) and ReLU activation, borrowing the SwiGLU structure from PaLM. LLaMA explicitly uses no bias terms in its FFN. The final mathematical representation for LLaMA's FFN layer is:

$$FFN_{SwiGLU}(x, W, V, W_2) = (\text{SiLU}(xW) \otimes xV)W_2$$

Because calculating two matrices ($W$ and $V$) would normally increase the number of parameters, LLaMA reduces the hidden unit dimension size to $\frac{8}{3}d$ (instead of the $4d$ used in PaLM) to strictly maintain the same overall parameter count as a standard FFN.

Finally, LLaMA changes how the model understands the order of words. Borrowing from GPTNeo, LLaMA completely removes the standard absolute positional embeddings that are usually added to the initial input. Instead, it injects RoPE (Rotary Positional Embeddings) at every single layer of the network, which helps the model better grasp the relative distance between tokens as the data passes deeper into the network.

The model was trained on a massive, carefully filtered dataset drawn from seven distinct sources. A key detail is that while most of the data was only seen by the model once (1 epoch), data from Wikipedia and Books were looped through about twice (2 epochs).

- CommonCrawl (67.0%): The largest chunk of data, taken from five datasets between 2017 and 2020. It underwent heavy deduplication, language identification, and quality filtering.
- C4 (15.0%): A public dataset that was processed using heuristic rules for quality control.
- GitHub (4.5%): Code from specific open-source licenses, filtered to remove boilerplate and low-quality files, deduplicated at the file level.
- Wikipedia (4.5%): Mid-2022 data across 20 languages, with formatting stripped out.
- Books (4.5%): A deduplicated combination of the Gutenberg project and the Books3 dataset.
- ArXiv (2.5%): Scientific papers processed from LaTeX files to remove unnecessary sections and improve text consistency.
- Stack Exchange (2.0%): Q&A data spanning multiple domains, with answers cleaned and sorted.

LLaMA-1 was trained purely through self-supervised learning, meaning it was not fine-tuned for any specific tasks or instructions during this phase. It used the AdamW optimizer, specifically tuning the $\beta_1$ and $\beta_2$ parameters to ensure stable convergence. It utilized a cosine learning rate schedule, which gradually reduces the learning rate to smoothly guide the model to convergence. It also used a "warmup" step to stabilize the early, volatile stages of training. To prevent overfitting and ensure numerical stability, the team applied a weight decay of 0.1 and a gradient clipping of 1.0. Learning rates and batch sizes were dynamically adjusted based on the specific size of the model being trained.

Training a model with up to 65 Billion parameters requires massive compute, so Meta implemented several engineering tricks to save memory and time. They used a highly optimized version of the causal multi-head attention mechanism to reduce memory usage and compute time. Instead of relying entirely on standard automatic differentiation systems, they manually implemented the backpropagation functions to be more efficient. They also used Activation Checkpointing to avoid storing expensive activation computations in memory, recalculating them only when necessary. This drastically reduces resource consumption. Furthermore, they heavily utilized model and sequence parallelism, alongside optimized GPU-to-GPU communication, to speed up the overall training process.

#### 2.4.2.2 LLaMA-2

LLaMA-2 scales up its fundamental training parameters to provide a more robust baseline. The training corpus was expanded to 2.0 Trillion tokens (roughly a 40% increase over LLaMA-1's 1.4T tokens), with a heavy focus on removing non-compliant or low-quality data. Meta applied legal and privacy filters to deliberately exclude data from websites known to contain large amounts of personal information. Crucially, they did not heavily filter the remaining data. They wanted to avoid "demographic erasure" (accidentally removing minority voices by over-scrubbing the data), ensuring the model remained broadly capable. The pre-training data is predominantly English. Because non-English data was limited, the model's proficiency in other languages is relatively weak and should be used with caution.

To turn the base model into the conversational LLaMA-2-Chat, the team applied Supervised Fine-Tuning. They created 27,540 highly curated "prompt-response" pairs. They found that using this smaller set of exceptionally high-quality data yielded better results than using massive, lower-quality third-party datasets. During this training phase, the loss calculation for the user's input prompt tokens was "zeroed out." This forces the model to ignore the prompt when calculating its error rate and focus 100% of its learning capacity on generating the best possible response. After SFT, the model underwent extensive alignment to ensure it was helpful and safe. They used over 1.4 million Meta-created examples, combined with 7 other datasets (like Anthropic Helpful/Harmless, StackExchange, etc., totaling nearly 3 million comparisons). The Meta examples featured deep multi-turn conversations (averaging 3.9 turns per dialogue), while the other datasets were mostly single-turn. The RLHF process wasn't a single step. It iteratively improved the model using two specific techniques:

- Rejection Sampling: Generating multiple responses and selecting the best one based on a reward model to update the main model.
- PPO (Proximal Policy Optimization): Further fine-tuning the model's policy to maximize rewards.

While the model was training, the human preference data was constantly being updated in parallel to ensure the Reward Models guiding the RLHF process stayed accurate and up-to-date. The model can now process and remember twice as much text in a single prompt, expanding the context window from 2048 to 4096 tokens. Unlike LLaMA-1's strict research-only restriction, LLaMA-2 introduced a base model and a chat-optimized model that are available for conditional commercial use.

For the larger 34B and 70B parameter models, LLaMA-2 replaces standard Multi-Head Attention (MHA) with Grouped-Query Attention (GQA) to drastically improve inference speed. LLaMA-2 uses GQA-8. It divides the queries into groups, and each group shares one KV pair (using 8 KV projections total). This offers the best of both worlds: inference speeds that rival MQA, while maintaining output quality that matches MHA.

Because LLaMA-2 generates text autoregressively (meaning it predicts token $y_3$ based on tokens $y_1$ and $y_2$), it uses a KV Cache optimization. Instead of recalculating the Attention Keys and Values for the entire sequence every time it wants to generate a new word, it caches the previously calculated KVs in memory. This is a classic "trade space for time" maneuver, vastly accelerating the generation process at the cost of consuming more VRAM. (This makes the transition to the more memory-efficient GQA mentioned above especially important).

To further boost performance, the matrix dimensions within the Feed-Forward Network (FFN) modules were expanded. While this increases the overall parameter count of the model, it directly enhances the model's ability to generalize information and handle complex tasks.
#### 2.4.2.3 LLaMA-3
LLaMA-3 maintains the same fundamental Decoder-only architecture as LLaMA-2 but introduces massive upgrades in data scale, tokenizer efficiency, and post-training pipelines. 

While the core architecture (RMSNorm, SwiGLU, RoPE) remains mostly unchanged, Meta made a few highly impactful adjustments. In LLaMA-2, Grouped-Query Attention (GQA) was only used for the larger 34B and 70B models. In LLaMA-3, GQA is used across all model sizes (including the smaller 8B model) to significantly boost inference speed. LLaMA-3 switches from the SentencePiece tokenizer to tiktoken. The vocabulary size was quadrupled from 32K to 128K. This allows the model to encode text much more efficiently, yielding better downstream performance (though it does increase the parameter count slightly in the embedding layers). The standard sequence length was doubled, moving from LLaMA-2's 4,096 tokens to 8,192 tokens (8k).

The most significant driver of LLaMA-3's performance leap is its training data. LLaMA-3 was trained on over 15 Trillion tokens (compared to LLaMA-2's 2 Trillion). The dataset includes 4 times more code data (drastically improving its coding and logic capabilities) and incorporates high-quality data from over 30 non-English languages to lay the groundwork for multilingual support. To ensure this massive dataset was high quality, Meta built advanced filtering pipelines. Interestingly, they found previous LLaMA models were great at identifying good text, so they actually used LLaMA-2 as a text quality classifier to filter the training data for LLaMA-3. 

During pre-training, the team made a fascinating discovery regarding AI scaling laws (specifically the Chinchilla scaling law). The Chinchilla law previously suggested that for an 8 Billion parameter model, the optimal amount of training data is roughly 200 Billion tokens. However, Meta found that even after pushing past that limit by two orders of magnitude (training on up to 15 Trillion tokens), the performance of both the 8B and 70B models continued to increase in a log-linear fashion. 

LLaMA-3's post-training process (which turns the base model into the Chat/Instruct version) is far more sophisticated than LLaMA-2's. A Multi-Method Approach: Instead of just standard RLHF, LLaMA-3 uses a complex combination of Supervised Fine-Tuning (SFT), Rejection Sampling, PPO (Proximal Policy Optimization), and a new addition: DPO (Direct Preference Optimization). The biggest improvements in model quality didn't just come from the algorithms, but from carefully curating the prompts for SFT and running rigorous, multi-round quality assurance on the human-annotated preference rankings.

LLaMA-3 greatly improved its alignment, significantly reducing the "false refusal" rate where LLaMA-2 would stubbornly refuse to answer completely safe prompts. It introduced a suite of new trust and safety tools, including Llama Guard 2, Code Shield, and CyberSec Eval 2.
#### 2.4.2.4 Alpaca

The Stanford Alpaca model was created using a highly cost-effective, three-step fine-tuning process. The authors managed to train a model that performs similarly to OpenAI's text-davinci-003 for less than $600 total ($500 for generating data and $100 for computing power).

The process began with humans manually writing 175 "self-instruct" seed tasks. Each of these tasks was formatted into a specific JSON structure containing three main components:

- Instruction: The actual command or task description (e.g., "Find the four smallest perfect numbers.").
- Input (Optional): Additional context needed to complete the task. For example, if the instruction was "summarize this article," the input would be the text of the article itself.
- Output: The correct answer or expected response.

Instead of paying humans to write thousands of more examples, the team used OpenAI's powerful text-davinci-003 model to automate the data creation. They fed the 175 seed tasks into text-davinci-003 using specific instruction templates (one template for tasks with an input field, and one for tasks without). They provided strict generation rules, asking the model to create new, similar instruction data while ensuring diversity, adhering to word counts and language requirements, and including examples of refusing inappropriate instructions. Through this "Modified Self-instruct" generation process, they expanded their 175 seeds into a massive dataset of 52,000 instruction-following examples.

Finally, they took the base Meta LLaMA 7B model (which had only been pre-trained to predict the next word) and applied Supervised Fine-Tuning using the newly generated 52K instruction dataset. By training LLaMA 7B on how text-davinci-003 responds to various prompts, the resulting model—Alpaca 7B—learned to follow instructions and converse in a highly capable manner. 
#### 2.4.2.5 Code LLaMA
Code Llama does not start from scratch. All versions are initialized using the pre-trained Llama 2 foundation models (specifically the 7B, 13B, and 34B parameter sizes). The Llama 2 base is fed a massive dataset of 500 Billion tokens of general code data. After the initial 500B code training, the pipeline splits to create three highly specialized versions of Code Llama:

- Code Llama - Python: The model takes a left turn and is trained on an additional 100 Billion tokens made exclusively of Python code. This creates a hyper-specialized Python expert.
- Code Llama (Base): The model undergoes Long Context Fine-Tuning using 20 Billion tokens. This expands its memory window, allowing it to read, process, and generate much larger files and codebases at once.
- Code Llama - Instruct: This version takes the Long Context model and applies Instruction Fine-Tuning using 5 Billion tokens. This trains the model to act like a chatbot, ensuring it can understand human conversational prompts and execute specific commands.

Applied only to the 7B and 13B models during the initial training phase, this task teaches the AI how to "fill in the blanks" within an existing file. The system takes a complete piece of code, hides a specific chunk of it, and replaces it with a <MASK> symbol. Using an autoregressive method, the model analyzes the surrounding context (the code both before and after the mask) and is forced to predict exactly what the missing code should be.

### 2.4.3 DeepSeek
#### 2.4.3.1 DeepSeek-V1
The DeepSeek LLM architecture heavily borrows from the successful LLaMA blueprint but introduces a distinct strategy for how it scales up its larger models. At the fundamental level, DeepSeek uses the same proven components as LLaMA to ensure stability and performance:
- Normalization: It uses a Pre-Norm structure equipped with the RMSNorm function.
- Activation & FFN: It uses SwiGLU as the activation function for its Feed-Forward Networks (FFN). The intermediate dimension of these FFN layers is strictly set to $\frac{8}{3}d_{model}$.
- Positional Encoding: It uses RoPE (Rotary Positional Embeddings) to understand token positions.
- Attention Mechanism: While the smaller models use standard Multi-Head Attention (MHA), the massive 67B model uses Grouped-Query Attention (GQA) to optimize inference costs and reduce memory bottlenecks.

The most unique aspect of DeepSeek's architecture is how it scales its 67B model. When most developers build larger models, they usually widen the intermediate layers of the FFN. DeepSeek took a different approach: they expanded the parameter count by increasing the network's depth (adding more layers) rather than its width. They found this specific structural choice yielded better overall performance.

Adjusting the layers this way achieves a couple of goals. First, it keeps their overall parameter counts perfectly aligned with other standard open-source models (like LLaMA's 7B and 65B/70B). Second, having these specific layer counts makes it much easier to slice the model up for pipeline partitioning (distributing the model efficiently across multiple GPUs). This carefully balanced design guarantees high computational efficiency while giving the model the flexibility and expressive power needed to handle highly complex tasks.

DeepSeek's pretraining process is heavily focused on two main pillars: a rigorous three-step data processing pipeline and a highly optimized, custom training infrastructure. DeepSeek's pretraining process is heavily focused on two main pillars: a rigorous three-step data processing pipeline and a highly optimized, custom training infrastructure.

1. Aggressive Deduplication: Instead of just removing duplicate files within small, localized batches, DeepSeek runs deduplication across the entire massive Common Crawl dataset. They found this global approach removes about four times as many duplicate documents, achieving an almost 90% deduplication rate across 91 data dumps.
2. Quality Filtering: The team applies robust evaluation standards, combining both linguistic and semantic analysis to view the data quality from both a microscopic and macroscopic level.
3. Remixing for Balance: To ensure the model doesn't become biased toward the most common topics, they intentionally adjust the data mix to artificially boost the presence of underrepresented domains, ensuring a more inclusive baseline.

DeepSeek-V1 use a Byte-Level BPE (BBPE) tokenizer with a base vocabulary of 100,000 words. It is designed with special rules to prevent merging different character types (like splitting numbers into individual digits). They specifically padded the final vocabulary size to 102,400 to maximize computational efficiency on GPUs during training.

To train massive models efficiently, DeepSeek uses their own lightweight, high-efficiency training framework called HAI-LLM. The framework seamlessly integrates Data Parallelism, Tensor Parallelism, Sequence Parallelism, and 1F1B Pipeline Parallelism. It also uses ZeRO-1 to partition optimizer states and reduce communication overhead. To eliminate "waiting" time, the system overlaps calculations with data transfers. For example, while the GPUs are crunching matrix math (GEMM), the system is simultaneously gathering or scattering data across the network. They also fuse multiple operations (like LayerNorm, GEMM, and Adam updates) together to speed up processing and utilize Flash Attention to maximize hardware utilization.

The model is trained in BF16 precision to save memory, but critical gradient calculations are done in FP32 to ensure training stability. They also invented an "in-place cross-entropy" trick that calculates losses directly in the CUDA kernel—saving precious GPU memory (HBM) by instantly overwriting old data rather than storing it. The system asynchronously saves the model's progress every 5 minutes. If the hardware crashes, they lose a maximum of 5 minutes of training. Furthermore, the system is flexible enough to resume training even if the physical GPU cluster configuration changes.

DeepSeek v1’s post-training is divided into two main stages: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). They assembled a highly curated dataset of 1.5 Million bilingual (English and Chinese) instruction instances, including helpful data (1.2M instances broken down into Math problems (46.6%), General language tasks (31.2%), and Coding exercises (22.2%)) and safety data (300K instances designed to cover various sensitive topics to ensure harmless responses).

During the SFT phase, the author discovered that the two model sizes required very different training treatments to prevent breaking. The smaller 7B model was fine-tuned for 4 epochs (Learning Rate: 1e-5). However, the massive 67B model suffered from severe overfitting, so they restricted its fine-tuning to only 2 epochs (Learning Rate: 5e-6). The team noticed a strange bug: as they added more Math SFT data, the model started generating infinite, repetitive text loops. They concluded this happened because weaker models struggled to fully grasp complex mathematical reasoning patterns, causing them to get stuck. To cure this repetition loop without losing mathematical capability, they relied heavily on the next step (DPO).

In the DPO phase, the team created pairs of "good" and "bad" responses for both helpfulness (creative writing, Q&A, instruction following) and harmlessness. They used their own DeepSeek Chat model to generate these candidate responses. The DPO training ran for exactly 1 epoch with a batch size of 512 and a learning rate of 5e-6. They also used a learning rate warmup combined with a cosine scheduler. DPO successfully cured the text-repetition issue from the SFT phase and significantly enhanced the model's open-ended generation capabilities, all while maintaining high scores on standard benchmarks.

In older scaling laws (like DeepMind's famous Chinchilla paper), model size was simply represented by its number of parameters ($N$). The formula for total compute budget ($C$) was roughly:

$$C \approx 6ND$$

Where $N$ is parameters, and $D$ is the number of training tokens.

DeepSeek realized this approximation was actually causing massive statistical errors (sometimes up to 50% discrepancy in smaller models). There are two reasons.

- It ignores Attention costs: The $6N$ formula completely ignores the heavy computational overhead required to calculate the attention mechanism across long sequences of text.
- It overvalues Vocabulary: It includes the calculations for the embedding/vocabulary layer. While vocabulary matrices have a ton of parameters, they don't actually contribute much to the model's "thinking" or reasoning capacity.

To fix this, DeepSeek threw out parameter counting ($N$) and introduced $M$, which stands for Non-Embedding FLOPs per token. It calculates the exact amount of computational work done per token, explicitly ignoring vocabulary and explicitly including attention costs. The new formula for model scale became:

$$M = 72 n_{layer} d_{model}^2 + 12 n_{layer} d_{model} l_{seq}$$

Notice how vocabulary size is completely gone, but sequence length ($l_{seq}$) is now a core part of the math. This allowed them to rewrite the compute budget formula much more accurately as:

$$C = MD$$

With this highly accurate math, DeepSeek ran massive "IsoFLOP" experiments. They tested varying combinations of model sizes ($M$) and data sizes ($D$) under fixed compute budgets ($C$) ranging from $1\text{e}17$ to $3\text{e}20$ FLOPs to find the absolute minimum error rate. They found that the optimal model scale ($M_{opt}$) and optimal data scale ($D_{opt}$) grow exponentially based on the compute budget ($C$), represented as:

$$M_{opt} \propto C^a$$

$$D_{opt} \propto C^b$$

The Chinchilla paper previously suggested that as your compute budget grows, you should split it roughly 50/50 between making the model bigger and adding more data ($a \approx 0.49$, $b \approx 0.51$). However, DeepSeek found that the higher the quality of your training data, the more these coefficients change. Using their highly refined "Current Data", their coefficients were:

- $a = 0.524$ (Model Scaling factor)
- $b = 0.476$ (Data Scaling factor)

Because $a$ is significantly larger than $b$, DeepSeek proved that when you have highly logical, high-quality data, any extra compute budget should be spent more aggressively on making the model larger, rather than just forcing it to read more tokens.

The DeepSeek team successfully trained and released a highly capable open-source model built on a massive 2 Trillion token dataset, focusing primarily on English and Chinese. While the model made significant breakthroughs in training efficiency and scaling laws, it still suffered from a few notable disadvantages in everyday use. Because the model relies on a static dataset with a strict cutoff date, it cannot access real-time information or answer questions about recent events. Like most LLMs of its generation, DeepSeek Chat is still prone to confidently inventing facts or generating non-factual information. Besides, because they hyper-focused their 2 Trillion training tokens almost exclusively on English and Chinese, the model's proficiency in other languages is significantly underdeveloped.

#### 2.4.3.2 DeepSeek-V2

DeepSeek-V2 transitions to a Mixture of Experts (MoE) architecture. It has a massive total parameter count of 236 Billion (236B). However, because it's an MoE model, it doesn't use all of them at once. For any given token, it only activates 21 Billion (21B) parameters, keeping computational costs manageable. It supports a massive context length of 128K tokens.

To solve the memory bottleneck caused by the KV Cache during inference (a common issue in standard Multi-Head Attention), DeepSeek-V2 introduces a brand new architecture called MLA (Multi-head Latent Attention). MLA compresses the massive KV cache down into a single "latent vector." This ensures highly efficient inference by drastically reducing the memory footprint required to remember previous tokens.The combination of MoE and MLA results in staggering efficiency improvements over their previous V1 dense model (DeepSeek 67B).

<p align="center">
  <img width="447" height="365" alt="24d42ced-7245-4867-abcd-7439973af550" src="https://github.com/user-attachments/assets/c36f79aa-404e-4349-aee6-d1d895565a0c" />
</p>

DeepSeekMoE overhauls the standard MoE layout using two primary strategies.

1. Fine-Grained Expert Segmentation: Instead of using a few massive experts, DeepSeek splits the experts into much smaller, more granular units. This keeps the total parameter count the same but drastically increases the number of experts available. This allows each individual expert to become highly specialized in a very narrow knowledge domain. This fixes "Knowledge Mixing": In traditional MoE, a single expert is forced to learn multiple distinct types of knowledge. Finer segmentation allows each expert to specialize deeply in one specific domain.
2. Shared Expert Isolation: It designates a specific set of experts as "Shared Experts." These are always activated for every single token. Their job is to capture and process broad, common contextual knowledge. By offloading general knowledge to these shared experts, the other routed experts are free to focus exclusively on highly unique, specialized tasks without redundant overlap. This fixes "Knowledge Redundancy": In traditional MoE, different experts often need the same foundational knowledge, leading to wasted parameters as multiple experts memorize the same things. Shared experts handle this common knowledge, freeing routed experts to focus on unique domains.

In DeepSeek-V2, this translates to 2 Shared Experts and 160 Routed Experts per layer. Each token activates the 2 Shared Experts and the top 6 scoring Routed Experts. The MoE architecture replaces the standard FFN layers in all layers except the first layer. The intermediate hidden dimension for each expert is $1536$. 

Because low-rank compression (in MLA) and fine-grained expert segmentation (in MoE) alter the mathematical scale of a layer's output, DeepSeek applies two fixes to guarantee training stability. First, they add an extra RMSNorm layer right after the compression latent vector. They also multiply the data by an extra scaling factor at the "width bottlenecks" (specifically, at the intermediate hidden states of both the compression latent vector and the routed experts).

Because DeepSeek-V2 has so many fine-grained experts, they are physically distributed across multiple GPUs (devices). Sending tokens back and forth across a massive GPU cluster creates extreme communication bottlenecks. To solve this, DeepSeek uses Device-Limited Routing. Before selecting the absolute top-K experts for a token, the router first restricts the token to a maximum of $M$ devices (usually $M \ge 3$). It selects the top-K experts only from those chosen devices. This drastically reduces network communication costs while maintaining top-tier performance.

If the router heavily favors a few experts and ignores the rest, the network suffers from "routing collapse" and compute bottlenecks. DeepSeek implements three distinct mathematical penalties (Auxiliary Losses) during training to force the model to balance the workload:
- Expert-Level Balance Loss ($\mathcal{L}_{ExpBal}$): Forces tokens to be distributed evenly across all 160 individual experts.

$$\mathcal{L}_{ExpBal} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i$$

Where:

$$f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i)$$

This calculates the actual fraction of tokens routed to expert $i$.

$$P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}$$

This calculates the average routing probability/affinity score assigned to expert $i$ across all tokens.

Variables:

$\alpha_1$: Hyperparameter for the expert-level balance factor.

$N_r$: Total number of routed experts.

$K_r$: Number of activated routed experts per token.

$T$: Total number of tokens in the sequence.

$\mathbb{1}(\cdot)$: Indicator function (equals 1 if true, 0 if false).

$s_{i,t}$: The affinity score between token $t$ and expert $i$.

- Device-Level Balance Loss ($\mathcal{L}_{DevBal}$): Ensures that the computational workload is distributed evenly across the physical GPUs, preventing any single machine from overheating or bottlenecking the cluster.

$$\mathcal{L}_{DevBal} = \alpha_2 \sum_{i=1}^{D} f'_i P'_i$$

Where:

$$f'_i = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j$$

The average actual token fraction sent to the experts residing on device $i$.

$$P'_i = \sum_{j \in \mathcal{E}_i} P_j$$

The sum of the routing probabilities for all experts residing on device $i$.

Variables:

$\alpha_2$: Hyperparameter for the device-level balance factor.

$D$: Total number of devices.

$\mathcal{E}_i$: The set of experts deployed on device $i$.

- Communication Balance Loss ($\mathcal{L}_{CommBal}$): Ensures that the actual network traffic (the sending and receiving of tokens between GPUs) remains balanced, preventing network traffic jams.

$$\mathcal{L}_{CommBal} = \alpha_3 \sum_{i=1}^{D} f''_i P''_i$$

Where:

$$f''_i = \frac{D}{M T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ is sent to Device } i)$$

Calculates the actual fraction of communication traffic directed to device $i$.

$$P''_i = \sum_{j \in \mathcal{E}_i} P_j$$

The same probability sum used in the device-level loss.

Variables:

$\alpha_3$: Hyperparameter for the communication balance factor.

$M$: The maximum number of devices a token is allowed to be sent to under the Device-Limited Routing mechanism.

Because auxiliary loss relies on probabilities, it cannot guarantee a perfectly balanced workload. To strictly enforce compute limits and prevent wasted resources, DeepSeek uses a Device-Level Token-Dropping strategy during training. The system calculates an exact compute budget for each physical device. If a device receives too many tokens, it simply drops the tokens with the lowest "affinity scores" until it fits the budget. (To ensure training stability, the system guarantees that roughly 10% of the training sequence will never be dropped).

DeepSeek-V2 used GRPO as its RL post-training method. This method will be introduced in the following chapters. To make Reinforcement Learning effective, DeepSeek needed a highly reliable "Reward Model." They built this by carefully curating and filtering preference data. For the code preferences, they gathered using direct feedback from compilers. For the math preferences, they gathered using actual ground-truth labels (verifiable right/wrong answers). They initialized this Reward Model using the DeepSeek-V2 Chat (SFT) model and trained it using point-wise or pair-wise loss. This RL training phase is crucial because it fully unlocks the model's potential, teaching it how to consistently choose the most accurate and satisfactory response out of all possible answers.

Running RL on a model of this massive scale puts immense pressure on GPU memory and RAM. To keep training speeds fast without crashing the system, the team implemented three major engineering optimizations:
- Hybrid Engine: They developed a system that uses completely different parallel-processing strategies for the training phase versus the inference phase, drastically improving overall GPU utilization.
- vLLM Integration: They used vLLM with large batch sizes as their inference backend to significantly speed up processing.
- Smart CPU Offloading: They designed a meticulous scheduling strategy that constantly offloads parts of the model to the CPU and loads them back onto the GPU as needed. This achieved a near-perfect balance between saving memory and maintaining training speed.

#### 2.4.3.3 DeepSeek-V2.5
DeepSeek-V2.5 is not built from scratch; it is the ultimate merger of two specialized models: the conversational DeepSeek-V2-Chat and the programming-focused DeepSeek-Coder-V2. By combining them, V2.5 retains the general conversational fluency of the Chat model while inheriting the heavy-duty programming capabilities of the Coder model.

DeepSeek took the original DeepSeek-V2-Chat and ripped out its base model, replacing it with the DeepSeek-Coder-V2-Base. This cross-pollination massively boosted the Chat model's coding and reasoning skills, resulting in the DeepSeek-V2-0628 version. Meanwhile, the DeepSeek-Coder-V2 underwent alignment optimization to improve its general, everyday capabilities, resulting in the DeepSeek-Coder-V2-0724 version. These two highly optimized branches (0628 and 0724) were fused together to create the final unified DeepSeek-V2.5.

DeepSeek-V2.5 outperforms both of its direct predecessors (0628 and 0724) across four standard English and Chinese benchmarks. In internal arena testing (judged by GPT-4o), its win rate against models like GPT-4o mini and ChatGPT-4o-latest improved noticeably, particularly in creative writing and Q&A. It features much clearer boundaries for what constitutes a safety violation. While its defense against malicious "jailbreak" attacks was strengthened, it simultaneously reduced its "false refusal" rate (where safety protocols overly generalize and block completely normal, safe questions). It showed significant score jumps in Python-specific evaluations (HumanEval Python) and LiveCodeBench. Its Fill-In-the-Middle (FIM) autocomplete capabilities improved by 5.1%.

#### 2.4.3.4 DeepSeek-V3
#### 2.4.3.5 DeepSeek-V3.2-EXP
#### 2.4.3.6 DeepSeek-R1

### 2.4.4 Qwen
#### 2.4.4.1 Qwen1
#### 2.4.4.2 Qwen1.5
#### 2.4.4.3 Qwen2
#### 2.4.4.4 Qwen2.5
#### 2.4.4.5 Qwen3
#### 2.4.4.6 Qwen3 Next
#### 2.4.4.7 GTE-Qwen3
#### 2.4.4.8 Qwen3 Embedding & Reranker

### 2.4.5 Gemini
