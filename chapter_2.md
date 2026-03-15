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

$$I_{next} = \text{EMD\_Layer}(I, H)$$

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
#### 2.4.1.1 GPT-1
#### 2.4.1.2 GPT-2
#### 2.4.1.3 GPT-3
#### 2.4.1.4 GPT-3.5
#### 2.4.1.5 GPT-4
#### 2.4.1.6 OpenAI o1
#### 2.4.1.7 GPT-OSS

### 2.4.2 LLaMA
#### 2.4.2.1 LLaMA-1
#### 2.4.2.2 LLaMA-2
#### 2.4.2.3 LLaMA-3
#### 2.4.2.4 LLaMA-4
#### 2.4.2.5 Alpaca
#### 2.4.2.6 Code LLaMA

### 2.4.3 DeepSeek
#### 2.4.3.1 DeepSeek-V1
#### 2.4.3.2 DeepSeek-V2
#### 2.4.3.3 DeepSeek-V2.5
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
