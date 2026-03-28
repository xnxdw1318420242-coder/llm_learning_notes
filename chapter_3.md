# 3. Pretraining
Pretraining is the foundational stage of building a Large Language Model. It involves constructing a massive neural network and feeding it an enormous volume of data to learn from. At its core, pretraining is a form of Transfer Learning driven by Large-scale Self-Supervised Pretraining.

- The Traditional Approach: Historically, neural networks started with randomly initialized parameters and relied on optimization algorithms (like stochastic gradient descent) to adjust those parameters from scratch for every specific task.

- The Pretraining Approach: Instead of starting from scratch, the model is first trained on massive amounts of low-cost, unlabeled data to extract common patterns, language rules, and general world knowledge. The resulting parameters are then used as a highly educated "starting point" for subsequent training.

The core logic of modern LLMs relies on a two-step transfer process.

- Pretraining (Learning the Commonalities): The model learns broad, generalized language patterns and knowledge from a sea of unlabeled text.
  
$$\theta^* \leftarrow \arg \min_{\theta} \mathcal{L}_{\text{pre}}(\theta, \mathcal{D}_{\text{pretrain}})$$

- Downstream Transfer (Task Specialization): Because the model already possesses a vast understanding of language and facts, it only requires a small amount of expensive, labeled data for Fine-tuning, or even just a few contextual examples (Prompt/In-context learning) to complete specific downstream tasks.

$$\phi^* = \arg \min_{\phi} \sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{\text{down}}} G(\mathbf{x}, \mathbf{y}; \phi) \quad \text{with initial } \phi \approx \theta^*$$

Pretraining technologies are widely adopted across machine learning because they effectively solve four major challenges:

- Data Sparsity: High-quality labeled data is expensive and extremely difficult to acquire in large quantities. Pretraining utilizes virtually limitless unlabeled data to train the model, vastly improving its baseline performance and generalization abilities without needing massive labeled datasets.

- Prior Knowledge Injection: Many complex tasks (like NLP) require a deep understanding of prior knowledge, such as linguistic structures, grammar rules, and common sense. Pretraining forces the model to inherently learn this foundational knowledge from the unlabeled text before attempting specialized tasks.

- Transfer Learning Capabilities: Tasks often share underlying commonalities (e.g., semantic understanding is required for both text classification and translation). Pretraining allows the model to consolidate these shared commonalities so that its capabilities can be easily transferred from one task to another.

- Model Interpretability: Pretraining helps the model learn to represent abstract features effectively. For example, in NLP, it helps the model form deep, structured representations of words and phrases, which can ultimately improve the model's interpretability.

By starting from a base of generalized knowledge rather than a blank slate, pretraining successfully addresses data scarcity, injects crucial prior knowledge, and enables flexible task transfer—ultimately reducing task-specific training costs while dramatically boosting performance.
## 3.1 Data
### 3.1.1 Data Collection
Large Language Model (LLM) datasets are generally classified into four primary types:

- Pre-training Corpus: Massive volumes of unlabeled text used for foundational learning.

- Instruction Fine-tuning Dataset: Labeled data used to teach the model how to follow specific tasks.

- Preference Dataset: Data used to align model outputs with human values (RLHF).

- Evaluation Dataset: Benchmarks used to measure model performance.

General data provides the model with its fundamental language modeling capabilities and broad world knowledge.

- Web Data (Common Crawl, RefinedWeb). Massive and multi-lingual. Requires strict cleaning to remove noise and sensitive info.

- Language Text (BNC, US National Corpus). Enhances linguistic characteristics and specific domain knowledge (finance, law).

- Books (Project Gutenberg, Smashwords). Provides high-quality, long-context information for deep learning.

- Academic Materials (arXiv, S2ORC, PubMed Central). Professional text used to master STEM, Medicine, and Science.

- Code (GitHub, StackOverflow). High-quality programming repositories; essential for logical reasoning.

- Parallel Corpora (ParaCrawl, MultiUN). Sentence pairs used specifically to improve machine translation.

- Social Media (Reddit, Twitter/X). Captures real-time dynamics and interactivity, though requires filtering for toxic content.

- Encyclopedias (Wikipedia). High-authority, high-density knowledge used to build the model's fact-base.

- Hybrid Datasets (The Pile, Dolma). Pre-mixed, diverse datasets containing multiple types of the above.

Specialized data is integrated to boost performance in complex reasoning or niche tasks.

- Multilingual Text: Establishes semantic links between languages. It enhances cross-lingual understanding and increases data diversity, which improves overall model robustness.

- Scientific Text: Includes arXiv papers, textbooks, and scientific web pages. It significantly improves scientific Q&A and reasoning. Because these contain formulas and protein sequences, they require specialized tokenization and preprocessing to convert them into a uniform format the LLM can process.

- Code Data: Sourced from GitHub and Stack Exchange. Its highly structured nature improves the model's logical reasoning and structured semantic understanding. This can be used to enhance the model's tool-calling and learning abilities. Formatting reasoning tasks as code often yields more accurate results.

To train a "vertical" or industry-specific model, the following sources are targeted:

- Domain-Specific Text: Papers, reports, and books provide professional terminology.

- Domain Web Content: Scraped from targeted industry websites.

- Domain News: Keeps the model updated on the latest events and trends.

- Industry Reports & White Papers: Provides professional analysis and background for decision-making.

- Domain Social Media: Reflects hot topics, expert opinions, and user needs within a specific field.

- Domain Dialogue Data: Customer service logs and Q&A platforms help the model learn common problems and solutions.

When selecting and processing datasets, developers must follow three core pillars:

- Quality & Legality: Ensure data is high-quality, legally sourced, and follows ethical norms.

- Data Augmentation: Utilize techniques like data synthesis and data transformation to expand the scale and diversity of the training set.

- Engineering Balance: As seen in the provided pie charts, models (like Falcon, LLaMA, and GPT-3) vary their data mixtures (e.g., LLaMA 65B uses ~87% Web data) to balance general knowledge with specialized reasoning.

### 3.1.3 Data Preprocessing
The pre-training data preprocessing pipeline is a rigorous, multi-stage framework designed to filter out low-quality, redundant, irrelevant, and harmful data. 

#### 3.1.3.1 Data Quality Filtering

To maximize efficiency and accuracy, filtering combines fast heuristic rules with highly accurate model-based classifiers.

Heuristic-Based Filtering is highly efficient, capable of processing 10M to 100M level datasets rapidly. It operates on both document and sentence levels. In the document level, datasets often restrict content to specific languages (e.g., RefinedWeb and The Pile for English). For web data, the method usually discards documents with > 100 consecutive duplicate words/sentences, or where the punctuation ratio exceeds 0.1. For Wikipedia, it removes pages with fewer than 25 UTF-8 words. For HTML, it drops HTML tags that lack basic stop words like the, be, to, of, and, that, have, with. Forum data should disacard user comments if a thread has < 3 replies. Documents are also filtered based on a specific "quality filtering score", text density, special character ratio, short line count, and Perplexity (to weed out unnatural or AI-generated text). Sentence-Level Cleaning removes incomplete sentences, special symbols, HTML/CSS/JavaScript tags, brackets, and redundant web elements (like "Like" buttons or navigation menus). We use the Perplexity metric to evaluate naturalness, identify and drop artificially generated or unnatural text.

Model-Based Filtering (Classifiers) are used for fine-grained filtering, though they are computationally heavier. FastText is highly efficient for lightweight models, but precision is limited by the model's capacity. BERT can be fine-tuned for specific pre-training data but has limited generalization capabilities. GPT-4 (Closed-source APIs) is highly capable but comes with high costs and a lack of flexibility for custom tasks.

Here, we introduce two extra methods for data filtering.

IFD is a metric designed to let the model itself determine how "valuable" a piece of training data is. It measures the value of an instruction by comparing how the model performs on a specific answer with and without the instruction present. To calculate an IFD score, the system looks at two primary indicators:
- Conditioned Answer Score (CAS): This measures how well the model generates an answer when the instruction is given. It evaluates the consistency between the input prompt and the output.
- Direct Answer Score: This measures the model's ability to generate that same answer without the instruction. It represents the intrinsic complexity or "inherent difficulty" of the answer itself.
The IFD Score Formula:
$$r_{\theta}(Q, A) = \frac{\text{Conditioned Answer Score}}{\text{Direct Answer Score}}$$

- High IFD Score: This indicates that adding the instruction significantly clarifies the task or adds value that wasn't there before. This data is considered high-value "Cherry" data because it helps the model learn to follow instructions.
- Low IFD Score: This indicates the data is either too simple (the model already knew the answer without the prompt) or the instruction and answer are unrelated. This data is usually filtered out.

The 3-Step IFD Workflow:
1. Learning from Brief Experience: A base model is given a "short-term" training on a small, highly diverse set of samples (selected via K-Means clustering) to give it basic instruction-following capabilities.
2. Evaluating Based on Experience: This "pre-experienced" model calculates the CAS and Direct Answer scores for the entire dataset to determine the IFD for every sample.
3. Retraining from Self-Guided Experience: Only the high-value "Cherry" data is used to fine-tune the final model, ensuring every training step is used to "patch the model's weaknesses."

<p align="center">
<img width="519" height="385" alt="86272d6e-e32e-4cf6-8632-644295b09a05" src="https://github.com/user-attachments/assets/c537a75f-f0a3-4e2d-a781-47bc37cbbf4e" />

</p>
MoDS is a more holistic selection framework. It evaluates data across three specific dimensions: Quality, Diversity, and Necessity to select the most valuable subset for training. 

The 3-Step MoDS Workflow:
1. Quality Evaluation. The system uses a Reward Model (often based on the DeBERTa architecture) to score every (Instruction, Input, Output) triple in the raw dataset. A threshold is set, and any data that doesn't meet the quality standard is discarded immediately.

2. Diversity Selection. High-quality data is useless if it is all the same. MoDS uses the K-Center-Greedy algorithm to identify a representative set of instructions that covers a wide range of tasks and semantics. This creates the Seed Instruction Dataset.

3. Necessity Selection (Finding the Model's "Gaps"). This step identifies what the model specifically needs to learn. The model is fine-tuned using the Seed Dataset from Step 2 to create an initial baseline. This initial model is then asked to predict outcomes for the rest of the high-quality data. Any samples where the model performs poorly are identified as the model's "shortcomings" or "weaknesses." These difficult samples are used to create an Augmented Instruction Dataset.

By combining the Seed Dataset (for breadth) and the Augmented Dataset (for specific improvement), MoDS results in a final model with superior performance across a wide range of tasks.

<p align="center">
<img width="468" height="355" alt="e34358b6-b3af-45d8-83a4-d851cc756f7e" src="https://github.com/user-attachments/assets/daba9a43-23d2-4ff2-bc04-44c24cd5b711" />
</p>

#### 3.1.3.2 Sensitive Content Filtering

Filtering out toxic content and Personally Identifiable Information (PII) is mandatory to prevent models from generating abusive outputs or leaking private user data. 

- Toxicity: The pipeline uses classifiers trained on datasets like Jigsaw to precisely identify and filter out toxic, abusive, or biased content.

- Privacy (PII): Heuristic rules are heavily used here. For example, the Dolma dataset uses a strict rule for emails, phone numbers, and IP addresses. If a document has < 5 private items, they are replaced with placeholders like [EMAIL_ADDRESS]. If a document has >= 6 private items, the entire document is directly deleted.

#### 3.1.3.3 Data Deduplication

Early research in Large Language Models (LLMs) believed that increasing model parameters was the most important factor for success. However, current science proves that data quality is the true bottleneck. By elevating data quality, smaller models can match or even beat massive, parameter-heavy models. Conversely, using low-quality or highly repetitive data causes training to fail. Crucially, if a model is trained on factually incorrect or outdated data, it will confidently generate false information—a phenomenon formally defined as Hallucination.

When assembling massive datasets for LLM training, duplicates are inevitable. Failing to remove them leads to several severe consequences:

- Overfitting and Rote Memorization: If data contains massive amounts of repetitive content, the model will "rote memorize" these specific samples instead of learning how to genuinely understand and generalize language.

- Behavioral Degradation: Anthropic discovered that training on duplicate data actively harms a model's ability to utilize context and causes it to output repetitive, localized loops.

- Training Instability (Double Descent): Duplicates can cause a phenomenon known as Double Descent, where the training loss drops, unexpectedly spikes, and then drops again, leading to severe instability during the training process.

- Privacy and Security Risks: Models that over-memorize repeated data are highly susceptible to leaking sensitive or private information that was accidentally included multiple times in the training corpus.

- Resource Inefficiency: Processing redundant data wastes massive amounts of computational power and dramatically increases training time.

- Evaluation Contamination: If duplicate data exists across both the training set and the testing set, the model effectively gets to "cheat" on its final exam. This contaminates the evaluation, making the test results wildly inaccurate.

Because comparing every document character-by-character is computationally impossible at a "Trillion-token" scale, engineers use distinct categories of deduplication:

- Exact Matching. This method identifies text segments that are character-for-character identical. It typically utilizes cryptographic hash values (like MD5) to instantly flag identical files, or suffix arrays to find and match the longest common substrings that meet a minimum length requirement. This method is often applied to remove verbatim copies or highly repetitive boilerplate text.

- Approximate (Fuzzy) Matching. It is designed to find "near-duplicates"—documents that are slightly different (e.g., different ad placements, minor formatting changes) but contain the exact same core content. It heavily relies on Locality-Sensitive Hashing (LSH).
  1. MinHash: The primary tool for approximate matching. It treats a document as a set of features (like n-grams) and applies multiple random hash functions to every element. For each hash, it selects the minimum value to create a small "signature." By comparing these signatures instead of the full text, the system can rapidly estimate the Jaccard similarity between two massive documents, skipping pairwise comparisons.
  2. SimHash: Transforms text features into a fixed-length bit string (a fingerprint). Similarity is measured using Hamming distance; the smaller the distance between two fingerprints, the higher the similarity.
  3. TF-IDF: Another "soft" method that compares documents based on the frequency and rarity of specific words.

- Semantic Deduplication. This method moves beyond characters and words to understand actual meaning. It utilizes embedding models (like SentenceTransformer) to convert raw text into mathematical vectors. It then calculates the semantic similarity between these vectors. This method is applied identifying and removing documents that express the exact same concept or meaning, even if they use entirely different vocabulary to say it.
  
In practice, real-world data pipelines do not rely on just one method. They use a multi-stage, multi-granularity approach:

1. Determine the Processing Unit: First, define the basic unit for deduplication based on the data type (e.g., line-level, paragraph-level, or document-level).

2. Broad Filtering (Document Level): Use Approximate Matching (like MinHash LSH) across the entire dataset to quickly identify and remove documents with high similarity (e.g., similarity scores > 0.8).

3. Internal vs. Cross-Unit Deduplication: * Internal: Check if there is repetitive, looping content within a single processing unit (like a single document) and delete/merge it.

4. Cross-Unit: Compare different processing units against each other to find broader overlaps.

5. Fine Clean-up (Sentence Level): Use Exact Matching to perform a granular clean-up, deleting specific identical strings or boilerplate that might still exist within otherwise unique documents.

6. Train/Test Split Verification: Perform a final rigorous deduplication check specifically between the training set and the test set to guarantee zero contamination.

#### 3.1.3.4 tokenization
Tokenization for modern Large Language Models (LLMs) relies on Subword Tokenizers (such as BPE, WordPiece, or Unigram) and is executed through a highly structured four-step pipeline. Here is the step-by-step breakdown of how to do tokenization, including specific strategies used by top-tier models like Qwen, LLaMA, and DeepSeek.

1. Normalization. This is the initial cleanup phase of the raw text. Traditional methods include removing needless whitespace, lowercasing, removing accents, and applying Unicode normalization (like NFC or NFKC). While traditional NLP heavily relied on normalization, modern LLMs often skip this step entirely to ensure they can process and generate text exactly as it appears in the real world without losing formatting.

2. Pre-tokenization. Before applying the main algorithm, the text is divided into distinct chunks. A token is strictly not allowed to cross the boundaries of these chunks. By Character Category Boundaries, tokens generally cannot cross different character categories (except for spaces). By Number Splitting, all numbers are split into individual digits (e.g., 123 becomes 1, 2, 3). Models like LLaMA, Qwen, and DeepSeek do this to dramatically improve the model's ability to generalize mathematical operations and encode numeric data. For Byte Fallback, rare or unknown characters are decomposed into raw UTF-8 bytes instead of replacing them with a useless <UNK> (unknown) token.

3. Model & Vocabulary Construction. This is where the actual subword algorithm (like BPE) is applied. Developers generally take one of two paths. To build from scratch, they use libraries like SentencePiece to train a brand-new vocabulary directly on the pre-training corpus (used by LLaMA, DeepSeek). To expand an existing vocabulary, they use a fast tokenizer like tiktoken. For example, Qwen started with OpenAI's cl100k_base vocabulary and augmented it with thousands of commonly used Chinese characters and words. Expanding the vocabulary provides massive benefits. It requires fewer tokens to represent the same text. Also, because encoding is more efficient, the model can effectively fit much more text into its context window. When expanding, the new token embeddings must be initialized. You can either initialize them randomly or calculate the mean value of the original vocabulary's embeddings.

4. Post-Processing. The final step modifies the tokenized sequence to make it ready for the neural network. The post-processor injects necessary structural tokens, such as [CLS] (classification tokens), Padding tokens, or specific Mask tokens depending on the training objective.

### 3.1.3 Data Augmentation
To improve Large Language Models (LLMs), researchers use data augmentation, which is categorized into two main perspectives: the Data Perspective and the Learning Algorithm Perspective.

The Data Perspective involves using the LLM itself to manipulate or create new data. 
1. Data Creation: Leveraging the LLM's "Few-Shot" learning ability to rapidly generate synthetic training data. For example, in the medical field, synthesizing medical dialogue summaries and mixing them with a small amount of human-labeled data achieves results comparable to large-scale human annotation. It is also used to generate query-document pairs to improve logic and retrieval.

2. Data Annotation: Using the LLM to automatically label unannotated data. In certain tasks (like judging political leaning), an LLM's "Zero-Shot" annotation accuracy can actually surpass human annotators at a lower cost.

3. Data Restructuring: Rewriting or expanding existing data to create diverse variations (e.g., paraphrasing for NLP few-shot tasks, or counterfactual generation). This is also heavily used in Knowledge Distillation to transfer an LLM's capabilities to smaller models.

4. Collaborative Annotation: A human-machine hybrid approach. The LLM handles high-confidence samples automatically, while human annotators only step in to handle uncertain or ambiguous data, optimizing costs while maintaining quality.

The Learning Algorithm introduces the Teacher-Student Learning paradigm, where a powerful LLM (Teacher) generates data to train a smaller model (Student). In Generative Learning, the LLM creates rich training data, including Supervised Instruction data (creating Instruction/Input/Output formats), In-Context Learning data (Researchers use LLMs to generate complex dialogue datasets, Knowledge-Graph augmented context samples, and time-series Q&A data. This significantly boosts the student model's Natural Language Generation (NLG) and contextual understanding in reasoning and chat tasks), and Alignment data (generating preference pairs. It also explores using multiple LLMs to simulate group preference optimization and constructing contrastive feedback data to optimize how the model aligns with human expectations). In Discriminative Learning, the LLM acts as a scorer or evaluator. In Classification Task Augmentation, LLMs show massive strength in generating training data for classification tasks, often surpassing traditional techniques in "few-shot" scenarios. This includes building specialized teacher models for data generation and designing optimized schemes for specific tasks like emotion analysis and intent recognition. In Regression Scoring Application, The LLM acts as an intelligent scoring system. Crucially, researchers are using LLMs as Reward Functions in Reinforcement Learning (RL), or using them to generate dense reward signals to solve the notorious "sparse reward" problem in RL. In text evaluation, the LLM is used to score summary quality and evaluate controllable text generation.

### 3.1.3 Data Scheduling

Once data is collected and augmented, it must be scheduled. This involves two core decisions: the mixing ratio of the data sources, and the training order (Curriculum).

Setting the correct ratio of different data sources is critical because it dictates the model's final capabilities. The specific pretraining mixture used for the highly representative LLaMA model includes Web Data > 80%, Code-intensive Data (GitHub/StackExchange) 6.5%, Book Data 4.5%, and Science Data (arXiv) 2.5%. Even when training a highly specialized model (like a pure coding model), you must still include a certain percentage of general web data to preserve the model's common sense and general language knowledge. To find the perfect mix, researchers often train smaller proxy models (e.g., 1.3B parameters) from scratch using various data mixtures, find the best-performing ratio, and apply it to the massive target model.

1. Coding Ability (CodeLLaMA). To build a coding model from a general base model (LLaMA-2), the data was sequenced strictly. CodeLLaMA Base: 2T General Tokens $\rightarrow$ 500B Code-heavy Tokens. CodeLLaMA-Python: 2T General Tokens $\rightarrow$ 500B Code-heavy Tokens $\rightarrow$ 100B Python-specific Tokens.

2. Mathematical Ability (Llemma). Llemma used CodeLLaMA as its foundation, proving that coding logic is a great stepping stone for math logic. Training Sequence: 2T General Tokens $\rightarrow$ 500B Code-heavy Tokens $\rightarrow$ 50~200B Math-specific Tokens. During the final math phase, Llemma explicitly kept 5% general domain data in the mix. This acted as a regularization technique to prevent the model from "catastrophically forgetting" its general language abilities while focusing purely on math.

3. Long-Context Ability. To expand the context window (modifying the RoPE positional encoding), the data sequence gradually introduces longer text. CodeLLaMA Context Expansion: 4K context window (using 2.5T tokens) $\rightarrow$ 16K context window (using 20B tokens).

### 3.1.4 Sentence Length

In Natural Language Processing (NLP) and Speech Recognition, data samples (like sentences or audio clips) often have varying lengths. However, deep learning frameworks require fixed-shape tensors for batch processing. Managing this discrepancy is critical for training efficiency and model performance.

Deep learning models rely on matrix operations, which require fixed shapes. Variable-length data presents several hurdles:

- Batching Complexity: Frameworks like TensorFlow and PyTorch cannot directly form a batch from sequences of different sizes. This increases the complexity of preprocessing and debugging.

- Resource Waste: Padding short sequences to match longer ones leads to "invalid computations" and uneven memory usage, lowering hardware utilization.

- Efficiency Bottlenecks: Processing padding tokens wastes computational cycles. Furthermore, if not handled correctly, gradients can propagate through padded areas, leading to "meaningless updates" that hurt the final model quality.

There are five primary strategies for managing data of different lengths:

- Padding. Fill shorter sequences with zeros or special placeholder tokens to match the length of the longest sequence in the batch. Use a Mask matrix during loss calculation and gradient updates. This tells the model (e.g., the Attention mechanism in Transformers) to ignore the padded values. By this method it is easy to implement and ensures batch compatibility, but it wastes significant compute resources if the difference between the shortest and longest sequence is large.

- Truncation. Cut off sequences that exceed a predefined maximum length and discard the remainder. This is best for tasks where the "tail" of the data is less important, such as sentiment classification. It reduces memory and compute costs but causes information loss. It is unsuitable for tasks requiring global context (like long-document summarization).

- Bucketing. Group sequences of similar lengths into the same batch to minimize the amount of padding needed. Define length "buckets" (e.g., 5–10 tokens, 11–15 tokens) and assign data accordingly. This can be done statically or dynamically during training. This method significantly improves hardware utilization and reduces padding interference. However, it makes data loading more complex and can reduce the "randomness" of the training distribution.

- Dynamic Batching. Instead of padding every batch to a global maximum, adjust the padding length in real-time based on the longest sequence within that specific batch. This is supported by dynamic computation graphs in TensorFlow and PyTorch. This method minimizes padding waste and is highly adaptive. However, it has higher implementation complexity and may not be as "friendly" to certain low-level GPU parallel optimizations.

- Variable-Length Native Models. Utilize model architectures that naturally support variable inputs, such as RNNs or Transformers with masking. Use framework-specific functions like dynamic_rnn or mask-based attention weights. This preserves the flexibility of the data and avoids extra compute on padding. However, it is harder to implement and highly dependent on specific hardware support.

Beyond just "fixing" the length issue, several optimization techniques should be used to speed up training:

- Data & Hardware Optimization
 1. Pre-generate Masks: Generate your masks during the data loading phase to avoid redundant calculations during the actual training steps.

 2. Parallel Loading: Use multi-threaded or multi-process DataLoaders to speed up data reading.

 3. Hardware Acceleration: Use GPUs or TPUs for parallel matrix operations. Utilize Mixed Precision Training (FP16/AMP) to reduce memory usage and speed up processing.

- Hyperparameter & Model Tuning
 1. Batch Size Adjustments: Increase batch sizes within the limits of your hardware to improve parallelism. Use Gradient Accumulation to simulate the effects of a large batch size if memory is limited.

 2. Structural Simplification: Use lightweight Transformer variants or prune redundant weights to reduce complexity.

 3. Quantization: Convert model weights from floating-point to integer formats to save space and compute.

- Software & Framework Optimization
 1. Graph Optimization: Enable XLA (TensorFlow) or TorchScript (PyTorch) to optimize the underlying computation graph.

 2. Optimized Libraries: Use libraries like cuDNN to maximize GPU acceleration effects.

## 3.2 Training Tasks 
### 3.2.1 Goals
Before diving into the math, it is essential to understand the different frameworks we use to train a Large Language Model. Modern training tasks are broadly categorized into three areas:

1. Self-Supervised Learning. This is the foundational phase where the model learns directly from raw, unlabeled text. In the Masked Language Modeling (MLM) task, the system randomly masks (hides) certain words in a sentence, and the model must predict what was hidden. In the Autoregressive Language Modeling (ALM) task, the model predicts the sequence strictly from left to right, guessing the next word based on the previous words. In the Sequence-to-Sequence Modeling task, an encoder-decoder architecture that reads a full input sequence and generates a completely new output sequence. Highly suitable for generation tasks like translation.

2. Contrastive Learning. This technique forces the model to understand the nuances of similarity and difference. SimCSE constructs "positive" sample pairs using dropout techniques. The training objective is to maximize the mathematical similarity between positive samples while pushing negative samples apart. ELECTRA uses a two-part generator-discriminator setup. The generator replaces masked words with plausible alternatives, and the discriminator must guess whether a specific word is original or replaced.

3. Knowledge Enhancement. These techniques explicitly inject deeper semantic understanding and real-world facts into the model. ERNIE enhances semantic understanding by specifically introducing entity and phrase masking (rather than just random individual words). K-BERT directly embeds Knowledge Graphs into the model to massively boost its reasoning capabilities.

For modern generative AI (like the GPT family), Autoregressive Language Modeling (ALM) is the dominant paradigm. The training objective is simple: based on the provided context (previous words), maximize the probability of correctly generating the next target token. Mathematically, the loss function is defined as:

$$\mathcal{L}_{LM}(\theta) = \mathbb{E}_{x \in \mathcal{D}_{\text{pretrain}}} \left[ - \sum_{t=1}^{|x|} \log P_\theta(x_t \mid x_{<t}) \right]$$

If we have a training sequence $w_1, w_2, \dots, w_n$, the probability of generating this exact sentence $P(w_1, w_2, \dots, w_n)$ can be decomposed using the standard conditional probability formula:

$$P(w_1, w_2, \dots, w_n) = p(w_1)p(w_2 \mid w_1)p(w_3 \mid w_1, w_2) \dots p(w_n \mid w_1, w_2, \dots, w_{n-1})$$$$= \prod_{i=1}^n p(w_i \mid w_1, \dots, w_{i-1})$$

This core conditional probability $p(w_i \mid w_1, \dots, w_{i-1})$ can be modeled using older N-gram models (like KenLM) or modern Transformer models (like GPT-4). 

In LLM evaluations, people rarely cite raw cross-entropy; instead, they talk about Perplexity. Perplexity measures how "uncertain" or "blurry" a model is when generating a token. Higher perplexity = a worse, more confused model. Its mathematical definition is:

$$\text{Perplexity}(LM) = \left( \prod_{i=1}^m \frac{1}{q(w_i \mid w_1, w_2, \dots, w_{i-1})} \right)^{\frac{1}{m}}$$

Perplexity and Cross-Entropy are the exact same metric, just expressed differently.

### 3.2.2 Long Context
To achieve this long-context capability, the research is divided into two primary directions: Extending Positional Encoding and Adjusting the Context Window. 

A model's ability to handle context is naturally limited by the length of the texts it saw during training. If it encounters text longer than that distribution, its performance degrades. The mainstream RoPE (Rotary Positional Embedding) method, without special modification, lacks good "extrapolation" capabilities. To fix this, researchers use techniques like position interpolation and position truncation to adjust the rotation angles of the sub-spaces so they don't exceed the original context window's limits. Some positional encodings naturally allow the model to build text beyond its original trained context window. Methods like T5 biases, ALiBi, and xPos exhibit varying degrees of this extrapolation ability. While extrapolation allows a model to fluently generate long text, its actual comprehension of that long text often falls short compared to short texts. To achieve true long-context comprehension, the model must undergo additional fine-tuning on longer texts.

Instead of just messing with the encoding, another highly effective strategy is to use a "restricted attention mechanism"  to modify how the context window itself operates. Here are three common methods:

1. Parallel Context Window. It divides the input text into several independent segments. Each segment is encoded independently, but they share the same positional encoding information. During generation, the attention mask is adjusted so the new token can access all preceding tokens across segments. This method cannot effectively distinguish the sequential relationship between different segments, leading to poor performance on certain tasks.

2. Λ-shaped Context Window. It selectively forces the model to only pay attention to two things: the tokens immediately neighboring the current Query, and the tokens at the very beginning of the sequence. Everything else in the middle is ignored. Because it completely drops the information of the ignored tokens, this method fails to fully utilize all the context information.

3. Token Selection. It aims to effectively approximate full attention by picking only the most important $k$ tokens. By token similarity, it splits tokens into "close" (inside the window) and "far" (outside the window). For the far tokens, it uses external storage to save their Key-Value pairs and uses a k-nearest neighbor search to fetch only the most relevant tokens needed for the current generation step. By chunk similarity, it divides the sequence into chunks of different lengths and extracts only the most relevant sub-chunks for attention calculation.
   
## 3.3 Optimizer

### 3.3.1 Naive SGD
Stochastic Gradient Descent (SGD) is a core optimization algorithm used to train machine learning and deep learning models. Its primary job is to adjust the model's parameters (like weights and biases) to minimize the loss function—which is the measure of how far off the model's predictions are from the actual truth.

Unlike standard Batch Gradient Descent, which calculates the error using the entire dataset before taking a single step, SGD updates the parameters using the gradient from just one randomly chosen data point (or a small "mini-batch") at a time.

Because it only looks at a fraction of the data per step, SGD is incredibly fast and computationally efficient. This causes its path to the minimum to be "noisy" or erratic, but this exact noise actually helps the model bounce out of suboptimal local minima.

The mathematical update rule for true SGD at a given time step $t$ is:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t; x^{(i)}, y^{(i)})$$

### 3.3.2 Momentum SGD

Momentum SGD is an optimization technique designed to solve the "zig-zagging" or oscillation problem often found in standard Stochastic Gradient Descent. standard SGD can suffer from high variance updates. Because the gradient is calculated on small batches, it often fluctuates wildly, especially in narrow "valleys" where the surface is much steeper in one dimension than another. This leads to a zig-zag path that makes convergence painfully slow.

The core idea of momentum is to let the optimizer have a "memory" of previous gradients. By accumulating a history of gradients, the components that fluctuate (like the vertical zig-zags) tend to cancel each other out over time, while the components that consistently point toward the minimum (the horizontal direction) reinforce each other.

$$v_t = \mu v_{t-1} + g_t$$

$$\theta_{t+1} = \theta_t - \gamma v_t$$
### 3.3.3 NAG

Nesterov’s Accelerated Gradient (NAG) is an advanced variation of Momentum SGD designed to improve stability and prevent the model from overshooting the optimal point. While standard Momentum helps speed up training, it has a significant drawback: Momentum accumulation. Because the optimizer builds up "velocity," it can keep moving too fast as it approaches the minimum, causing it to oscillate wildly around the optimum instead of settling into it. NAG solves this by introducing a "look-ahead" mechanism. Instead of calculating the gradient at the current position and then adding momentum, NAG calculates the gradient at the position where the model would be after the momentum step.

The update rule for NAG is defined by the following two equations:

$$v_t = \mu v_{t-1} + g(\theta_t - \mu v_{t-1})$$

$$\theta_{t+1} = \theta_t - \gamma v_t$$

$\theta_t - \mu v_{t-1}$ represents the "look-ahead" position. We essentially "jump" to where the momentum would take us.

By using a more accurate gradient based on the "future" position, it reaches the optimum faster than standard Momentum. The "braking" effect prevents the optimizer from overshooting, making it much more reliable when it is close to the global minimum. It essentially gives the optimizer a sense of the terrain ahead, allowing it to "slow down" if the look-ahead gradient indicates a sharp turn or an uphill climb.

### 3.3.4 AdaGrad

AdaGrad (Adaptive Gradient) is an optimization algorithm that moves away from a single global learning rate. Instead, it dynamically adjusts the learning rate for every individual parameter based on its historical performance. Standard SGD uses a uniform learning rate for all parameters. However, AdaGrad is built on the philosophy that different parameters should use different learning rates. For parameters with large gradients (Those that oscillate or update frequently), AdaGrad aggressively decreases their learning rate to prevent overshooting. For parameters with small gradients (Those that update slowly), AdaGrad increases their relative learning rate to help them converge faster.

The update rule for AdaGrad is defined by two main components. AdaGrad first tracks the accumulation of all historical squared gradients for each parameter:

$$s_t = s_{t-1} + g_t^2$$

The parameters are then updated using the accumulated history to scale the learning rate:

$$\theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{s_t} + \epsilon} g_t$$

The main advantage of this approach is that it allows the model to converge more quickly by automatically "braking" on high-frequency features and "accelerating" on rare or slow features.

Learning Rate Schedulers like Cosine Annealing  and StepLR adjusts the global learning rate over time. AdaGrad takes this a step further by making those adjustments specific to each weight in the network.

Cosine Annealing adjusts the learning rate according to the periodic characteristics of a cosine function. Instead of just going down, the learning rate decreases from a maximum value ($lr_{\text{max}}$) to a minimum value ($lr_{\text{min}}$) and then periodically increases back. This wave-like pattern repeats throughout training. By raising the learning rate occasionally, the model can "jump out" of suboptimal local minima. This allows the model to explore a broader parameter space more effectively.

$$lr_t = lr_{\text{min}} + 0.5 \cdot (lr_{\text{max}} - lr_{\text{min}}) \cdot \left(1 + \cos\left(\frac{t}{T} \cdot \pi\right)\right)$$ 

StepLR is a simpler, "staircase" approach where the learning rate stays constant for a set number of epochs and then drops abruptly. You define a step size and a decay factor (a number less than 1, like 0.1). Every time the model completes the defined number of epochs (the "step"), the current learning rate is multiplied by the decay factor. The method is very simple to implement and helps the model settle into a minimum as it gets closer to the end of training. However, as the change is very sudden, it can sometimes disrupt training stability compared to smoother schedulers.
### 3.3.5 RMSProp

RMSProp was proposed by Geoffrey Hinton as a way to "forget" very old gradients. Instead of a simple sum, it uses an Exponentially Weighted Moving Average (EWMA). In AdaGrad, the denominator $s_t$ grows indefinitely. In RMSProp, the denominator is a weighted average that favors recent gradients. This prevents the learning rate from becoming too small to move, allowing the model to continue learning indefinitely.

Update the squared gradient average:

$$s_t = \rho s_{t-1} + (1 - \rho) g_t^2$$

Update the parameters:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t$$

### 3.3.6 Adam

Adam is currently the "gold standard" for most deep learning tasks, especially for training Large Language Models. It is essentially Momentum + RMSProp combined into one powerful algorithm. Adam tracks two different "moments":

- First Moment ($m_t$): The mean of the gradients (this is the Momentum part).
- Second Moment ($v_t$): The uncentered variance of the gradients (this is the RMSProp part).

Estimate the moments:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias Correction:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad , \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Since $m_t$ and $v_t$ are initialized at zero, they are biased toward zero at the start of training. Adam fixes this with a correction step:

Update Parameters:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

It is computationally efficient, requires little memory, and is well-suited for problems with very large data or parameters.
### 3.3.7 AdamW

AdamW (Adam with Weight Decay) is a modification of the standard Adam optimizer designed to improve model generalization. It is currently the industry standard for training state-of-the-art Large Language Models, including LLaMA 2. 

While the original Adam algorithm is powerful, it often fails to generalize as well as SGD with Momentum. To fix this, researchers typically add $L_2$ regularization. However, in the standard Adam implementation, $L_2$ regularization does not function the same way it does in SGD because it becomes entangled with the moving averages (moments). The defining feature of AdamW is that it decouples (separates) the weight decay from the gradient update steps. In standard Adam with $L_2$, the regularization term is added directly to the gradient ($g_t$) before the first and second moments ($m_t, v_t$) are calculated. In AdamW, the weight decay is applied independently at the very end of the parameter update step. This ensures that the weight decay strictly shrinks the weights without being distorted by the adaptive learning rate's denominators.

$$\theta_t \leftarrow \theta_{t-1} - \eta_t \left( \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

By decoupling the weight decay, AdamW recovers the original intent of $L_2$ regularization: pushing weights toward zero to prevent overfitting. This simple change is what allows models like LLaMA 2 to achieve much better stability and performance during massive pre-training runs.

## 3.4 Incremental Pre-training
Incremental Pre-training is the process of taking that already pre-trained base model and continuing to train (or fine-tune) it using newly acquired data. The goals is to allow the model to continuously learn new knowledge and adapt to new domains without losing the foundational knowledge it already has. Think of it as "updating on an existing foundation" rather than building a new house from scratch every time you want to add a room. It reuses the parameters and knowledge the base model already learned, constantly expands the model's knowledge base with new events or domain-specific info, and significantly reduces the computing power and time required compared to starting over.

Training models isn't a one-and-done process. Incremental training solves several major real-world challenges:
- Data and Knowledge are Constantly Evolving: The real world is dynamic. New vocabulary, jargon, and historical events emerge daily. Furthermore, specific industries (like healthcare, finance, or law) experience rapid regulatory and market changes. If a model only relies on its original training data, its knowledge becomes outdated.

- Training from Scratch is Prohibitively Expensive: Training a Large Language Model (LLM) requires hundreds of GPUs, weeks or months of time, and massive energy consumption. Doing this every time new data arrives is financially and logistically impossible.

- Preventing "Catastrophic Forgetting": If you only train a model on new data, it tends to overwrite and forget its old capabilities—a phenomenon known as catastrophic forgetting. Incremental training balances the retention of old knowledge with the absorption of the new, maintaining overall model stability.

By utilizing incremental pre-training, developers and organizations gain several key advantages:

- Maintained "Freshness": The model consistently understands recent concepts and news.

- Rapid Domain Adaptation: When moving into a highly specialized niche, the model can quickly adapt to the specific vocabulary and rules of that domain, shortening development cycles.

- Resource Savings: It saves massive amounts of compute power, time, and human effort.

- Handling Continuous Data Streams: It is perfect for online or real-time systems (like chatbots or sentiment monitoring) that receive a continuous flow of incoming data.

The lifecycle of an incrementally trained model usually follows this path:
1. Initial Pre-training: Train a base model on a massive, general dataset.

2. Data Acquisition: As time passes or business needs shift, collect new training corpora.

3. Incremental Updating: Combine the base model with the new data. During this phase, specific techniques (like mixed sampling or regularization) are used to learn the new info while protecting the old parameters.

4. Validation & Evaluation: Test the newly updated model on both general tasks (to ensure it didn't forget the basics) and new tasks (to ensure it actually learned the new material).

During incremental pre-training, it is normal for the "Loss" (error rate) to go up. A short-term increase in Loss is completely normal and expected. The new data might look significantly different from the original data (e.g., a completely different industry's jargon). The model needs time to adapt its parameters to this "unfamiliar" input. The model experiences tension. It is trying to retain its original representations while simultaneously altering them to fit the new data. If it fails to perfectly balance this immediately, its performance on older data might temporarily dip. Factors like learning rates or the way old and new data are mixed in training batches can cause short-term instability. A healthy incremental training run will show Loss spiking initially as it encounters unfamiliar data, and then slowly, steadily converging (dropping) as it masters the new information. You must monitor two specific metrics:

1. New Data Validation Loss: Proves the model is actually learning the new concepts.

2. Old Data/Core Task Loss: Proves the model is successfully avoiding catastrophic forgetting.

If the Loss spikes significantly, continuously, and refuses to come back down, you should investigate:

- Data Quality: Are the new data labels correct? Is the distribution shift simply too extreme?
- Hyperparameters: Is your learning rate too high? Is your mix of old/new data imbalanced?
- Architecture: Do you need more advanced techniques to force the model to remember?

To ensure the model successfully absorbs new data without forgetting the old, engineers use several common techniques during the incremental update step:

- Mixed Sampling (Replay/Memory-based): Keep a small stash of the original training data and mix it in with the new data. This forces the model to constantly "review" the old material.

- Regularization Methods (e.g., EWC - Elastic Weight Consolidation): Identify which mathematical weights/parameters were most crucial for the original tasks. Apply strict mathematical penalties to prevent the system from changing those specific weights too drastically.

- Knowledge Distillation: Use the original, untouched base model as a "Teacher" to guide the updated "Student" model, ensuring the student's outputs don't deviate wildly from what the teacher would have said on old tasks.

- Modular Training: Architect the model so that certain structural modules handle the old knowledge, while different modules are optimized for the new knowledge, preventing them from clashing.

Incremental pre-training is a necessary "break-in" process. As long as the model eventually adapts to the new distribution and shows stable performance on both new and historical tasks, temporary fluctuations in the error rate are simply the cost of learning.

The Learning Rate determines the "step size" a model takes when updating its parameters along the gradient direction during training. If the LR is too low, the model learns too slowly or struggles to grasp new knowledge. If it's too high, it "over-steps," causing the loss function to oscillate wildly or even diverge entirely. It directly dictates the stability of the training process, impacting both how well and how fast the model converges.

Incremental Pre-training involves taking a model that has already learned a vast amount of general knowledge and training it further on new data or domains. Because the base model has already reached a high-quality, stable parameter space, adjusting the LR requires a delicate balance. If the learning rate is too high, the model overfits the new data, drastically altering existing weights, and causes Catastrophic Forgetting (destroying the old knowledge). If it is too low, the model fails to effectively absorb the new data. Generally, you should set the starting LR for incremental pre-training to about 10% of the maximum LR used during the initial pre-training phase. You are "fine-tuning" an already capable model, not building from scratch. Smaller steps prevent massive disruptions to established parameters. If the new data is massive and fundamentally different from the old data, you can slightly increase the LR. However, start small, observe the loss curve, and raise it gradually. Be conservative and use a smaller LR. With limited data, a large step size will cause the model to overfit the small dataset and forget everything else. When increasing the batch size by a factor of $k$, the LR should also be scaled up, but rarely at a 1:1 linear ratio. If the batch size increases by 4x, the LR should roughly increase by 2x (effectively following a square-root scaling rule). This maintains consistent update efficiency while preventing instability. If the training or validation loss spikes and refuses to drop, or if performance on the old dataset plummets, reduce the LR immediately. To further prevent catastrophic forgetting, mix in a certain percentage of the old training data, or use regularization techniques like Knowledge Distillation or EWC.

Warmup is a scheduling technique where training begins with an intentionally tiny Learning Rate, which is then slowly and steadily increased until it hits your target (maximum) LR. After hitting the peak, the LR typically decays normally. This is because, at the very start of training, gradients can be chaotic. Jumping straight into a high LR can cause massive loss fluctuations or "gradient explosions." Starting small allows the model to find a stable optimization direction first, ensuring smoother and more optimal learning once the LR peaks.

Warmup Ratio is the percentage of total training steps (or epochs) dedicated entirely to the Warmup phase. It should be set based on training length. For 1 Epoch (Typical Pre-training), set it around 0.01 (1%). For 3 Epochs (Typical Supervised Fine-Tuning - SFT), set it around 0.03 (3%). The overall final impact of warmup is smaller, but a ratio between 0.01 and 0.06 is still recommended for a safe "take-off." If you plan to use a exceptionally high peak LR, you need a larger warmup_ratio (e.g., 0.05 to 0.1) to create a longer buffer and prevent initial model collapse. If your peak LR is naturally tiny, your warmup ratio can be much smaller (e.g., 0.005). Extremely large batch sizes cause immense gradient accumulation early on. To have it, increase the warmup_ratio to force a slower, gentler LR ascent in the beginning.

