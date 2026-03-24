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

#### 3.1.3.2 Sensitive Content Filtering

Filtering out toxic content and Personally Identifiable Information (PII) is mandatory to prevent models from generating abusive outputs or leaking private user data. 

- Toxicity: The pipeline uses classifiers trained on datasets like Jigsaw to precisely identify and filter out toxic, abusive, or biased content.

- Privacy (PII): Heuristic rules are heavily used here. For example, the Dolma dataset uses a strict rule for emails, phone numbers, and IP addresses. If a document has < 5 private items, they are replaced with placeholders like [EMAIL_ADDRESS]. If a document has >= 6 private items, the entire document is directly deleted.

#### 3.1.3.3 Data Deduplication

Anthropic discovered that training on duplicate data causes the model to output repetitive, localized loops. Furthermore, duplicates weaken a model's ability to utilize context, actively harming its generalization and in-context learning. It can also cause a phenomenon known as Double Descent, where the training loss drops, unexpectedly spikes, and then drops again, leading to severe training instability.

The two primary categories for deduplication matching are Exact Matching and Approximate Matching. Exact Matching identifies text segments that are character-for-character identical. It typically utilizes suffix arrays to find and match the longest common substrings that meet a minimum length requirement. This is often applied at the sentence level to remove verbatim copies or highly repetitive boilerplate text. Approximate Matching is designed to find "near-duplicates"—documents that are slightly different (e.g., different ads, minor formatting changes) but contain the same core content. Comparing every document character-by-character would be computationally impossible at a "Trillion-token" scale, so it uses Locality-Sensitive Hashing (LSH).

MinHash is the primary tool for approximate matching. It treats a document as a set of features (like n-grams). It applies multiple random hash functions to every element in that set. For each hash function, it selects the minimum hash value to represent the set. These minimum values form a "signature." By comparing these small signatures instead of the full text, the system can rapidly estimate the Jaccard similarity between two massive documents. This allows the pipeline to skip pairwise comparisons of every single element, making it highly efficient for processing ultra-large-scale datasets. SimHash transforms text features into a fixed-length bit string (a fingerprint). Similarity is measured using Hamming distance; the smaller the distance between two fingerprints, the higher the similarity between the texts.

TF-IDF is another "soft" dedup method that compares documents based on the frequency and rarity of words.

Real-world pipelines don't just pick one. They use a multi-stage, multi-granularity approach. In the dataset / document level, they use Approximate Matching (MinHashLSH) to remove documents with high similarity (e.g., similarity scores > 0.8). This is faster for broad filtering. In the sentence level, they use Exact Matching to perform a finer "clean-up," deleting specific identical strings that might still exist within otherwise unique documents.

Early research believed that increasing model parameters was the most important factor. However, current science proves that data quality is the true bottleneck. By elevating data quality, smaller models can match or beat massive models. Conversely, using low-quality data causes training to fail. Crucially, if a model is trained on factually incorrect or outdated data, it will confidently generate false information—a phenomenon formally defined in the text as Hallucination.

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

## 3.2 Training Tasks 
To achieve this long-context capability, the research is divided into two primary directions: Extending Positional Encoding and Adjusting the Context Window. 

A model's ability to handle context is naturally limited by the length of the texts it saw during training. If it encounters text longer than that distribution, its performance degrades. The mainstream RoPE (Rotary Positional Embedding) method, without special modification, lacks good "extrapolation" capabilities. To fix this, researchers use techniques like position interpolation and position truncation to adjust the rotation angles of the sub-spaces so they don't exceed the original context window's limits. Some positional encodings naturally allow the model to build text beyond its original trained context window. Methods like T5 biases, ALiBi, and xPos exhibit varying degrees of this extrapolation ability. While extrapolation allows a model to fluently generate long text, its actual comprehension of that long text often falls short compared to short texts. To achieve true long-context comprehension, the model must undergo additional fine-tuning on longer texts.

Instead of just messing with the encoding, another highly effective strategy is to use a "restricted attention mechanism"  to modify how the context window itself operates. Here are three common methods:

1. Parallel Context Window. It divides the input text into several independent segments. Each segment is encoded independently, but they share the same positional encoding information. During generation, the attention mask is adjusted so the new token can access all preceding tokens across segments. This method cannot effectively distinguish the sequential relationship between different segments, leading to poor performance on certain tasks.

2. Λ-shaped Context Window. It selectively forces the model to only pay attention to two things: the tokens immediately neighboring the current Query, and the tokens at the very beginning of the sequence. Everything else in the middle is ignored. Because it completely drops the information of the ignored tokens, this method fails to fully utilize all the context information.

3. Token Selection. It aims to effectively approximate full attention by picking only the most important $k$ tokens. By token similarity, it splits tokens into "close" (inside the window) and "far" (outside the window). For the far tokens, it uses external storage to save their Key-Value pairs and uses a k-nearest neighbor search to fetch only the most relevant tokens needed for the current generation step. By chunk similarity, it divides the sequence into chunks of different lengths and extracts only the most relevant sub-chunks for attention calculation.
   
## 3.3 Training Optimization
