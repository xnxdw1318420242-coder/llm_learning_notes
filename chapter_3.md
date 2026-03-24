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
