# 4. Post-Training
## 4.1 SFT (Supervised Fine-Tuning)
Supervised Fine-Tuning (SFT) is a pivotal stage in the LLM training pipeline that follows large-scale unsupervised pre-training. It involves further training a pre-trained base model using a smaller, high-quality labeled dataset consisting of prompt-response pairs. SFT aligns model outputs with actual user needs (e.g., turning a text-completer into a helpful assistant), achieves significant performance gains with greatly reduced computational resources and time costs compared to training from scratch, and allows for quick adaptation to professional fields such as Medicine, Law, or Finance. Pre-trained models learn a general distribution but struggle with precise instruction following and specific output formats. Fine-tuning allows the model to preserve its vast language abilities while using a small amount of task-specific data or human feedback to steer its output, and achieving a significant boost in generation quality, safety, and controllability.

SFT introduces special tokens that the model never encountered during pre-training to define dialogue roles:
- Standard Roles: user, assistant, and system.

- Extended Roles: Depending on the task, you can add roles like background, aside, or event.

- The eos_token: This is perhaps the most important addition. Pre-trained models often don't know when to stop; SFT teaches the model to signal the end of a response.

- Knowledge Construction: Unique special tokens can be used to "inject" specific knowledge to verify if the model is overfitting or truly learning.

The time it takes for a model to generate a response can be modeled as

$$y = kx + b$$

- $b$ (Time to First Token): This is heavily influenced by prompt length and the KV cache mechanism. It is often dozens of times larger than $k$.

- $kx$ (Generation Time): Generation speed is positively correlated with the total number of tokens produced. This is why techniques like Chain of Thought (CoT), while effective, increase latency—they force the model to generate many more tokens.

In SFT, loss is typically not calculated on the prompt, because prompt homogeneity is often high. If the model learns to predict the prompts, it develops a bias. However, loss can be included if every prompt in the dataset is strictly unique. The SFT process includes these steps:

1. Data Preparation. Data quality is the most critical factor in the SFT process. High-quality labels lead to precise task mastery, while noisy data causes performance to plummet. The dataset must cover various tones, styles, contexts, and edge cases to ensure the model generalizes well. Besides, while SFT needs much less data than pre-training, it requires a "moderate" amount to avoid underfitting. Data collection requires specific $(x, y)$ pairs (input prompt and target response), the same BPE / SentencePiece vocabulary as pre-training to maintain embedding consistency, and concatenating the prompt and target into a single sequence: [x, special_token, y]. Unlike pre-training data which is always "full-pack" (e.g., 4K/8K chunks), SFT data is naturally variable in length.

2. Base Model Selection. The foundation dictates the result. GPT series are preferred for generation, while BERT is better for classification and extraction.

3. Finetuning Methods. Full Fine-tuning updates all parameters for maximum performance but at a high cost. Partial Fine-tuning (e.g., Freeze) fixes most layers and only tunes specific modules. This is efficient and helps retain pre-trained knowledge. Setting the LR is a balancing act. Setting it too high leads to Catastrophic Forgetting of pre-trained knowledge. Too low results in painfully slow convergence. Schedulers standard practice involves using Cosine Annealing or Linear Decay.

4. Evaluation. Continuous monitoring and objective testing are the only ways to verify if the SFT process is actually working. The validation set is to monitor performance during training. It acts as a "sanity check" to help developers adjust training strategies in real-time and, most importantly, prevent overfitting. For the results to be objective, the test set must be completely independent of both the training and validation sets. A model's score on the test set is the ultimate "gold standard" for its final performance. Optimization isn't just about looking at a final score. You must manually analyze samples where the model predicted incorrectly. This "post-mortem" is the best way to find latent data issues or specific model deficiencies. If the results are not good enough under specified top-k / temperature, you need to adjust the learning rate or batch size.

5. Domain Adaptation. To move a model from a general assistant to a specialist (like a doctor or a lawyer), specific adaptation techniques are required. Domain Knowledge Injection involves introducing specialized labeled data or professional knowledge bases. If a new task is similar to an old one, you can perform a "preliminary fine-tuning" on the similar task before doing the final tuning on the target task. This multi-step fine-tuning strategy is highly effective at boosting final model performance.

The hallucination is when the model provides a factually incorrect answer (confidently "talking nonsense"), or when the model has the knowledge but produces a misaligned answer after alignment processing. SFT and RLHF can train a model to refuse to answer questions it doesn't know. In real-world products, raw model hallucinations are often "blocked" or rewritten using RAG (Retrieval-Augmented Generation) or function calling within a full-chain system.

In conclusion, in the lifecycle of a Large Language Model, Pre-training and Supervised Fine-Tuning (SFT) represent the two foundational stages of development. While they share the same underlying architecture, their goals, data requirements, and methodologies are fundamentally different. The relationship between these two is synergistic. You cannot have effective SFT without a robust pre-trained base, and a pre-trained base is rarely "useful" for end-users without SFT. Pre-training provides the high-quality starting point (feature representation). Without it, SFT would require significantly more data and struggle to generalize. SFT improves the model's controllability and safety. It teaches the model special tokens like user, assistant, and system, and most importantly, the eos_token so the model knows when to stop talking.

Here is a thorough breakdown of these two stages:

- Pre-training is the initial phase where a model learns the basic structure, patterns, and general knowledge of human language. It does not target a specific task but seeks to master the "rules of the game." It uses Self-Supervised Learning, where the data itself provides the signal, predicting the next word in a sequence (e.g., GPT series), or hidden words within a sentence (e.g., BERT). Pre-training usually utilizes giant, diverse datasets from the web, books, and code. Each data entry is typically "full-pack" (filling the context window, e.g., 4K or 8K tokens).

- SFT is the second phase, occurring after the base model is already established. It transforms a "knowledgeable" model into a "helpful assistant." It is training on a small-scale labeled dataset to adapt the model to specific application needs and instruction following, using Supervised Learning with explicit target answers provided by humans or high-quality teacher models. The data type is specific prompt-response pairs. Unlike pre-training, the data length is natural—the sequence is only as long as the prompt and the answer actually are. While SFT adapts the model, it is crucial to control knowledge injection. New knowledge should only represent 10% to 20% of the mix; forcing too much can cause the model to forget its foundational pre-trained reasoning.

## 4.1.1 Instruction Tuning

## 4.2 Reinforcement Learning in LLM 
