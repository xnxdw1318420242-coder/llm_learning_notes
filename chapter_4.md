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

### 4.1.1 Instruction Tuning
Essentially, this is the process that turns a "raw" pre-trained model into a helpful, conversational assistant by teaching it to obey human commands. Instruction Tuning is a supervised fine-tuning process that uses specialized Instruction Datasets. These datasets contain explicit commands (e.g., "Translate the following text into French" or "Summarize this article") along with the correct answers. Its goal is to move the model beyond simple text completion and teach it the specific pipeline of Reading the Instruction $\rightarrow$ Processing the Input $\rightarrow$ Generating a Compliant Output. The model becomes highly sensitive and "obedient" to human prompts, making it capable of handling diverse tasks and complex dialogue scenarios.

To train the model precisely, researchers represent each data entry as a triplet $(i^k, x^k, y^k)$:

$i^k$ (Instruction): The task command.

$x^k$ (Input): The specific content to be processed.

$y^k$ (Output): The expected "gold standard" result.

During the training phase, the Instruction and the Input are merged into a single sequence called a Prompt:

$$prompt^k = [i^k, x^k]$$

In the context of an Autoregressive Language Model, this prompt is converted into a sequence of tokens $(i_1, \dots, i_m, x_1, \dots, x_n)$. The model is then trained to predict the target tokens ($y^k$) one by one, based on that combined context.

When preparing the data, the core philosophy is that while your prompts can be slightly informal, your answers must have absolutely zero errors (not even a single punctuation mistake, like mixing up English and Chinese quotation marks). You must expose the model to every type of task it might encounter, and this might include standard ChatGPT tasks (translation, emoji chat), traditional NLP tasks (NER, reading comprehension), and specific business needs (e.g., feeding it Spring Festival couplets or lantern riddles). Every single piece of SFT data must have a task type label. Never mix data blindly, or error analysis later will be a disaster. Hard tasks get more data, simple tasks get less data.

Data format diversity is important. To prevent the model from finding "lazy" patterns, avoid just using "Translate A to English." Create diverse scenarios: "I'm traveling in the UK and need to ask a local... " or "I'm an English teacher explaining to my students..." This prevents the model from overfitting to specific trigger tokens. Also mix short and long data. For long prompts, intentionally hide the key instructions at the start, middle, and end of the text. This prevents the model's attention mechanism from degenerating to only looking at the first and last tokens. Besides, include prompts that demand extremely long outputs (e.g., "no less than 10,000 words") so the model learns it shouldn't just stop after a few tokens. The placement of key information in the prompt must be highly random. Furthermore, the prompts should guide the model to switch between different topics among multiple rounds of conversation. Some queries might be relevant to the whole conversation, and some might not; The model should be able to tell if the query is relevant to the whole conversation. Lastly, the answers should also be diverse. Since the loss function takes in the answer, limited diversity in answers will result in model overfitting.

To prepare the data for SFT, the first step is to prepare a diverse and high-quality set of instructions. You can write them manually, use open-source Instruct datasets (like those on Huggingface), or automatically generate them. Based on Stanford's research, you can also prepare a few "seed prompts" for each task type and feed them to a powerful pre-trained model to randomly sample and generate a massive list of new, similar questions. LLMs have vast knowledge but struggle with fine-grained mastery. If a task is hard for a normal human to do in one step, do not expect the model to do it in one prompt. Break complex tasks down (e.g., Prompt 1: "Design an outline." $\rightarrow$ Prompt 2: "Expand the outline."). Format every instance as a triplet $(i^k, x^k, y^k)$ adding clear context, like: Instruction: ... \n Input: ... \n Output: ... Filtering is needed to drop the low-quality or conflicting instructions.

Once you have the prompts, you need to generate the "gold standard" target answers. If the budget is unlimited, use GPT-4 or Claude 3. But is the budget is limited, deploy open-source models like DeepSeek locally. Then, test your prompts on ChatGPT first to find the exact phrasing that yields the best result. Add "few-shot" examples to the prompt, but ensure you have a diverse "seed pool" of examples so the generated answers don't all look identical. There are two helpful strategy to prepare the answers:

- The "Small Model Bootstrapping". Use GPT-4 to generate about 1,000 high-quality answers and train a small model on this data. Then use your newly trained small model to generate the remaining 10,000+ data points. GPT-4's adherence to strict custom formats is only about 70%. A small model specifically trained on your format will hit 100% compliance.

- The CoT (Chain of Thought) Trick. CoT massively improves answer accuracy (especially in classification). However, training with CoT wastes inference time later. The trick: Use GPT-4 with CoT to generate the high-quality answer, but omit the CoT reasoning steps in your final SFT dataset to save training and inference costs.

The golden rule of SFT data is quality over quantity. Whether using GPT-4 or a self-trained model, data quality issues will occur. You must establish strict rules or use human reviewers to check the data. Models will often generate highly homogeneous (repetitive) data for specific tasks. You must deduplicate this. If you cannot easily filter it, aggressively delete highly repetitive training data. 

Your model needs to survive the real world, where users are messy and tasks are complex. To generate robustness data, include a specific ratio of data where the Answer is perfect, but the Prompt is flawed (e.g., typos, incomplete sentences), and apply special tags to this data. For tasks like RAG, Agent/Function Calling, or long-context processing, the challenge isn't algorithmic—it's pure data production and engineering. You must build this through iterative trial and error.

SFT truly begins after the model goes online. Human-crafted seeds are limited, but user creativity is infinite. Real users don't interact perfectly. For example, in code generation, a user might get an error, feed it back to the model, and take 4 to 5 rounds of interaction to get working code. This real-world multi-turn data is incredibly valuable and almost impossible to manually annotate; it must be extracted from user logs. Users ask complex things (e.g., "replace this specific word in the translation," or "what does this specific word mean?"). Extracting these logs trains the model in topic shifting, self-correction, and sticking to its opinions. When using log feedback (upvotes/downvotes) for RLHF/DPO, be aware of intentional data poisoning where users deliberately upvote bad answers to sabotage the model. Clean this strictly. 

The Data Flywheel Cycle: Periodically extract user logs $\rightarrow$ filter high-value prompts $\rightarrow$ call GPT-4 to generate perfect answers for them $\rightarrow$ add to your dataset to update the model.

Next, we introduce four main specialized categories for SFT training data preparation. 

1. RAG (Retrieval-Augmented Generation). For RAG, the upper limit of your system's capability is determined by the accuracy of your external retrieval database. To prepare the model to handle this data, you must rely on two external models: a Binary Classifier (to decide if RAG should be triggered—always trigger it for knowledge queries, regardless of whether you think the model already knows the answer) and an IR Model (to quickly fetch documents). When constructing the SFT data for RAG, you must train the model to handle four specific edge cases:

- Empty Retrieval: Teach the model exactly how to reply when nothing is found, preventing it from wildly hallucinating.
- Contradictory Retrieval: Ensure the model synthesizes the information rather than just blindly trusting the first or last sentence it reads.
- Irrelevant Retrieval: Train the model to recognize when the fetched documents don't answer the prompt and to state that clearly.
- Incorrect Retrieval (The Golden Rule): You must force the model to strictly answer based on the retrieved content, even if the model "knows" the retrieved content is wrong. The fundamental premise of RAG is that the Database > Internal Model Knowledge. If the model starts judging the database, RAG fails.

2. Agents / Function Calling. Function calling is simply the primary way "Agents" are implemented. The SFT preparation here is highly structural. You must define the rules in the system prompt. For example: "When encountering a math task, output <special_token> and call the calculator." Standard SFT uses system, user, and assistant. For Agent data, you must add new conversation turns for the tool itself. For example, add a Call Calculator turn and a Calculator Return Result turn into the data sequence. Most of the time spent on Agent technology isn't coding; it is spent training human data annotators to perfectly align with these specific formatting and calling standards.

3. Long Context Processing. Expanding a model's context window (e.g., up to 200K tokens) requires both technical training adjustments and very specific data construction. You will use NTK extrapolation to adjust the RoPE base and require Sequence Parallelism to handle the massive VRAM requirements caused by quadratic attention calculations. Do not just stitch short texts together. Models are lazy and will try to guess answers based on position. Your data must force the model to genuinely read the whole text (good sources: whole papers, books, or huge RAG outputs).

4. Complex Instructions. Complex instructions are prompts with multiple constraints (e.g., "Write >200 words, insert emojis, make it rhyme, and output a specific token"). Purely stacking SFT data will only get you so far. Standard next-token prediction struggles to satisfy multiple constraints simultaneously because it can't "plan ahead." Ultimately, solving this requires CoT (Chain of Thought) or self-correction abilities (similar to OpenAI's o1 route), where the model plans its answer or fixes it mid-generation if it realizes a constraint was missed. Until those advanced techniques are fully mature, you have to "hardcode" the data. If you use GPT-4 to generate a response requiring "at least 200 words", and GPT-4 only gives you 189 words, don't throw the data away. Just change the prompt to say "at least 180 words." Adjusting the prompt to fit the output is a highly effective, pragmatic way to build compliant SFT data without overly strict filtering.

Here are some overall practical tips for SFT data preparation.

1. Synthetic Data $\neq$ Messy Data. While most SFT data is synthetic, simply generating it isn't enough. You need multi-path synthesis and de-biasing. Generate multiple versions of an answer from different models. Then, use rules to rewrite them, add a touch of human polishing, and perform a final filtering pass. This significantly reduces the bias introduced by a single model's specific "preference" or tone.

2. Avoid "Catastrophic Forgetting". During SFT, a model can easily forget its general world knowledge—a phenomenon called Catastrophic Forgetting. You must appropriately mix in pre-training data - This reminds the model that it is a "generalist", not just a "rote-learning student" focused on one specific task.

3. Determine Epochs by Task Type. Don't just default to a single training number. Your training epochs should vary based on your data volume. In general scenarios, stick to 1 epoch to prevent the model from overfitting. For domains like Medicine, Finance, or minority languages where data is scarce, you should train for about 3 epochs so the model can fully "sense" the nuances of that domain.

4. Full Training > PEFT. If you have the hardware, the slide is very direct: Use full-parameter training if you can, and skip PEFT (Parameter-Efficient Fine-Tuning). PEFT was designed for when compute resources are limited. However, if your GPUs are sufficient and your data is stable, full-parameter SFT almost always yields the best results.

5. Respect the "Alignment Tax". Every model has its own "understanding boundaries." If you force a model to answer questions that are far beyond its current capabilities, it won't just fail—it will actually start performing worse on tasks it already knew how to do. If you push the model past its limit, you break its existing abilities rather than teaching it new ones.

#### 4.1.1.1 Packing
When training Large Language Models (LLMs), specifically during Supervised Fine-Tuning (SFT), data samples vary wildly in length—some are only a few dozen tokens, while others are thousands. SFT Packing is an optimization technique designed to handle this variance efficiently. In traditional training, a model uses a fixed sequence length. If a training sample is shorter than that length, the remaining space is filled with padding tokens. Padding tokens do not contribute to learning, yet the model still spends "compute" processing them. In datasets with many short samples (like Q&A pairs or instruction data), a massive amount of GPU power is wasted on processing useless padding. Instead of padding each sample individually, SFT Packing "stitches" multiple training samples together into a single, continuous sequence (a "fake long sentence") to fill the block size. This minimizes invalid padding and maximizes training efficiency.

Because padding tokens are largely removed, the model processes more "real" data per second. This is particularly effective for short-sample datasets like summaries or chat logs. If a batch originally only fit 1–2 long sentences, Packing might allow it to fit over a dozen short ones. This leads to higher GPU utilization and faster gradient updates. Since this is a data-level operation, it does not require changing the model's architecture. It is fully compatible with various positional encodings like RoPE or Absolute PE.

Early training frameworks (like the initial versions of DeepSpeed-Chat) often split multi-turn dialogues into separate samples. With Non-packing, a 3-turn dialogue $[q1, a1], [q2, a2], [q3, a3]$ would be treated as three independent samples. This causes the total data volume to swell (tripling in some cases) and results in very low efficiency. With Packing, these turns can be glued together, allowing the model to understand the entire conversational logic in one go without wasting tokens.

There are three main versions of how Packing is implemented in frameworks like LLaMA Factory:

1. Direct Concatenation (The Basic Version). This throws all tokens into a pool and cut them strictly by the block_size. Samples are often cut off mid-sentence at the start or end of a block, leading to ruptured semantics and lost context.

2. Knapsack Packing. Samples are sorted by length and then grouped (like the "knapsack problem" in algorithms) to fit as perfectly as possible into the block_size. Any remaining small gaps are padded. While the blocks are mostly full, reducing token waste, because all samples in a block use a standard attention mask (all 1s), different samples "see" each other. A model might accidentally learn that an unrelated prompt in the same block is the context for its current answer.

3. 4D Attention Mask (The Advanced Version). Samples are packed, but each is assigned a unique ID (0, 1, 2...). A Block Diagonal Mask (or 4D Mask) is generated. Even though they are physically in the same sequence, the mask ensures tokens only "attend" to other tokens within their own sample. This creates total isolation, solving the cross-contamination problem entirely.

Packing is not always a "free" upgrade. There are two major risks to consider:

- Gradient "Dilution" of Short Queries. Without Packing, if a batch contains only one short text, the model’s entire gradient for that update comes from that one piece of data, leading to intense optimization. With Packing, that same short text is bundled with 7–8 others. Its gradient contribution is now only a small fraction of the total batch. Short but difficult queries (like rhetorical questions or ambiguous instructions) may not be learned effectively because the "signal strength" is diluted by the surrounding data.

- Interference in Multi-turn Dialogues. If you don't use the isolation techniques like 4D Attention Mask, the model may get confused. The model might struggle to distinguish where one conversation ends and another begins. Even using <eos> or <pad> as separators isn't always enough because, without Block Attention, the mathematical attention mechanism still links them. In the worst case, the model treats "someone else's history" as the context for the current dialogue, resulting in "hallucinated" context and wasted compute.

SFT Packing is a powerful tool, but it requires the right infrastructure. Use Packing only if your framework supports Block Attention or 4D Attention Masks (currently supported by LLaMA Factory, DeepSpeed, and transformers v4+). This ensures you get the massive speed benefits of Packing without the "semantic pollution" caused by cross-contamination.
nsures the center of the class is actually a high-quality response

#### 4.1.1.2 Diversity

In the development of Large Language Models (LLMs)—specifically during the Supervised Fine-Tuning (SFT) stage—the philosophy has shifted from "the more data, the better" to "Quality and Diversity over Quantity." By the time a model reaches SFT, it is already "smart" from its massive pre-training phase. The goal of SFT isn't to teach the model new facts, but to teach it how to provide a high-quality response. Training on messy, repetitive, or low-quality data actually degrades the model's performance.

If your data is all the same, the model’s capabilities will be one-dimensional. Data diversity is the foundation of a model’s "cognitive breadth." To achieve true diversity, researchers focus on three distinct "pillars":

1. Task Type (What the model does). You must tell the model what "job" it is currently performing. Use established lists (like OpenAI's task list) including translation, summarization, roleplay, and code review. Include traditional NLP tasks like Named Entity Recognition (NER), Reading Comprehension (MRC), and Intent Recognition to give the model a solid baseline. Include tasks like legal/financial analysis or specific cultural tasks (e.g., writing traditional riddles). Don't distribute data evenly. Use a weighted structure: more data for complex tasks (like multi-step logical reasoning) and less for simple ones.

2. Data Form (How the information is written). This prevents the model from "pattern matching" based on specific keywords. Instead of saying "Please translate A to B," use scenario-based prompts: "I am an English teacher, help me translate this for my students." Progressively increase the complexity of instructions (e.g., the WizardLM approach) to help the model learn in "stages." Vary the length of prompts and ensure key information isn't always at the beginning. This forces the model's Attention mechanism to scan the entire input. Mix short and long answers. Occasionally demand "1000+ word detailed explanations" to prevent the model from becoming "lazy" and only giving short replies. In multi-turn dialogues, include samples where the topic shifts suddenly. This teaches the model to distinguish between current context and past history.

3. Semantic Diversity (The distribution in embedding space). Even if you have 100,000 instructions, they are useless if they all mean the same thing. Convert text into vectors (embeddings) to see how they "clump" in mathematical space. Ensure the data covers "Core" mainstream expressions as well as "Rare" or "Edge-case" styles.

There are several strategies to find a small, representative "handful" of data from a massive pile.

- K-Means Clustering. Convert text to vectors, cluster them, and pick the sample closest to each center. This is very straightforward. However, when we only picks "representatives," not necessarily "good" samples, a bad sample can be a cluster center.

- Diversity + Quality Weighting. Combine clustering with a "Quality Score" (derived from GPT-4 ratings or logic/grammar checks). This ensures the center of the class is actually a high-quality response, but requires much more complex to run.

- K-Nearest Neighbors (KNN) Weighting. Look at how "similar" a sample is to its neighbors. The less similar it is, the more "rare" and "precious" it is. This method looks for "individual personality" in the data rather than just the most common denominator.

Diversity is equally vital in the pre-training phase. Recent research (specifically the D4 Paper) highlights:
- Synonyms are Knowledge: Do not delete data just because it means the same thing. Similar sentences expressed in different ways (e.g., high-EQ vs. low-EQ phrasing) are valuable "knowledge expressions" the model needs to learn.

- Document-Level Key: Diversity across websites, languages, and styles is more important than just sentence-level variety.

- Density Balance: Balance "fragmented" web data with "dense" academic or technical texts.

#### 4.1.1.3 Sandbox Verification

In the high-stakes world of Large Language Model (LLM) training, "blindly" feeding the model massive amounts of data is a recipe for wasted budget and mediocre results. Sandbox Verification is a systematic pre-rehearsal system designed to evaluate and screen data quality before the expensive, full-scale training begins. The ultimate goal of a Sandbox is to ensure the model "eats" only the data that is most useful, cost-effective, and performance-enhancing.

Before committing to full-scale training, the Sandbox system seeks to answer three critical questions:
1. Utility: Which specific datasets actually help the model's performance?
2. Compatibility: Do different data combinations work well together, or do they "clash" and overlap?
3. Scalability (ROI): If we invest more money to expand our data, which specific categories provide the best return on investment?

The Sandbox process follows a logical progression from individual unit analysis to complex combinations, and finally to high-cost scaling.
1. Single-OP Processing (Unit Analysis). In this stage, the raw dataset $D$ is dismantled into different Operation Pools (OP) based on dimensions like source, task type, and style. Each pool is further divided into three tiers based on quality or importance (e.g., 0–33%, 33–67%, and 67–100%). This allows researchers to simulate the "cost-performance" ratio of the data. Then instead of a giant model, we train $3N + 1$ lightweight models (where $N$ is the number of pools) to test each pool’s impact. Data is ranked by its actual contribution to model metrics. You might find that the top 33% of one pool is better than random data, while the bottom 33% of another is essentially "junk" that adds no value.

2. Multi-OP Processing (Combination Analysis). Once we know which individual pools perform well, we need to see if they play nice together. The goal is to find a set of pools ($OP^*$) that are mutually non-overlapping and mutually reinforcing. This step involves three types of analysis:
   - Correlation Analysis: Checking if different pools have high semantic overlap. If two pools are too similar, we don't need both.
   - Diversity Analysis: Ensuring the data is "diverse" as well as "accurate." High diversity across different domains and styles leads to better model generalization.
   - Duplication Analysis: Identifying redundant content within the combination to prevent wasting training costs on repetition.
  
3. Higher-Cost Data Scaling. Steps 1 and 2 are performed within a "cost-controlled" environment using small models and limited training. Once $OP^*$ is identified, we move to the final stage with high confidence. Use the validated high-quality combinations to expand into a massive, high-quality dataset. This refined data is finally used for high-cost tasks like full-scale Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning.

The true value of the Sandbox is financial and technical transparency. It identifies "marginal" data that takes up space without adding intelligence. By the time you reach the expensive scaling phase, you aren't guessing. You know exactly which data combinations are worth the "big money." Models trained on Sandbox-verified data are more stable and perform better on target tasks.

Sandbox Verification turns model training from a "blind gamble" into a calculated engineering process where every dollar spent on compute is backed by data-driven evidence.

### 4.1.2 Training Strategy
Utilizing a model from Hugging Face involves coordinating four distinct categories of files that work together to turn your raw text into an intelligent response.

1. The Preprocessing Phase (The Tokenizer). A model cannot read English; it only reads numbers (tokens). The Tokenizer related files act as the translator. tokenizer.json & tokenizer_config.json contain the complete instructions for the conversion process. They define the "rules of engagement," including special symbols like [CLS] or [SEP]. The vocab.json file maps specific words or sub-words to unique integer IDs. The merges.txt file is specifically used for BPE (Byte Pair Encoding) to decide how smaller character pieces should be merged into recognizable tokens. You load these files first to transform your input string into a tensor of IDs that the model can actually process.

2. The Architecture Phase (The Blueprint). config.json tells your software exactly how many layers, hidden units, and attention heads to "spawn" in your computer's memory. Once the model is built, generation_config.json dictates how it behaves during a chat. It sets parameters like temperature (creativity), Top-k sampling, and maximum generation length.

3. The Execution Phase (The Weights). The Model Weight files are the core "brain matter" containing billions of parameters learned during training. Modern models use the "SafeTensors" format stored in .safetensors files because it is secure and faster to load. Large models are often sharded (split into multiple files like model-00001-of-00004) so they can be handled by standard hardware. model.safetensors.index.json provides a map. It tells the loader exactly which specific parameters are located in which sharded file, ensuring they are loaded into the correct part of the architecture. You "pour" these weights into the structure defined by your config.json.

4. The Logistics & Legal Phase (Management). The remaining files handle how you download the model and what you are allowed to do with it. .gitattributes defines the Git LFS (Large File Storage) settings. Because weights are gigabytes in size, standard Git can't handle them; LFS is required to fetch the actual parameter data. README.md & NOTICE provide the manual for the model and credit any third-party dependencies or research papers. LICENSE is critical for legal utilization. It defines the redistribution and commercial use rules (e.g., Apache 2.0 or proprietary licenses).

When initiating SFT, the choice of framework significantly impacts both performance and ease of deployment. While both DeepSpeed and Megatron are powerful, DeepSpeed is generally preferred for SFT due to its superior open-source ecosystem. It supports AutoModelForCausalLM directly, avoids complex format conversions (like Hugging Face to Megatron), and integrates seamlessly with inference frameworks like TGI and vLLM. There are parameters to be finetuned no matter which framework to choose:
- Core: epoch, learning_rate, batch_size, scheduler_type, and gradient_accumulation_steps.
- Performance: zero_stage, offload, max_seq_len, and seq_parallel_size.
- Auxiliary: weight_decay, warmup_steps, and dropout.

Success in SFT relies on finding the "Goldilocks zone" for your specific model and data scale. The recommended data volume is approximately 100k samples. While you can experiment between 10k and 1M, dropping below 10k often leads to underfitting, while exceeding 1M significantly increases costs with diminishing returns. Generally, 1 to 3 epochs are suggested. Small models require a higher learning rate to learn effectively. Large models require a lower learning rate to maintain stability. Gradient accumulation is commonly set to 16, 32, 64, or 128 to simulate larger batch sizes on limited hardware. Always use warmup at the start of training to stabilize the process. Conversely, Dropout is usually disabled; while it can prevent overfitting, it often introduces unwanted training instability in SFT.

The loss curve is your primary diagnostic tool for understanding model health. 7B/13B models usually start with a loss around 2.0. 72B models usually start lower, between 1.0 and 2.0. Complex data, special tokens, and generative tasks may cause initial loss to spike to 3.0. SFT loss typically converges toward 0.5. If it drops much lower, the model may be losing its ability to generate diverse outputs. Rising loss is a red flag. This is almost always an indicator of code implementation bugs rather than data difficulty. Staircase loss is a classic sign of overfitting, which can manifest in two ways:
- Format Overfitting: The model masters the style (e.g., always outputting JSON). This is often a positive sign of instruction following.
- Content Overfitting: The model memorizes specific answers. This is a negative sign that ruins generalization.

In an SFT context, underfitting means the model is "under-performing" because it hasn't mastered the training data. This usually manifests in specific task types where the model’s outputs are either incorrect, irrelevant, or significantly below the quality of the training labels. Common culprits are:
- Data Issues: Either you don't have enough samples, or the samples you have are noisy/low quality.

- Task Difficulty: The instructions or logic required are too complex for the model's current size or capability.

- Hyperparameter "Mismatches": Wrong learning rates, too few training rounds (epochs), or improper gradient accumulation settings.

To fix the problem, you first have to figure out where the model is failing.

- Test on Training Data: If the model can't even correctly answer questions it was just trained on, it simply hasn't "learned" the data yet.

- The "Knowledge Gap" Check: SFT is great at teaching style and instructions, but it's bad at injecting massive amounts of new facts. If the model is failing a task, check if the base model even knows what that is. Ask the pre-trained base model to "continue writing" about the topic. If it generates gibberish, no amount of SFT will fix it—the knowledge wasn't there to begin with.

- Spot Checks: Manually review samples. Look for "scrambled logic" or sentences that violate natural language patterns (like weird word orders). If the logic is fine but the answer is wrong, it might be a data volume problem.

If you've confirmed the model is underfitting, try these four strategies in order:
1. Adjust Training Dynamics. Increase the epochs, or if the loss curve is dropping too slowly, increase the LR to help the model escape "local optima" (getting stuck in a mediocre state). If the loss is wildly fluctuating, decrease it to stabilize learning.

2. Data-Level Upgrades. If a specific task is underperforming, take your best examples of that task and train on them for more rounds than the rest of the data. Or, simply add more high-quality data for that specific task type.

3. "Dumb Down" the Prompt (Simplification). Provide more background information in the prompt so the model doesn't have to rely solely on its internal memory. Breaking a massive, complex prompt into smaller, more manageable sub-prompts (decomposition) is often the key to moving from underfitting to success.

4. Prompt Injection. Try embedding some of the expected "target tokens" directly into the prompt. By giving the model a "running start," you lower the complexity of the generation task.

With Overfitting, the model gives "hallucinated" answers that look like training data (e.g., insistently claiming the capital of the US is Beijing because of a data error). If a model makes a mistake, ask related questions to see if the error is widespread or localized to a specific token pattern. Don't rely on global fixes like Dropout or Weight Decay. Instead, retrieve and delete the specific "poisoned" data causing the local overfitting.

Catastrophic Forgetting is the "brain drain" of the AI world. As the model learns new, specific tasks, it starts to overwrite the broad, general knowledge it picked up during its massive pre-training phase. The root cause is when we update parameters to find the "optimum" for a new task, we often drift away from the original knowledge. Sometimes, the new goal actually contradicts the old logic, forcing the model to "discard" the past. WHen this happens, you get a model that is great at its new job but suddenly fails at basic tasks it used to be a pro at. The fix is to freeze part of the parameters, to use smaller learning rate so that the model doesn't overreact to the new data, or use progressive tuning to gradually introduce tasks to let the model adapt without shock.

Increased Model Bias happens when SFT accidentally turn a model into a "yes-man" for whatever biases were hidden in your fine-tuning data. If your tuning data is unrepresentative or the labels are unfair, the model will pick up those bad habits. It can also over-adapt, focusing so much on a specific pattern that it ignores all other valid possibilities. The fix is to rigorously check for diversity and fairness before you even start training by data audits, to run specific bias detection tests after SFT, and to use technical "nudges" to pull the model back to a neutral center in model calibration.

Decreased Generalization is another common side effect of SFT is turning the model into a "one-trick pony." It becomes so specialized that it loses its "common sense" when facing data it hasn't seen before. This is because the model gets too comfortable in the narrow distribution of the SFT dataset and stops looking for broader patterns. The model's "flexibility" disappears. It works perfectly in the lab but breaks the moment a real-world user asks a question in a slightly weird way. To fix it, train on several related tasks at once so the model has to find a general logic that works for all of them. Don't just give the model "perfect" data; give it a wide variety of scenarios.

SFT can accidentally turn a model into a "yes-man" for whatever biases were hidden in your fine-tuning data.

SFT is a deeply empirical process. When a model fails, the most effective debugging method is to:

- Compare the SFT response with the base model's completion.

- Identify exactly which token the error begins at.

- Determine if the failure is a Format Mismatch or a Content Error.

- Avoid over-relying on theoretical "interpretability" papers—focus on data retrieval and iterative testing to find the root cause.

While Supervised Fine-Tuning (SFT) is a fundamental requirement for building modern AI, it isn't a "magic wand." In fact, if used improperly, it can actually make a model "dumber"—especially when it comes to high-difficulty tasks like handling negations, refutations, or complex logic. The fundamental mechanism of SFT is to maximize the probability of the "correct" token. However, because SFT only provides positive examples, it never tells the model what not to say.  If you train a model that "ABCD $\rightarrow$ E" is a correct path, the model might "cleverly" assume that "ACD $\rightarrow$ E" is also acceptable. Because it hasn't been taught that certain sequences are wrong, it over-generalizes and begins to hallucinate. If you try to correct a model using SFT by showing it a "negative" sentence (like "Don't say X"), the model might actually increase the probability of saying "X." It doesn't understand that you are negating the behavior; it thinks you are praising that specific sequence of words.

Because of the unidirectional nature of autoregressive LLMs (often called Causal Masking), the model can only see what comes before a token, never what comes after. Imagine a sentence that starts with a false statement but ends with a refutation (e.g., "Pluto is a planet, a statement that is now scientifically outdated."). While training on that sentence, the model sees "Pluto is a..." and learns that "planet" is a high-probability next word. It cannot "look ahead" to the end of the sentence to see that the statement is being refuted. This makes SFT notoriously bad at handling irony, sarcasm, or complex logical "reversals" where the meaning of the beginning of the sentence changes based on the end.

SFT is a microscopic process. It focuses on the next token, not the whole message. It lacks the structural understanding to evaluate whether a paragraph makes sense as a whole. This is where Reinforcement Learning from Human Feedback (RLHF) steps in. RLHF can use Process Reward Models (PRM) to score an entire sequence at once. In RLHF, a human (or reward model) can say, "This whole sentence is wrong," and that signal travels backward to correct the logic. SFT simply cannot do this because it has no concept of a "whole-sentence score."
### 4.1.3 Multi-Turn Dialogue

When training Large Language Models (LLMs) on multi-turn dialogues—typically during the Supervised Fine-Tuning (SFT) phase—a key technical decision is whether to calculate the training loss for every response in the conversation or only for the final one. The choice depends on your specific training goals and how you want the model to prioritize information.

If the goal is context understanding, Loss is calculated for every response (Assistant turn) within the dialogue history. This helps the model better internalize conversational flow and maintain consistency over long interactions. If the goal is final answer quality,  loss is calculated exclusively for the very last response in the conversation. This prioritizes the model's ability to provide a high-quality "end result," which can be more computationally efficient for specific tasks.

In practice, we control which tokens contribute to the model's weight updates using a Loss Mask. This tells the training script to effectively "ignore" certain parts of the input sequence. When using PyTorch’s CrossEntropyLoss (the standard loss function for LLMs), there is a specific parameter called ignore_index. By convention, this is set to -100. Any label in your training data assigned the value of -100 will be skipped by the loss function. It will not affect the gradients or influence the model's learning for those specific positions.
### 4.1.4 PEFT
As Large Language Models (LLMs) continue to scale from millions to billions of parameters, the traditional way we "train" or adapt them has hit a major wall. Based on the provided images, here is a structured breakdown of why Parameter-Efficient Fine-Tuning (PEFT) has become essential. 

In the early days of AI, models were small enough (millions of parameters) that Full Fine-Tuning—updating every single weight in the model—was easy and accessible. However, modern models (like the GPT series or Llama) are so massive that full fine-tuning presents two massive hurdles:
- Extreme Computational Cost: Full fine-tuning requires massive amounts of VRAM (Video RAM) and processing power. Most developers and researchers only have access to "consumer-grade" hardware (like a single high-end gaming GPU), which simply cannot handle the memory load of a full billion-parameter model update.
- Sluggish Training Speed: Because the system has to calculate and update billions of gradients, the training process is incredibly slow.

If you use Full Fine-Tuning to adapt a model for five different tasks (e.g., translation, coding, sentiment analysis, etc.), you end up with five different versions of the entire model. PEFT solves this by only updating a tiny fraction of the parameters, meaning you only need to store the small "delta" (the changes), which are often just a few megabytes.

PEFT was developed to bridge the gap between high-performance adaptation and low-resource requirements. You don't need to touch all the parameters. By only tuning a small subset or adding auxiliary layers, you drastically reduce the VRAM and storage footprint. State-of-the-art (SOTA) PEFT techniques, like LoRA, have proven that you can achieve performance levels nearly identical to Full Fine-Tuning while only training a fraction of the parameters. PEFT allows massive models to be fine-tuned on consumer-grade hardware, "democratizing" AI development so it isn't restricted to giant tech companies with supercomputers.
#### 4.1.4.1 BitFit

BitFit (Binary Task Fine-tuning) is an exceptionally sparse and efficient strategy within the Parameter-Efficient Fine-Tuning (PEFT) family. While most methods focus on adding new layers (Adapters) or modifying weight matrices (LoRA), BitFit takes a minimalist approach by only modifying the bias terms of a pre-trained model.

Standard Full Fine-Tuning is powerful but comes with significant baggage. Updating every parameter in a multi-billion parameter model creates a massive, unique file for every single task you want to solve. This makes deployment and maintenance nearly impossible as the number of tasks grows.

BitFit was designed to meet four "ideal" conditions for fine-tuning:

- Performance: Matching the accuracy of full fine-tuning.

- Efficiency: Modifying only a tiny fraction of the parameters.

- Deployment: Enabling data to arrive in "streams" for efficient hardware use.

- Consistency: Keeping the modified parameter types consistent across different downstream tasks.

In a Transformer model, most parameters are stored in the weight matrices ($W$) of the Attention and MLP layers. BitFit freezes all of these weights and only updates the bias terms ($b$). Specifically, it targets the biases in:

- Attention Modules: Calculations for Query ($Q$), Key ($K$), and Value ($V$), as well as the biases used when merging multiple attention heads.

- MLP Layers: The feed-forward networks between attention blocks.

- Layer Normalization: The biases used to scale and shift activations.

In models like BERT-Base or BERT-Large, the bias parameters account for only $0.08\%$ to $0.09\%$ of the total parameter count. Despite this tiny footprint, the research shows impressive results. It significantly outperforms "Frozen" methods (where no parameters are tuned) and stays within reach of Full Fine-Tuning. Research indicates that not all biases are created equal. The most critical changes happen in the Query biases and the Intermediate Feed-Forward (FFN) layers (where the dimension expands from $N$ to $4N$). If you freeze either of these specific bias groups, the model's performance drops sharply.

Example code:
```python
import torch
from transformers import AutoModelForSequenceClassification

def apply_bitfit(model):
    """
    Freezes all parameters in the model except for the bias terms.
    """
    trainable_params = 0
    all_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        
        # BitFit Logic: Only unfreeze parameters with 'bias' in their name
        # We also typically keep the classification head (classifier) trainable
        if "bias" in name or "classifier" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"Total Parameters: {all_params:,}")
    print(f"Trainable (BitFit) Parameters: {trainable_params:,}")
    print(f"Percentage Trainable: {(100 * trainable_params / all_params):.4f}%")
    
    return model

# 1. Load a standard pre-trained model
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Apply the BitFit strategy
model = apply_bitfit(model)

# 3. Setup the optimizer 
# Only pass the parameters that actually require gradients
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=1e-4
)
```

#### 4.1.4.2 Hard Prompt Tuning
The fundamental idea of Prompt-Tuning is to convert a downstream task (like sentiment analysis) back into a pre-training task (like fill-in-the-blank). In the past, to make a model like BERT identify emotions, we had to attach a brand-new "classifier head" (an MLP layer) on top of it. This required a significant amount of labeled data to train that new layer from scratch. Prompt-Tuning avoids this by "tricking" the model into using its original skills to solve the new problem. 

Using the BERT sentiment analysis example, the process follows three distinct steps:

- Template Construction. We take the original input and wrap it in a "Template" that includes a [MASK] token, like "[CLS] I like the Disney films very much. [SEP] It was [MASK]."

- Reusing the MLM Classifier. Instead of using a new classification layer, we use the model’s original Masked Language Model (MLM) head. The model looks at the mask and predicts the probability distribution of every word in its vocabulary that could fill that hole.

- Label Word Verbalizer (Mapping). Since we only care about specific categories (Positive vs. Negative), we create a "Verbalizer" to map words to labels. For example, if the model predicts "great" or "amazing" $\rightarrow$ Label as Positive.

GPT-3 took this concept a step further with In-Context Learning (ICL) and Demonstration Learning. This is the "Zero-shot" or "Few-shot" approach. You don't update any model parameters at all. You simply provide the task description in the prompt. You give the model examples (demonstrations) within the prompt to show it how to behave.

There are two types of Prompt Tuning. Hard Prompting (Discrete) is what is shown in the examples above—manually or automatically searching for the best text-based templates that humans can read. Soft Prompting (Continuous) is a more advanced PEFT (Parameter-Efficient Fine-Tuning) method. Instead of using real words, we "tune" specific mathematical vectors (Prompt Embeddings) that are prepended to the input. While humans can't read these "virtual tokens," they are often much more effective at guiding the model.

#### 4.1.4.3 Prefix Tuning

Instead of retraining an entire massive model, Prefix-Tuning focuses on adding a small, learnable "hint" that guides the model to perform specific tasks. Before Prefix-Tuning, researchers relied on Prompt Tuning, which involved designing text-based templates (called "Hard Prompts"). For example, to summarize a text, you might manually add the words "In summary:" at the end. However, this had several major flaws:

- High Sensitivity: Small changes in the manual template could cause huge drops in model performance.

- Suboptimal Results: Human-designed words (discrete tokens) are often not the "perfect" mathematical instructions for a model.

- Manual Labor: It is incredibly difficult and time-consuming for humans to design the "optimal" prompt for every different task.

Prefix-Tuning replaces human words with Virtual Tokens (also known as Soft Prompts or Continuous Prompts). Implicit vs. Explicit: While a "Hard Prompt" is an explicit hint humans can read, a "Prefix" is an implicit, learnable hint consisting of continuous vectors. Instead of picking from a fixed dictionary of words, Prefix-Tuning searches for the best "instruction" in a much larger, continuous mathematical vector space. These virtual tokens don't correspond to any real word in a natural language; they are purely optimized for the model's performance.

<p align="center">
<img width="397" height="307" alt="3f87fc48-e431-4229-979f-33135428f68f" src="https://github.com/user-attachments/assets/aeb05816-77aa-46d9-b33d-e0fe7110ab13" />
</p>

Prefix-Tuning changes its structure depending on the type of model being used:

- Decoder-Only Models (e.g., GPT-2): The prefix is added only at the very beginning of the sequence. Formula: $z = [PREFIX; x; y]$ (where $x$ is the input and $y$ is the output).

- Encoder-Decoder Models (e.g., BART): Different prefixes are added to the start of both the encoder and the decoder. The encoder prefix guides the encoding of the input, while the decoder prefix guides the generation of the output. Formula: $z = [PREFIX; x; PREFIX'; y]$.

Research shows that putting the "hint" at the very beginning (Prefix) is slightly better than putting it in the middle (Infix: $[x; INFIX; y]$).

<p align="center">
<img width="498" height="193" alt="b72ff315-5d8c-4aeb-97d6-92393fdc24a8" src="https://github.com/user-attachments/assets/2305fc8e-cd4a-4467-98d5-0b23d949fc30" />

</p>

To make Prefix-Tuning stable and powerful, the authors introduced three critical "Key Points":

- Layer-wise Prefix-Tuning. Researchers found that just adding a prefix to the initial input (the embedding layer) wasn't expressive enough. To truly "steer" the model, Prefix-Tuning adds a Prefix Vector to every single Transformer block in the model. This ensures that the task-specific guidance influences every stage of the model's "thinking" process, not just the beginning. For every index $i$ in the sequence, the model checks if it falls within the "Prefix" range. If it is a prefix index ($i \in P_{idx}$), the activation $h_i$ is copied directly from a specialized parameterized matrix ($P_\theta$). if it is not, $h_i$ is computed using the standard Language Model (LM) layers. Because every subsequent calculation depends on the prefixes to its left, these "deep prompts" influence the hidden states at every level of the model's architecture.

<p align="center">
<img width="310" height="182" alt="ffc13c36-e237-4b79-a404-a76e2f157d55" src="https://github.com/user-attachments/assets/61fc7e7e-74fa-40c0-85bc-ae9a5861ccaf" />
</p>

- MLP Projection (Reparameterization). Directly updating the prefix vectors during training can lead to instability and performance loss. To fix this, the authors use a Multi-Layer Perceptron (MLP) to "reproject" the parameters. They decompose the large matrix into a smaller matrix ($|P_{idx}| \times k$) followed by a larger MLP ($k \times dim(h_i)$). During training, the gradients update the MLP. Once training is finished, the MLP is discarded. Only the resulting Prefix parameters are kept for the actual model run, meaning there is no extra "weight" added to the model during use.

<p align="center">
<img width="400" height="162" alt="4f11aebb-8999-4d95-8b2b-eb386e1f3efe" src="https://github.com/user-attachments/assets/2940688e-3471-48b7-87d9-6f2a69b8558d" />
</p>

- The "Length" of the prefix (how many virtual tokens you use) is a critical balance between performance and the model's "context window."  The authors found that a default prefix length of 10 virtual tokens is often the "sweet spot." Because the prefix is so short, the total number of new parameters is incredibly small—roughly 0.1% of the original model's parameters. The length of the prefix directly impacts the model's ability to handle long documents; if the prefix is too long, it leaves less room for the actual input text.

Example code:
```python
import torch
import torch.nn as nn
from transformers import AutoModel

class PrefixEncoder(nn.Module):
    def __init__(self, config, num_prefix_tokens=10, intermediate_dim=512):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_size = config.hidden_size

        # KEY POINT 2: Reparameterization (Small Matrix + MLP)
        # This is the 'raw' virtual token matrix (|P_idx| x k)
        self.embedding = nn.Embedding(num_prefix_tokens, intermediate_dim)
        
        # The MLP that projects 'k' to the full hidden states
        # We project to (num_layers * 2 * hidden_size) to provide Key and Value for every layer
        self.mlp = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, self.num_layers * 2 * self.hidden_size)
        )

    def forward(self):
        # Generate the prefix vectors
        tokens = torch.arange(self.num_prefix_tokens).to(self.embedding.weight.device)
        past_key_values = self.embedding(tokens) # [num_tokens, intermediate_dim]
        past_key_values = self.mlp(past_key_values) # [num_tokens, num_layers * 2 * hidden_size]
        
        # KEY POINT 1: Layer-wise distribution
        # Reshape so we have a distinct 'Prefix' for every single layer
        return past_key_values.view(self.num_layers, 2, self.num_prefix_tokens, -1)

# Usage Logic
model = AutoModel.from_pretrained("gpt2")

# FREEZE BASE MODEL:
for param in model.parameters():
    param.requires_grad = False

# Add our Prefix Tuning logic
prefix_config = model.config
prefix_tuning_layer = PrefixEncoder(prefix_config)
```

Prefix-Tuning is a powerful "Plug-and-Play" method that excels in efficiency and multi-task management. It has extreme parameter efficiency, high training efficiency (Because the number of trainable parameters is a tiny fraction of the whole model, the backpropagation and gradient update process is significantly faster, saving both time and computational energy), superior for multi-task scenarios (You only need to store and swap out a tiny task-specific "Prefix Vector" for each new objective, making it ideal for systems that require frequent task switching), and strong portability and transferability (The prefixes can be viewed as modular knowledge that can potentially be transferred or adapted across various related tasks).

Despite its efficiency, Prefix-Tuning has specific "bottlenecks" that engineers must carefully manage. The length of the virtual tokens is a "Goldilocks" problem. If it is too short, the prefix may not provide enough "guidance" or representational power for the model to complete complex tasks effectively. If it is too long, increasing the length consumes more of the model’s limited context window and increases training costs. Unlike Full Fine-Tuning, Prefix-Tuning only modifies the input features (activations) at each layer. It cannot alter the model's internal "logic" or pre-trained knowledge stored in the hidden parameters. On certain specialized tasks, its performance may still lag behind full parameter fine-tuning because the model's "core" cannot be reshaped. On specific, highly complex tasks, the "Soft Prompt" approach might not be as robust as methods that allow for internal weight adjustments (like LoRA or Full FT).

Prefix-Tuning is not a "one-size-fits-all" solution, but it has a clear home in specific domains:

- Primary Use Case: Text Generation Tasks. It is exceptionally well-suited for Machine Translation, Text Summarization, and Dialogue Generation.
- Secondary Use Case: Some classification tasks.
- High-Value Scenarios: It performs best in multi-task and cross-domain applications, where the ability to quickly transfer pre-trained capabilities to a new task with minimal storage overhead is a massive competitive advantage.
  
#### 4.1.4.4 Soft Prompt Tuning
In Soft Prompt Tuning, instead of using real words, we use continuous embedding vectors (Virtual Tokens). These "Soft Prompts" are learned through backpropagation. Humans can't read them, but the model understands them perfectly as optimized instructions. The core technical workflow involves "tricking" the model into using its original skills for a new purpose.

Given an input sequence $X$, we prepend a series of $m$ trainable prompt tokens $P$: $[P; X]$ or $[p_1, p_2, \dots, p_m, x_1, x_2, \dots, x_n]$. The dimension of the prompt vectors $d$ is set to match the model's embedding layer dimension. These virtual tokens aren't just random noise. They can be initialized in two ways:
- Random Initialization: Starting from scratch.
- Manual Initialization: Initializing the vectors from existing word embeddings (like "summarize" or "translate") to give the model a "head start."

The most critical part of Prompt-Tuning training process is what stays still and what move. The entire pre-trained model ($\theta$) is frozen. Not a single original weight is changed. Only the parameters of the prompt vectors ($\theta_P$) are updated using a task-related Loss function ($\mathcal{L}$). The mathematical goal is:

$$\min_{P} \mathcal{L}(f(X_{ext}; \theta), y)$$

While Prompt-Tuning and Prefix-Tuning sound similar, there is key distinction. Prefix-Tuning injects virtual tokens into every layer of the Transformer and often uses an MLP for adjustment. Prompt-Tuning is a simplified version of Prefix-Tuning. It only prepends tokens to the input embedding layer (the very first block). Because it doesn't touch the internal layers, it uses significantly fewer parameters than Prefix-Tuning.

Prompt-Tuning becomes more effective as the model gets bigger. For a massive model like T5-XXL (11 Billion parameters), a full fine-tuning would require 11B updates. Prompt-Tuning achieves nearly the same performance by tuning only 20,480 parameters. Experiments prove that automatically generated soft prompts perform nearly as well as full model tuning and exceed the performance of human-designed hard prompts. The task switching is also rapid. Since you only need to save the tiny prompt file (the "delta") for each task, you can switch the model from "Translator" to "Summarizer" instantly just by swapping the input prefix.

Prompt-Tuning allows for a unique optimization called Prompt Ensembling. In a single Batch, you can train/query the same task using different prompts (asking the same question in different ways). This is mathematically similar to training different models but at a much lower cost than traditional model ensembling.

There are constraints and limitations for this method:
- Dependency on Prompt Length: The length $m$ of the prompt is a key hyperparameter that needs tuning. Different tasks have different needs.
- Limited on Small Models: On smaller pre-trained models, Prompt-Tuning may not perform as well as full parameter fine-tuning. It truly shines once you cross the 10 Billion parameter threshold.
- Adaptation for Complex Tasks: Some extremely complex tasks may still require more internal model modification (like LoRA) to achieve peak accuracy.

#### 4.1.4.5 P-Tuning V1 & V2

P-Tuning is famous for the research paper title "GPT Understands, Too," because it successfully boosted the performance of GPT-style models in Natural Language Understanding (NLU) tasks, where they previously lagged behind BERT. 

Traditional prompting relies on Hard Prompts (human-designed text templates). P-Tuning shifts the focus from "searching for words" to "optimizing embeddings." Instead of using real natural language tokens (like "The capital of..."), P-Tuning inserts continuous, differentiable "virtual tokens" into the input. These are not words humans can read; they are purely mathematical vectors optimized by the model. Unlike Prefix-Tuning (which only adds tokens at the beginning), the position of virtual tokens in P-Tuning is optional—they can be inserted as a prefix, a suffix, or even in the middle of the input. The goal is to replace real tokens in human-designed templates with differentiable virtual tokens to better "awaken" the model's potential.

A unique feature of P-Tuning (specifically v1) is the use of a Prompt Encoder to handle the virtual tokens before they enter the LLM. Pre-trained word embeddings in LLMs are highly discrete. If you initialize virtual tokens randomly, the model might easily get stuck in a "local optimum" (a sub-optimal solution). Virtual tokens should be related to each other. By passing them through a Bi-directional LSTM followed by an MLP (Multi-Layer Perceptron), the system creates a "chain of logic" between the tokens, allowing for faster convergence and better results. This setup is a form of parameter reparameterization—training the LSTM/MLP to output the "perfect" embeddings for the task.

The method evolved to handle more complex tasks and different model scales:

- P-Tuning v1. Virtual tokens are added only to the input embedding layer. It is less effective on smaller models (under 10B parameters) and struggles with complex sequence-labeling tasks.

- P-Tuning v2. Virtual tokens are added to every single Transformer layer (Layer 1, Layer 2, ..., Layer N). It is universal across different model scales and tasks. It achieves performance comparable to full fine-tuning while keeping the base model frozen.

To construct a P-Tuning workflow, first define a prompt $P$ of length $m$. These tokens have a dimension $d$ equal to the model's embedding dimension. Then, concatenate the prompt $P$ with the original input $X$:

$$X_{ext} = [P; X] \in \mathbb{R}^{(m+n) \times d}$$

In the training process, the entire pre-trained model parameters ($\theta$) are frozen. Only the parameters of the virtual tokens ($P$) or the Prompt Encoder are updated by minimizing the loss function:

$$\min_{P} \mathcal{L}(f(X_{ext}; \theta), y)$$

<p align="center">
<img width="904" height="204" alt="0bea350a-8c57-4109-9a1e-e018f4d41210" src="https://github.com/user-attachments/assets/7f47d103-aafd-4ef7-9b83-eb3b6db60a7d" />
</p>

P-Tuning v2 is a universal and scalable Parameter-Efficient Fine-Tuning (PEFT) method designed to overcome the critical limitations of its predecessors, Prompt-Tuning and P-Tuning v1. While earlier methods proved that "GPT understands, too," they struggled with smaller model scales and complex NLU (Natural Language Understanding) tasks.

The original P-Tuning and Prompt-Tuning methods faced three primary "bottlenecks" that hindered their widespread adoption:
- Lack of Scalability for Small Models: When model parameters were between 100M and 1B, there was a massive performance gap between prompt-based tuning and full fine-tuning. These methods only reached parity when models exceeded 10B parameters.
  
- Lack of Task Universality: While effective for simple classification, they struggled with "hard" NLU tasks like Sequence Labeling (e.g., Named Entity Recognition) and complex reasoning.

- Limited Parameter Impact: Because prompts were only inserted at the input embedding layer, their influence on the model’s deeper layers was indirect and the number of trainable parameters was extremely small (around 0.01%).

The fundamental change in P-Tuning v2 is the transition from "Input-layer prompting" to "Deep Prompt Tuning." Instead of just prepending virtual tokens to the input, P-Tuning v2 injects Prompt tokens into every single layer (Layer 1 to Layer N) of the Transformer architecture. Unlike Prefix-Tuning (where prompts are often linked via an MLP), the prompts at each layer in P-Tuning v2 are independent and not calculated from the previous layer. By adding prompts to deeper structural levels, the "task-specific hint" has a much more direct and powerful influence on the model's final predictions.

P-Tuning v2 introduces several strategic shifts to make the model more robust and universal:
- Increased Trainable Parameters. By utilizing prompts across all layers, the ratio of trainable parameters increases from the tiny 0.01% of v1 to a more substantial 0.1% to 3%. This provides enough "tuning space" to handle complex tasks while remaining highly efficient
- Removal of Reparameterization (The Encoder). In v1 and Prefix-Tuning, complex encoders (like LSTM or MLP) were used to stabilize training. In v2, the authors found that these encoders actually provided very little improvement and could even hurt performance on smaller models. Thus, v2 removes the heavy reparameterization.
- Task-Specific Prompt Length. Research showed that prompt length is a crucial hyperparameter that depends on task complexity. For simple Tasks (e.g., Sentiment Analysis), a short prompt of ~20 tokens is sufficient. Complex Tasks (e.g., Reading Comprehension) requires a much longer prompt of ~100 tokens to achieve optimal performance.
- Return to Traditional Classification Heads. Instead of using a Verbalizer (mapping words to labels), P-Tuning v2 returns to the traditional approach of using a randomly initialized Classification Head (Linear Head) applied over the tokens. This makes the method universal across all NLU tasks, including sequence labeling.
- Multi-Task Learning Strategy. To overcome the optimization difficulty of randomly initialized prompts, v2 suggests pre-training on multi-task prompts before adapting to specific downstream tasks. This "shared knowledge" serves as a better starting point for the model.

P-Tuning v2 allows models under 10B parameters to finally achieve performance comparable to full fine-tuning. In tasks like RTE (Recognizing Textual Entailment), P-Tuning v2 (especially when applied to BERT) significantly outperforms standard full fine-tuning. It is the first prompt-based method to effectively tackle sequence labeling and complex NLU benchmarks where previous versions failed.

<p align="center">
<img width="901" height="199" alt="42c8c3fc-83e2-4dca-9486-542c9ad04a97" src="https://github.com/user-attachments/assets/1acd5581-caad-4cf0-8f7b-4ad84c94cbfd" />
</p>

P-Tuning v1 example code:
```python
import torch
import torch.nn as nn

class PTuningV1Encoder(nn.Module):
    def __init__(self, num_virtual_tokens, embedding_dim, hidden_dim=128):
        super().__init__()
        # 1. Raw Virtual Token IDs
        self.embedding = nn.Embedding(num_virtual_tokens, embedding_dim)
        
        # 2. KEY POINT: LSTM Reparameterization (Bi-directional)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # 3. MLP Projection Head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, embedding_dim)
        )

    def forward(self):
        # Generate the continuous embeddings for the prompt
        indices = torch.arange(self.embedding.num_embeddings).unsqueeze(0).to(self.embedding.weight.device)
        input_embeds = self.embedding(indices) # [1, tokens, dim]
        
        # Pass through LSTM to establish token relationships
        output, _ = self.lstm(input_embeds)
        
        # Project back to model dimension
        prompt_embeddings = self.mlp_head(output)
        return prompt_embeddings

# --- Usage Logic ---
# To train:
# 1. Get original embeddings of X: inputs_embeds = model.embeddings(input_ids)
# 2. Get prompt embeddings: prompt_embeds = encoder()
# 3. Concatenate: full_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
# 4. Pass full_embeds to the frozen model.
```

P-Tuning v2 example code:
```python
class PTuningV2Prompt(nn.Module):
    def __init__(self, num_layers, num_virtual_tokens, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        
        # KEY POINT: Independent parameters for EVERY layer
        # Each layer needs a set of Key and Value vectors
        # Shape: [num_layers, 2 (K and V), num_tokens, hidden_size]
        self.prompt_parameters = nn.Parameter(
            torch.randn(num_layers, 2, num_virtual_tokens, hidden_size)
        )

    def forward(self, batch_size):
        # Expand for the current batch size
        # Result shape: [num_layers, batch_size, 2, num_heads, tokens, head_dim]
        # (Note: Requires reshaping based on the specific model's head count)
        prompts = self.prompt_parameters.unsqueeze(1).expand(-1, batch_size, -1, -1, -1)
        
        # Construct the 'past_key_values' structure for HuggingFace models
        past_key_values = []
        for i in range(self.num_layers):
            # Extract Key and Value for this specific layer
            layer_k_v = (prompts[i, :, 0], prompts[i, :, 1])
            past_key_values.append(layer_k_v)
            
        return tuple(past_key_values)

# --- Usage Logic ---
# model = AutoModel.from_pretrained(name)
# freeze_model(model) # Set requires_grad=False
# p_v2 = PTuningV2Prompt(config.num_layers, 100, config.hidden_size)
#
# outputs = model(input_ids=input_ids, past_key_values=p_v2(batch_size))
```
**Advantages**
- High Parameter Efficiency: Unlike traditional methods that require updating the entire model, P-tuning only requires training a small number of prompt vector parameters. This significantly reduces the computational overhead and resources needed for fine-tuning.

- Better Task Adaptability: Because P-tuning uses continuous prompt vectors, its representational power is much stronger than manually designed "hard" (discrete) prompts. This allows the model to adapt to different tasks with much higher flexibility.
  
- Broad Versatility: It is not limited to a single type of task. It can be applied effectively across a wide range of Natural Language Processing (NLP) scenarios, including classification, sequence labeling, and content generation.

**Disadvantages**
- Critical Prompt Length Selection: The performance of P-tuning is highly sensitive to the length of the prompt vector. Finding the "optimal length" requires significant manual tuning, and what works for one task may not work for another.
  
- Dependency on Task Data: The effectiveness of P-tuning is heavily tied to the specific data of the task. In certain highly specialized or complex scenarios, P-tuning may still underperform compared to Full Parameter Fine-Tuning, as it cannot modify the model's internal weights.

#### 4.1.4.6 Adapter Tuning

Adapter Tuning is a highly effective method within the Parameter-Efficient Fine-Tuning (PEFT) family. Its primary goal is to adapt massive pre-trained models to specific downstream tasks without the massive computational and storage costs associated with "Full Fine-Tuning." Instead of modifying the model's original "knowledge" (its weights), Adapter Tuning inserts small, trainable modules into the existing architecture.

The logic of Adapter Tuning is straightforward:
1. Freeze the Base Model: Every original weight in the Transformer layers is locked.
2. Insert Adapters: Small, task-specific "Adapter" modules are inserted into every layer of the model.
3. Train Only the Adapters: During fine-tuning, the gradients only update the Adapter parameters, the LayerNorm layers, and the final classification Head.

The Adapter module itself is designed to be extremely "lean" using a bottleneck structure. It consists of four main parts:
1. Down-Projection ($W_{down}$): It takes the high-dimensional input (dimension $d$) and projects it down to a much lower dimension ($m$). Usually, $m \ll d$. This is the key to reducing the parameter count.
2. Non-linearity: A non-linear activation function (like ReLU or GeLU) is applied to help the model learn complex patterns.
3. Up-Projection ($W_{up}$): The low-dimensional features are projected back up to the original dimension ($d$).
4. Skip Connection (Residual): The original input $h$ is added back to the output of the up-projection.

The Mathematical Formula:

$$h \leftarrow h + f(hW_{down})W_{up}$$

Skip Connection ensures that even if the Adapter's weights are initialized near zero, the model still functions as an "identity mapping" (the original information passes through unchanged). This provides immense training stability.

Based on the research, each Transformer layer typically receives two Adapter modules:

- Location 1: Inserted immediately after the Multi-Head Attention projection.
- Location 2: Inserted immediately after the second Feed-Forward (FFN) layer.

By placing them here, the Adapters can "intercept" and refine the most critical information-processing steps of the Transformer.

Adapter Tuning adds only 0.5% to 5% of the original model's parameter count. Despite the tiny footprint, it achieves performance within 1% of a full fine-tuning model. Since you only save the Adapters, task-specific files are tiny, making it easy to store dozens of "specialized" models for one base model. It allows for rapid transfer of capabilities to new downstream tasks and cross-domain applications.

<p align="center">
<img width="272" height="202" alt="80c535dd-6b84-42f1-8aa2-67278d8e49ec" src="https://github.com/user-attachments/assets/f674d7f1-ac1c-4d34-95f3-8c6ef6973795" />
</p>

The Houlsby Adapter is the "classic" design. It treats the adapter as a series of checkpoints that the data must pass through inside each Transformer layer. It inserts two adapter modules into every single Transformer block. In standard setups, this adds about 3.6% to the model's total parameters per task. By placing adapters in these "serial" locations, the model forces the information to be refined and transformed at the most critical points of the layer’s logic.

Unlike the Houlsby version, the Parallel Adapter doesn't wait in line. It works side-by-side with the main model layers. The adapter is treated as a parallel sub-network that runs at the same time as the Transformer layer. Instead of the layer feeding into the adapter, both paths produce outputs independently. These outputs are then merged using weighted averages or summation. This often leads to smoother training and can be more efficient for certain hardware parallelizations, as the adapter doesn't "block" the main computation path.

If the 3.6% parameter increase of a Houlsby adapter is still too much, the Compacter is the solution. It is a hyper-optimized variant designed to reduce the adapter's own storage footprint. The Compacter borrows a trick from LoRA. It breaks down the internal weight matrices ($W_{up}$ and $W_{down}$) into the product of even smaller, low-rank matrices. By using this decomposition, the Compacter can achieve the same performance as a standard adapter while training a significantly smaller fraction of parameters (often under 0.1%).

AdapterFusion is an advanced architectural framework designed to integrate knowledge from multiple tasks into a single model without the negative side effects of traditional training methods. It builds upon standard Adapter Tuning to provide a "non-destructive" way to combine expertise across different domains. Traditional methods for integrating knowledge from multiple tasks (Multi-task Learning) face several critical bottlenecks:
- Sequential Fine-tuning: Training a model on Task A then Task B. This often requires knowing the "perfect" order of tasks beforehand and carries a high risk of Catastrophic Forgetting (losing knowledge of Task A while learning Task B).
- Multi-task Learning (MTL): Training on all tasks simultaneously. This is difficult to balance because different tasks can interfere with each other, especially if datasets vary greatly in size.
- The Adapter Solution: Standard Adapter Tuning solves the forgetting problem by using task-specific modules, but it doesn't have a built-in way to let Task B "borrow" useful information from the Adapter already trained for Task A.

AdapterFusion was created to bridge this gap, allowing a model to dynamically identify and combine the most relevant knowledge from various task adapters. AdapterFusion splits the learning process into two distinct phases to ensure stability and efficiency.

1. Stage 1: Knowledge Extraction. In this stage, the model focuses on learning individual tasks in isolation. There are two training options. For Single-Task Adapters (ST-A), task-specific Adapter modules are inserted into the pre-trained model for each of the $N$ tasks. For Multi-Task Adapters (MT-A), multiple tasks are optimized jointly via multi-task learning.

2. Stage 2: Knowledge Composition (The Fusion Phase). Once the task-specific adapters are trained, they are "fused" together to solve a target task $m$. The parameters of the pre-trained model and the parameters of the $N$ adapters from Stage 1 are completely frozen. A new set of parameters, called AdapterFusion, is introduced to learn how to combine the $N$ adapters. The goal is to learn a parameterized "mixer" that determines which adapters are most useful for the current input context.

The core of AdapterFusion is an Attention mechanism that acts as a dynamic selector. It is placed above the adapters within each Transformer layer. AdapterFusion uses the relationship between the model's current state and the adapter outputs to decide weighting:
- Query ($Q$): This is the output of the pre-trained Transformer’s internal sub-modules (like the Feed-Forward or Attention layer). It represents the "current context."
- Key ($K$) and Value ($V$): These are derived from the outputs of the $N$ task-specific adapters.
The model calculates a dot product between the Query and all the Keys. This is passed through a SoftMax function to generate weights. The model assigns high weights to the adapters that are most relevant to the current input, aggregating the info into a single optimized representation.

<p align="center">
<img width="217" height="293" alt="649266c7-eff6-4998-bd7a-8c03478d0fc0" src="https://github.com/user-attachments/assets/4e79f894-5ee7-4fe1-846a-b5558ea1f47a" />
<img width="250" height="270" alt="d799cfef-2e10-4c15-a71c-a50731fa8c8f" src="https://github.com/user-attachments/assets/d558c8c4-e3e3-48dd-8653-15aaf06b4cef" />
</p>

Comparative experiments show that AdapterFusion outperforms both Full Fine-tuning and standard Adapter Tuning in most scenarios. By splitting training into Knowledge Extraction and Knowledge Composition, it effectively solves the issues of catastrophic forgetting, inter-task interference, and training instability. It allows for a shared multi-task structure where adapters can be reused and combined as needed for new tasks.

While powerful, AdapterFusion also has disadvantages. Adding multiple adapters and the fusion layer increases the total parameter count of the model. Because the model must process multiple adapters and a fusion attention layer for every token, it decreases model inference performance (speed/efficiency) compared to a single-task model.

While standard Adapters are revolutionary for training, they come with a hidden cost:

- The Training Win: Adapters are 60% faster during training than full fine-tuning because they update so few parameters.

- The Inference Loss: Because you are adding extra layers to the model, adapters actually make the model 4% to 6% slower during inference (forward pass).

To dynamically and efficiently remove adapters to reduce the parameter count and increase efficiency in both the backward pass (training) and the forward pass (inference), all without affecting task performance.

AdapterDrop uses two distinct methods to "slim down" the model depending on the architecture. 

1. Layer-wise Dropping (Removing Layers). This strategy focuses on the vertical depth of the Transformer. Research shows that not every layer needs an adapter to maintain high performance. By dropping adapters from the first five Transformer layers while performing inference on 8 tasks, researchers saw a 39% increase in speed. Even with multiple layers dropped, the model maintains good performance/results, proving that deep layers often carry the heavy lifting for task-specific logic.

2. AdapterFusion Pruning (Removing Breadth). When using AdapterFusion (which combines multiple task-specific adapters), the model can become very heavy because it’s looking at many adapters at once. Consider pruning most of the adapters within the Fusion layer. Experiments showed that using only two remaining adapters achieved results comparable to a full AdapterFusion model using eight adapters. This specific pruning method increased inference speed by 68%.

The technical materials emphasize a clear takeaway for AI engineers: Perform AdapterFusion pruning before actually deploying these models. It is a simple but highly effective technique that allows you to realize massive efficiency gains and hardware savings while fully maintaining the performance of the original, more complex model. AdapterDrop turns adapters from a "training-only" benefit into a lean, mean inference machine, making them viable for high-traffic, real-world applications where every millisecond of latency counts.

The MAM Adapter (short for Mix-And-Match Adapter) represents a sophisticated "unified" approach to Parameter-Efficient Fine-Tuning (PEFT). Instead of viewing methods like LoRA, Prefix Tuning, and Adapters as competing tools, the MAM architecture treats them as building blocks that can be decomposed and recombined into a single, high-performance framework. Engineers often wonder why different methods like Adapter, Prefix Tuning, and LoRA—which look structurally and mathematically different—often yield very similar results. The authors of the MAM Adapter research analyzed these methods and decomposed them into four design dimensions:
- Functional Form: The specific math used (e.g., bottleneck layers).
- Insertion Form: Whether the module is Sequential (one after another) or Parallel (working side-by-side).
- Modified Representation: Which specific part of the model is being changed.
- Composition Function: How the new weights are merged back into the original model's output.

Before building the MAM Adapter, researchers performed "Ablation Studies" to see which specific placements worked best. They discovered two critical rules:
- Parallel > Sequential: Adding an adapter in parallel (working simultaneously with the main layer) is more effective than the standard Houlsby-style sequential approach.
- Feed-Forward Networks (FFN) are the best place for Parallel Adapters.
- Attention Mechanisms (MHA) are most effectively modified by Soft Prompts (Prefix Tuning).

The final MAM Adapter is not just one module, but a dual-layered upgrade to the Transformer block. It "mixes and matches" the two most efficient sub-methods found during the study:
- Prefix Tuning (Soft Prompts): These are used to modify the Attention (MHA) layers. This is extremely parameter-efficient, requiring only about 0.1% of additional parameters to steer the model's focus.
- Parallel Adapters: These are attached specifically to the Feed-Forward Network (FFN). Research proved that modifying the FFN via a parallel bottleneck is the most powerful way to "update" the model's internal knowledge during fine-tuning.

The MAM Adapter is designed to maximize the "Accuracy-to-Parameter" ratio. In tasks like XSum (text summarization) and MT (machine translation), MAM achieves results that are nearly identical to Full Fine-Tuning (achieving a ROUGE-2 score of 21.90 vs. Full FT's 21.94). It achieves this high-tier performance using only 6.7% of the parameters required by full fine-tuning. 
#### 4.1.4.7 LoRA 

In the era of Large Language Models (like GPT-3 or Llama), models possess billions of parameters. When we want to teach these models a specific new task, the traditional approach is Full Fine-Tuning—updating every single weight in the model. This requires an enormous amount of computational power, GPU memory, and storage (because you have to save a massive new model file for every single task). Researchers discovered that heavily over-parameterized models actually reside in a low "intrinsic dimension" (intrinsic rank). This means that when a model learns a new task, the actual, meaningful changes to the weights are highly redundant and don't require updating the full matrix. You can achieve the same learning in a much smaller mathematical subspace.

LoRA's central idea is to simulate the massive weight changes of fine-tuning using Low-Rank Matrix Decomposition.Instead of directly modifying the original pre-trained weight matrix ($W$), LoRA does the following:
1. Freezes the Base Model: The original pre-trained weights ($W \in \mathbb{R}^{d \times k}$) are completely locked and do not change.

2. Adds a "Bypass" (Delta Matrix): LoRA introduces a weight update matrix ($\Delta W$) that runs parallel to the original weights.

3. Decomposes the Update: To save parameters, $\Delta W$ is decomposed into the product of two much smaller matrices: $A$ and $B$. $B \in \mathbb{R}^{d \times r}$ (A down-projection matrix). $A \in \mathbb{R}^{r \times k}$ (An up-projection matrix). $r$ is the Rank, and it is significantly smaller than the original dimensions ($r \ll d, k$).

During training, the forward pass becomes:

$$h = Wx + \Delta Wx = Wx + BAx$$

Instead of updating $d \times k$ parameters, you only update $(d \times r) + (r \times k)$ parameters. This drastically reduces the parameter count, saving massive amounts of memory and time.

How matrices $A$ and $B$ are initialized before training begins is one of the most critical parts of LoRA's success. Matrix B is initialized to all Zeros. Matrix A is initialized with a Random Gaussian Distribution ($N(0, \sigma^2)$). If $B = 0$, then $B \times A = 0$. This means at step zero of training, $\Delta W = 0$, and the model outputs exactly what the original pre-trained model ($W$) would output. This provides immense stability:
- Protects Pre-trained Knowledge: It ensures the model's foundational capabilities aren't instantly destroyed.
- Gradual Adjustment: It allows the new weights to ease into the model rather than causing massive, disruptive fluctuations early on.
- Gradient Safety: It minimizes the risk of exploding or vanishing gradients at the start of training.

If both were zero, the gradients for both would be zero, and the model wouldn't learn anything (the "symmetry problem"). Initializing $A$ randomly:
- Breaks Symmetry: It gives the model multiple distinct mathematical "directions" to explore during optimization.
- Provides Effective Gradients: Since the gradient of $B$ depends on $A$ (and vice versa), having $A$ be non-zero ensures $B$ receives a gradient immediately to kickstart the learning process.
- Increases Representation Power: It helps the low-rank matrix capture more complex feature combinations.

In practice, the $\Delta W$ update is multiplied by a scaling factor: $\frac{\alpha}{r}$. So the final formula is: $h = Wx + \frac{\alpha}{r}(BAx)$
- $\alpha$ (Alpha) is a hyperparameter you set to control how much influence the new LoRA weights have on the final output.
- If $\frac{\alpha}{r}$ is too large, the LoRA weights overpower the base model, leading to overfitting.
- If $\frac{\alpha}{r}$ is too small, the LoRA weights barely do anything, and the fine-tuning effect is unnoticeable.

When setting up a training run, a common and safe practice is to set $\alpha$ to be exactly twice the value of $r$ (e.g., if $r=8$, $\alpha=16$).

Choosing the right $r$ is crucial, but more isn't always better. $r = 4$, $8$, or $16$ is usually more than enough. The research shows that simply increasing $r$ does not necessarily capture a more meaningful subspace; a low-rank matrix is sufficient. However, applying the technique to various weight matrices might yield better results. Experiments definitively prove that applying LoRA to the $W_q$ (Query) and $W_v$ (Value) matrices inside the Transformer's Attention layers produces the absolute best results. Simple tasks need a smaller $r$; highly complex tasks need a larger $r$. A stronger, smarter base model requires a smaller $r$ (and less training data) to learn a new task. If you are training on a massive dataset, you generally need a larger $r$ to absorb all that new information.

<p align="center">
<img width="337" height="308" alt="86ba839d-31bc-4071-aaec-71bd00c54ae6" src="https://github.com/user-attachments/assets/2ab1a4d1-b239-4e5a-9790-56026ec48e6d" />

</p>

Below is a summary of the advantages for LoRA.

- Zero Inference Latency: After training is complete, you can mathematically add $BA$ directly into the original $W$ matrix ($W_{new} = W + BA$). During deployment (inference), the architecture is identical to the original model, meaning it doesn't run a single millisecond slower.

- High Parameter Efficiency: You achieve performance matching Full Fine-Tuning while only updating a tiny fraction of the parameters.

- Frozen Base Model: Because $W$ is never altered, you can train multiple different LoRA modules (one for coding, one for translation, one for math) and seamlessly swap them in and out on top of the same frozen base model.

- Strong Universality: It isn't restricted to a specific architecture; it can be applied to almost any dense layer in various Transformer models.

Example code:
```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super().__init__()
        
        # 1. The Original Pre-trained Layer (W)
        self.base_layer = nn.Linear(in_features, out_features)
        
        # FREEZE the base model weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # 2. LoRA Hyperparameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r # The scaling coefficient

        # 3. The LoRA Low-Rank Matrices (A and B)
        # Matrix A: Projects down from input dimension to rank 'r'
        self.lora_A = nn.Linear(in_features, r, bias=False)
        # Matrix B: Projects up from rank 'r' to output dimension
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # 4. Apply Initialization Rules
        self.reset_parameters()

    def reset_parameters(self):
        """
        The critical initialization step to ensure stability at step 0.
        """
        # Initialize A with a random Gaussian/Uniform distribution
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        
        # Initialize B with absolute ZEROS
        # This guarantees that lora_B(lora_A(x)) == 0 when training starts
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Calculate the frozen base output: Wx
        base_output = self.base_layer(x)
        
        # Calculate the LoRA delta update: BAx * (alpha/r)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        
        # Final output is the sum of both paths
        return base_output + lora_output

# --- Testing the Layer ---
# Create a simulated attention layer with input dim 1024, output dim 1024
layer = LoRALinear(in_features=1024, out_features=1024, r=8, lora_alpha=16)

# Notice how the number of trainable parameters is tiny!
# W matrix: 1024 * 1024 = 1,048,576 parameters (FROZEN)
# A matrix: 1024 * 8 = 8,192 parameters (TRAINABLE)
# B matrix: 8 * 1024 = 8,192 parameters (TRAINABLE)
```

#### 4.1.4.8 QLoRA

QLoRA (Quantized Low-Rank Adaptation) is the ultimate memory-saving evolution of the LoRA technique. While standard LoRA is incredibly efficient, QLoRA takes it a step further by drastically compressing the frozen base model, allowing you to train massive models on consumer-grade hardware. The primary goal of quantization is to compress high-precision weights into a lower precision (like 4-bit) to save space. Standard 4-bit quantization scales weights using this formula:

$$W_q = \text{round}\left(\frac{W - W_{min}}{W_{max} - W_{min}} \cdot (2^4 - 1)\right)$$

This maps all weights into 16 discrete integer bins (from 0 to 15). However, this standard method is not ideal for neural networks.

Neural network weights typically follow a zero-centered normal distribution with a standard deviation of $\sigma$. NormalFloat (NF) is an information-theoretically optimal data type that ensures every quantization bin contains an equal number of values, preventing data loss.

1. Normalization: The theoretical normal distribution $N(0,1)$ is scaled to fit precisely within a $[-1, 1]$ range.
2. Asymmetric Splitting: To handle the zero-center perfectly, NF4 splits the distribution asymmetrically using $2^{k-1}$ quantiles for each half. It divides the negative range $[-1, 0]$ into 8 parts and the positive range $[0, 1]$ into 9 parts.
3. Merging: This creates 17 boundaries, but since zero appears in both sets, they are merged (removing the duplicate zero) to leave exactly 16 unique quantiles.
4. Mathematical Calculation: The exact boundaries ($q_i$) are calculated using the quantile function $Q_X(\cdot)$ of the standard normal distribution:

$$q_i = \frac{1}{2}\left(Q_X\left(\frac{i}{2^k+1}\right) + Q_X\left(\frac{i+1}{2^k+1}\right)\right)$$

By matching the distribution of the model's weights to this mathematically optimal NF4 range, QLoRA preserves the model's accuracy far better than standard linear quantization.

Even when model weights are reduced to 4-bit, the system still needs to save quantization constants (scaling factors) for every block of weights so it knows how to de-quantize them later. In QLoRA, weights are processed in blocks of 64. If each block requires a 32-bit float (float32) constant, the memory overhead adds up quickly, costing an average of 0.5 bits per parameter just for the constants. QLoRA applies a second round of quantization. It groups every 256 quantization constants together and quantizes them from 32-bit floats down to 8-bit. By quantizing the constants themselves, the extra memory overhead is slashed down to a mere 3.17%, squeezing out even more VRAM savings. (Note: Because it is doubly quantized, the model must undergo a two-step de-quantization process during the forward pass to restore the values).

When fine-tuning large models, sudden spikes in memory (especially during the optimizer's gradient update step) can cause devastating Out-Of-Memory (OOM) errors that crash your training. QLoRA utilizes NVIDIA's unified memory feature to create a safety net. Similar to how a computer uses a hard drive for "pagefiles" when its standard RAM is full, QLoRA automatically transfers optimizer data from the GPU RAM to the CPU RAM when the GPU is running out of space. When the GPU needs that data back for the next update step, it is seamlessly paged back into the GPU memory. This ensures the training process is crash-proof, even in tightly constrained hardware environments. 

QLoRA is strictly designed for model deployment and training in limited hardware environments. If you only have consumer-grade GPUs or even CPU environments, QLoRA makes it possible to run and adapt massive foundation models at 4-bit precision. It drastically lowers memory consumption through its advanced quantization (NF4 + Double Quantization) while maintaining high task performance by leveraging the adaptability of LoRA. It effectively democratizes large language models, breaking down the hardware barriers that previously restricted their use to massive data centers.

<p align="center">
<img width="517" height="212" alt="98670416-9984-46ff-8814-8935d1243f41" src="https://github.com/user-attachments/assets/c616667d-2c94-4053-83ff-9172f48cbe59" />
</p>

#### 4.1.4.9 AdaLoRA

AdaLoRA (Adaptive Low-Rank Adaptation) is a highly optimized, "smart" evolution of the standard LoRA method. While traditional LoRA is efficient, it applies a "one-size-fits-all" approach to a model's architecture. AdaLoRA fixes this by dynamically distributing your computational budget to the parts of the model that need it most. 

Standard LoRA requires you to pre-define a fixed rank ($r$) that is applied uniformly across every targeted weight matrix in the model. This ignores the reality that different modules and layers have vastly different levels of importance during fine-tuning. For example, standard LoRA typically only trains the Attention modules and ignores the Feed-Forward Networks (FFN). However, research shows that the FFN is actually often more important for storing and adapting task-specific knowledge. Instead of a rigid assignment, AdaLoRA acts like a smart financial manager. It calculates an "importance score" for each weight matrix and adaptively allocates the parameter budget across them.

AdaLoRA achieves this dynamic allocation through two major technical shifts. One is adjusting the incremental matrix allocation. Crucial matrices receive a high rank to capture fine-grained, highly specific task information. Less important matrices have their rank reduced (pruned). This not only saves computational budget but also actively prevents the model from overfitting on unimportant features. The other shift is parameterization via SVD (Singular Value Decomposition). Instead of just using two generic matrices ($A$ and $B$), AdaLoRA parameterizes the weight updates using the form of SVD:

$$W = W^{(0)} + \Delta = W^{(0)} + P \Lambda Q$$

Calculating exact SVD for massive matrices at every training step is computationally disastrous. AdaLoRA avoids this by adding an extra penalty term to the training loss. This penalty enforces the orthogonality of matrices $P$ and $Q$ automatically, stabilizing the training without the heavy math. It actively prunes (cuts out) the unimportant singular values inside $\Lambda$ based on their importance metrics, while keeping the singular vectors intact so the model can still recover and learn stably.

To figure out which layers deserve a higher rank, AdaLoRA uses the gradient information (how much the model wants to change a layer) to adjust the rank dynamically. It uses the Frobenius norm of the gradients to calculate this:

$$\Delta r = \alpha \frac{||\nabla W||_F}{\sum_{i=1}^L ||\nabla W_i||_F}$$

- $\Delta r$: The exact amount the rank $r$ should be adjusted for a specific layer.
- $\alpha$: An adjustment coefficient (a hyperparameter) that controls the scale/ratio of the change.
- $||\nabla W||_F$: The Frobenius norm of the weight's gradient at that specific layer. It measures the absolute "degree of update" happening in that layer during the current training step.
- $\sum_{i=1}^L ||\nabla W_i||_F$: The sum of the gradient Frobenius norms across all layers. This normalizes the score so that layers are judged relative to one another.

Layers with highly complex updates get larger ranks, while quieter, less complex layers are restricted to smaller ranks.

AdaLoRA provides highly efficient, fine-grained adaptation for specific downstream tasks on massive pre-trained foundations. By dynamically adjusting the rank, it entirely eliminates "resource waste." It guarantees that every single trainable parameter is doing heavy lifting, preserving the model's powerful performance while shrinking the footprint. 

## 4.2 Reinforcement Learning in LLM 

Reinforcement Learning (RL) is essentially the "trial-and-error" branch of AI. It’s about teaching an agent how to behave in an environment to maximize rewards. Think of it as training a digital athlete: it doesn't just learn from a static textbook; it learns by actually playing the game.

RL isn't just for simple tasks; it is specifically designed for multi-step sequential decision-making problems. These are situations where an action taken now might not show its true value until many steps later. Key real-world applications include:
- Games: Competitive strategy games like AlphaGo.

- Physical Systems: Autonomous driving and robot operation control.

- Digital Strategies: Online interaction strategies for recommendation systems.

RL excels in these areas because it can self-adapt through trial-and-error interaction, even when the environment’s probability transition model is unknown or when the dimensions of possible "states" and "actions" are massive. Algorithms in Reinforcement Learning are broadly divided into two categories based on how they view the "world" around them: Model-Free Learning and Model-Based Learning. Model-Free Learning is the most common and "simple" approach. The agent doesn't try to understand the hidden laws of physics or the "why" behind the environment. Instead, it just reacts to what it sees. It doesn't build a mental map (model) of the environment. It focuses on direct optimization of policies or value functions through repeated sampling from the real world. It is simple to implement and highly applicable to complex real-world scenarios where the environment is too messy to model. Model-Based Learning is more "cerebral." The agent tries to build a internal simulation of the world. It first learns or assumes a transition distribution $P(s' | s, a)$—meaning it tries to predict what the next state ($s'$) will be given its current state ($s$) and action ($a$). It then uses this "virtual environment" to plan or simulate future moves. It is excellent for planning future decisions before actually taking them. Its performance is strictly capped by the quality of its model. If the "virtual environment" doesn't match the real environment, the agent's performance will be poor.

In large-scale real environments where every action has a huge impact on the future, choosing between these two methods involves a trade-off between the simplicity of Model-Free sampling and the strategic planning of Model-Based simulation.

<p align="center">
<img width="813" height="404" alt="0d9035e1-6c90-487b-a0ad-81784f7eaef1" src="https://github.com/user-attachments/assets/a7c64a17-7acc-43f7-ac67-99361b1848c9" />
</p>

Or, RL algorithms can also be broadly classified based on how the agent learns to make decisions. We can break down the field into three primary methodologies: Value-Based, Policy-Based, and the hybrid Actor-Critic.

- Value-Based Methods: "Calculating the Worth". The core philosophy here is to assign a specific "Value" to every possible action in a given state. For every State, the agent calculates a "Value" for every feasible Action. It then follows a simple rule: choose the action with the highest Value. These values are updated and iterated using the Bellman Equation. This equation is critical because it balances immediate rewards with long-term future gains. Representative algorithms include Q-Learning and SARSA (State-Action-Reward-State-Action). Best use cases include situations where you have a set list of moves (e.g., Pacman moving Up, Down, Left, or Right), or when the final goal is to always perform the same "best" action in a specific state.

- Policy-Based Methods: "Learning the Strategy". Instead of calculating values for actions, these methods model the Policy ($\pi$) directly. The agent learns a probability distribution of actions. For a given state, it doesn't say "Action A is worth 10 points"; it says "There is a 70% chance Action A is the best move." The agent interacts with the environment and uses Gradient updates to optimize the strategy, aiming to maximize the expected total return. Representative algorithms include Policy Gradient (PG) series. Best use cases include scenarios where actions aren't simple buttons, but fluid movements (e.g., the precise rotation angle of a robot's joint), and situations where you need to be random to be optimal (e.g., in Rock-Paper-Scissors, the best strategy is a random 1/3 split for each move).

- Actor-Critic: The Best of Both Worlds. This is the most modern and popular approach in Deep Reinforcement Learning because it combines the strengths of the previous two methods. The Actor is responsible for learning the policy (deciding which actions to take).
## 4.2.1 Markov Decision Process
