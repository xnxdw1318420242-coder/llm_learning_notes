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

## 4.1.2 Training Strategy
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
## 4.1.3 Incremental Pre-training

## 4.1.4 PEFT

## 4.2 Reinforcement Learning in LLM 
