# 4. Post-Training
## 4.1 SFT (Supervised Fine-Tuning)
Supervised Fine-Tuning (SFT) is a critical phase in the LLM training pipeline that occurs after large-scale pre-training. It involves further training a base model on a smaller, high-quality dataset of labeled prompt-response pairs. Its core objective is to align the model’s outputs with specific human requirements and task-specific goals. Its task alignment ensures the model output matches actual user needs (e.g., a dialogue system). It should achieve high performance with far fewer resources and data compared to pre-training. Domain adaption requires it to rapidly adapting the model to specialized fields like Medicine, Law, or Finance.

Unlike pre-training, SFT introduces specialized tokens to define the structure of a conversation:

- system: Sets the persona or global constraints.

- user: The human input.

- assistant: The model's response.

- eos_token: Crucial for teaching the model exactly when to stop generating. Pre-trained models often struggle to stop because they were never exposed to end-of-sentence signals.

The time taken for a model to generate a response can be approximated by the formula:

$$y = kx + b$$

$b$: The "Time to First Token" (TTFT), which is almost linearly correlated with the prompt length and is heavily influenced by the KV cache mechanism.

$kx$: The generation time, where $k$ is the cost per token and $x$ is the total number of tokens generated.
## 4.2 Reinforcement Learning in LLM 
