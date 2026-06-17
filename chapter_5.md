# 5. Model Optimization

## 5.1 Training Optimization

### 5.1.1 Memory Usage

Understanding exactly where a Large Language Model (LLM) consumes memory is the first step to efficiently training and deploying it. Memory consumption can be broken down into two main categories: Model States and Residual States.

During training, VRAM is divided into two primary buckets. Model States  represents the memory required to hold the model parameters, their gradients, and the optimizer tracking them. Assuming $M$ is the number of parameters and dtype_bytes is the byte size of the data type used:

- Weights: $M \times \text{dtypebytes}$
- Gradients: $M \times \text{dtypebytes}$
- Optimizer: For AdamW, this requires tracking three variables (parameter copies, momentum, and variances). Therefore, it costs $3 \times M \times \text{dtypebytes}$.

Total Model States Training Memory = $5 \times M \times \text{dtypebytes}$.

Residual States is the transient memory used during the forward and backward passes:

- Activations: Every operator's output must be cached for the backward propagation step. Activation memory scales proportionally with the formula:

$$\text{layers} \times \text{hiddendimensions} \times \text{seqlength} \times \text{batchsize}$$

- Temporary Buffers: Memory temporarily allocated for Gradient Fusion, cross-device gradient synchronization (All-Reduce), and Gradient Norm Computations.
- Memory Fragmentation: Wasted memory gaps similar to external fragmentation in standard OS memory management.

During inference, memory calculation is much simpler because there are no gradients or optimizers.

- Model States: Just the weights ($M \times \text{dtype\_bytes}$).
- KV Cache: The memory used to store past key and value vectors during autoregressive generation. The exact formula is:

$$2 \times \text{batchsize} \times \text{numlayers} \times \text{seqlen} \times \text{hiddendim} \times \text{dtypebytes}$$

When calculating training memory, an easy rule of thumb is to look at the multiplier per billion parameters ($M$).

- Single Precision (FP32): The global dtype_bytes = 4. Because Model States require $5 \times M$, the total baseline memory is 20 bytes per parameter (or ~20GB per 1 Billion parameters).
- Pure BF16 Training: The global dtype_bytes = 2. The total baseline memory is 10 bytes per parameter (or ~10GB per 1 Billion parameters). Configured in Transformers via TrainingArguments(bf16=True).
- LoRA Training: Because LoRA freezes the base model and only tracks gradients and optimizer states for tiny injected adapter layers, it drastically shrinks Model States to roughly 4 bytes per parameter (or ~4GB per 1 Billion parameters). Because Model States shrink so drastically in LoRA, the Residual States (like activations) will suddenly appear to take up a massive relative percentage of the total VRAM.
- The Mixed Precision (MPT) Misconception: It is widely misreported that Mixed Precision (FP16/BF16) halves your memory. This is false in standard frameworks like Transformers/Accelerate. While compute happens in lower precision, the framework must maintain a full FP32 copy of the master weights to prevent underflow during optimizer updates. Instead of 1 FP32 model, you end up with 1 FP32 copy + 1 FP16 copy, meaning memory actually stays at roughly 20 bytes per parameter (or can even be $1.5\times$ higher in specific setups). MPT is used for computational speed, not memory saving.

If you hit an Out of Memory (OOM) error, here are the actionable strategies to fix it.

- Reduce Sequence Length or Batch Size. Since layers and hidden dimensions are fixed by the model architecture, the only way to linearly reduce Activation memory is to reduce seq_length or batch_size. However, a batch size that is too small leads to unstable gradient calculations. A sequence length that is too short can permanently degrade the model's ability to handle long-context tasks.

- Gradient Checkpointing. Instead of storing all intermediate activations in memory for the backward pass, Gradient Checkpointing only saves a few "checkpoints." During backpropagation, it recalculates the missing activations on the fly. Full Checkpointing reduces activation memory massively (e.g., dropping from 60GB down to 8GB) but adds a 36% computation overhead. Selective Checkpointing only checkpoints specific operators that take up huge memory but are fast to recompute (like Attention). This reduces the overhead to just 4%. While Gradient Checkpointing prevents OOM errors, it slows down overall training speed by roughly 20%.

- Gradient Accumulation. If you are forced to use a tiny batch_size (like 1) to fit the model into VRAM, your training will suffer. Gradient accumulation allows you to simulate a large batch size by running multiple small forward/backward passes and accumulating the gradients before finally stepping the optimizer. It overcomes VRAM limits to achieve effective batch sizes, but the extra forward and backward passes significantly slow down training time.

### 5.1.2 Model Compression

Model Compression is a crucial sub-field of Training Optimization. As neural networks grow larger, model compression techniques are essential to reduce their memory footprint, lower deployment costs, and speed up inference without heavily sacrificing performance.

**Quantization** is the process of reducing the mathematical precision of the model's weights and activations (e.g., converting 32-bit floating-point numbers into 8-bit integers). This drastically shrinks the model's physical size and speeds up computation.

- Linear Quantization vs. Non-linear Quantization: Linear quantization maps floating-point values to integers using a uniform, evenly spaced scale. Non-linear quantization uses uneven spacing, which is often more accurate because neural network weights usually cluster around zero in a bell curve rather than being evenly distributed.

- Post-Training Quantization vs. Quantization-Aware Training: In Post-Training Quantization (PTQ), you take an already fully trained model and compress it. It is fast and cheap but can result in a drop in accuracy. In Quantization-Aware Training (QAT), you simulate the effects of quantization during the training process itself. The model learns to adapt to the lower precision, resulting in much higher final accuracy.

Classic algorithms include QAT (Quantization-Aware Training), PTQ (Post-Training Quantization), and DoReFa-Net (A specific network architecture designed to train with low bitwidth weights and activations).

**Pruning** works on the premise that not all parts of a neural network are equally useful. It involves physically deleting the least important connections (weights) or entire neurons to make the model smaller and faster.

- Structured Pruning vs. Unstructured Pruning: Unstructured Pruning deletes individual weights scattered randomly throughout the network. While it reduces parameter count, modern hardware (like GPUs) struggles to accelerate this irregular, "swiss-cheese" sparse matrix. Structured Pruning deletes entire rows, columns, or channels of weights. This maintains a neat, dense matrix structure, making it highly efficient for hardware acceleration.

- Magnitude-based Pruning vs. Importance-based Pruning: Magnitude-based is the simplest method. If a weight's absolute value is very close to zero, it is deemed useless and deleted. Importance-based uses more complex metrics (like gradient analysis or loss sensitivity) to determine if a specific weight or layer is critical to the model's reasoning, regardless of its raw size.

Classic algorithms include Magnitude Pruning and Lottery Ticket Hypothesis: A famous theory stating that within any massive, randomly initialized network, there exists a smaller, sparse sub-network (the "winning ticket") that can be trained from scratch to match the performance of the full model.

**Knowledge Distillation** doesn't physically shrink an existing model; instead, it uses a massive, highly capable model (the "Teacher") to train a smaller, more efficient model (the "Student").

- Offline Distillation vs. Online Distillation: In offline mode, the Teacher is already fully trained and frozen. It simply provides the answers, and the Student learns from them. In online mode, both the Teacher and the Student are trained simultaneously, continuously learning and updating together.

- Logits-based Distillation vs. Feature-based Distillation: For logits-based, the Student tries to mimic the final output probabilities (the logits) of the Teacher model. In feature-based, the Student goes deeper, trying to mimic the intermediate hidden layers and internal feature representations of the Teacher, allowing it to learn the Teacher's internal "thought process."

Classic algorithms include KD (Knowledge Distillation, the foundational algorithm introduced by Geoffrey Hinton), FitNets (an evolution that forces the student to learn from the intermediate hidden layers (feature-based) of the teacher), and Attention Transfer (The student learns to mimic the "attention maps" of the teacher, focusing on the same important parts of the data).

