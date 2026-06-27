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

- Model States: Just the weights ($M \times \text{dtypebytes}$).
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

### 5.1.3 Parallelism

<p align="center">
<img width="899" height="324" alt="dab13acc-0d25-4572-a3a3-24b55ef6e959" src="https://github.com/user-attachments/assets/79b601a6-5c5d-468f-b70f-dc96f5d792c3" />
</p>

#### 5.1.3.1 Data Parallelism

As deep learning models and datasets grow exponentially, the required computation, storage, and training times increase drastically. Distributing these requirements across multiple devices is the key to speeding up training. Data Parallelism is a parallel execution strategy that strictly follows the Single Program Multiple Data (SPMD) principle:

- Single Program: In deep learning training, the model network architecture and parameters on every single process/device are absolutely identical.

- Multiple Data: Every process handles a completely different slice of the dataset.

By splitting the data and computation across different processes under a global dataset, this method heavily reduces the compute and storage pressure on any single device. It increases overall training throughput simply by adding more training hardware.

Currently, the mainstream deep learning frameworks implement data parallelism based on Distributed Synchronous SGD (Distributed Synchronous Stochastic Gradient Descent). There are two primary modes for this: DP (Data Parallel) and DDP (Distributed Data Parallel).

DP is a single-process, multi-thread strategy that can only be executed on a single machine. The "Main" GPU handles gradient aggregation and optimizer updates. The step-by-step process is:

1. Single-process controls multiple GPUs (Blue Text): Its fundamental nature is single-process multi-threading.

2. Load the model to the Main GPU, then duplicate it to all specified GPUs.

3. Split the input data along the Batch dimension. Each GPU independently performs its own forward computation.

4. Sync all results back to the Main GPU to finalize gradient calculations and parameter updates. The Main GPU then copies the fresh, updated weights back to all other GPUs.

Because it uses a single process to control multiple GPUs, there is an inherent problem of load imbalance between the GPUs, with the main GPU bearing a significantly heavier load.

DDP completely breaks the single-process lock constraints. It uses an AllReduce architecture and a multi-process approach, allowing it to work across both single and multiple machines. The load is distributed across each GPU node. The communication time cost is constant and entirely independent of the number of GPUs. It equals the parameter volume $V$ divided by the bandwidth $B$. DDP does not require a Main GPU to broadcast the full model to everyone. It communicates using a ring-all-reduce topology. Because the total transmission volume remains constant regardless of the GPU count $N$, ring-all-reduce provides linear acceleration capabilities as you add more hardware.

To ensure the "Multiple Data" part of SPMD works, the dataset must be split. Before an epoch starts, divide the entire dataset by the number of parallel processes. Each process only reads its assigned chunk. Alternatively, a single process (usually rank0) reads the data, splits it into chunks based on the process count, and dispatches the blocks to the respective processes.

The dataset is typically split into $N$ parts (where $N$ is the parallel degree). Because every GPU must sync gradients once per iteration, it is mandatory that every training card experiences the exact same number of iterations per epoch. If they don't, cards with more iterations will hang infinitely waiting for communication from finished cards. To guarantee equal iterations, frameworks use Data Padding or Discarding. Padding means duplicating/adding some data to the subsets that have fewer iterations. Discarding means dropping tail-end data from subsets that have too many iterations.

Because data must be shuffled every epoch, you must choose when to do it:

- Shuffle BEFORE splitting: Shuffle the whole dataset $\rightarrow$ Apply padding/discarding $\rightarrow$ Split the data.
- Shuffle AFTER splitting: Apply padding/discarding $\rightarrow$ Split the data $\rightarrow$ Shuffle each individual subset independently.

The most critical challenge in Data Parallelism is guaranteeing that the model parameters ($W$) on every single process remain strictly identical at all times. Because each process calculates Loss on different data, they generate different local gradients. If they updated independently, the models would diverge. To solve this, two rules must be strictly enforced:

- Identical Initial Parameters ($W_0$). The model on every process must start exactly the same. All processes use the exact same random seed and initialization order. Initialize the full model on one specific process, then broadcast those parameters to all others.

-  Identical Gradient Updates ($\Delta W$). At every step, the gradient used to update the weights must be the exact same globally. This breaks the training loop into three parts - Forward Pass: Every process computes forward propagation on its unique data chunk, yielding a different Loss; Backward Pass: Every process calculates backpropagation based on its unique Loss, yielding different local gradients (Before updating, we must use an AllReduce Sum communication operation. After applying AllReduce Sum, every process holds the exact same gradient value (which is the sum of all local gradients combined). Each process then divides this sum by the total number of processes to get the true average global gradient); Parameter Update - Each process independently updates its parameters using this averaged global gradient. Because they started with the same $W_0$ and applied the exact same $\Delta W$, the resulting weights remain perfectly synchronized across all GPUs.

When switching to Data Parallel training, you must adjust the learning rate. The fundamental rule is: The learning rate is directly proportional to the global batch size. There are two ways to configure this. Either you force the sum of the batch sizes across all GPUs to equal the batch size you used on a single card. In this scenario, usually keep the learning rate on each compute device identical to single-card training. Or You let every individual GPU maintain the full single-card batch size. Therefore, your new global batch size is $N$ times larger than before ($N$ = number of devices). You need to set the learning rate of each compute device to $N$ times the single-card training learning rate. Because the initial learning rate is now massively multiplied, it is highly detrimental to the model's convergence. To prevent the model from exploding at step 1, you usually need to use a warm-up mechanism. You start the training with a very small learning rate and slowly increase it iteration by iteration until it reaches your large, target learning rate.

#### 5.1.3.2 Tensor Parallelism

As the NLP industry evolved from models like BERT to GPT, neural networks grew significantly deeper and wider, scaling from hundreds of millions to hundreds of billions of parameters. While Data Parallelism (DP) distributes the dataset across multiple GPUs, every single GPU must still hold a complete, intact copy of the entire model's compute graph and parameters. When the parameter scale reaches hundreds of billions, storing the model parameters requires hundreds of GBs of VRAM, exceeding the capacity of a single GPU card.) Because Data Parallelism cannot solve this memory bottleneck, we must use Model Parallelism. Model Parallelism fundamentally differs from Data Parallelism: instead of different devices handling different data, different devices are responsible for calculating different parts of a single compute graph. There are two types of Model Parallelism:

- Inter-layer Parallelism: Slicing the model by layers across devices, formally known as Pipeline Parallelism.
- Intra-layer Parallelism: Slicing the parameters within a single layer across devices, formally known as Tensor Parallelism.

Tensor Parallelism involves slicing the parameter matrices within a single network layer and distributing those slices across multiple GPUs. For example, a massive matrix multiplication that would normally happen on one card is split into smaller matrix multiplications across different cards. During the forward and backward passes, the separated data is integrated back together using communication primitives like All gather or All reduce.

While necessary for massive models, Tensor Parallelism has clear disadvantages:

- Communication Overhead: In multi-machine, multi-GPU environments, it requires all-reduce communication across servers. This is slower than the high-bandwidth communication within a single-machine multi-GPU server because inter-machine networking costs more.

- Lower GPU Utilization: A high degree of model parallelism shatters matrices into many small multiplications, which can decrease GPU utilization.

Therefore, Tensor Parallelism must rigorously solve two core problems:

- Splitting Method: How exactly to distribute the parameters.
- Mathematical Equivalence: How to guarantee the sliced math perfectly equals the original single-card math.

Below is exactly how this is solved across the three main components of a Transformer: the Embedding Layer, the MatMul Layer, and the Loss Calculation.

**Embedding Splitting**

If the total vocabulary is very large, it will cause a single card's VRAM to be unable to accommodate the Embedding layer parameters. The Embedding layer is sliced along the word_size (vocabulary) dimension. If using 2 GPUs, GPU 1 holds the first half of the vocabulary, and GPU 2 holds the second half.
When a batch of data ($bz$) looks up words:

- If a GPU does not possess a specific word, it outputs a vector of purely 0s for that word.
- Both GPUs output a tensor of shape [bz, hidden_size].
- The system uses an AllReduce Sum operation across the devices. By summing the real vectors with the 0 vectors from the other cards, every device ultimately receives the exact, mathematically equivalent full embedded tensor.

**MatMul Splitting**

For a standard Matrix Multiplication $Y = XA$ (where input $X$ is $M \times N$, parameter $A$ is $N \times K$, and output $Y$ is $M \times K$), there are two mathematically equivalent ways to split the massive parameter matrix $A$:

- Method A: Column Splitting. The matrix $A$ is sliced vertically into columns: $A = [A_1 \mid A_2]$. GPU 1 calculates $Y_1 = XA_1$. GPU 2 calculates $Y_2 = XA_2$. Both outputs are incomplete matrices of shape $M \times K/2$. An AllGather operation is used to concatenate $Y_1$ and $Y_2$ side-by-side, resulting in the final, complete $M \times K$ matrix $Y$.

- Method B: Row Splitting. The parameter matrix $A$ is sliced horizontally into rows: $A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$.  To make the math work, the input $X$ must also be split by columns: $X = [X_1 \mid X_2]$. GPU 1 calculates $Y_1 = X_1 A_1$. GPU 2 calculates $Y_2 = X_2 A_2$. Both GPUs output a matrix of the full final shape $M \times K$, but with partial values. An AllReduce operation is used to sum the matrices ($Y = Y_1 + Y_2$), yielding the exact final result.

**FFN Optimization**

A Transformer's Feed-Forward Network (FFN) consists of two sequential linear fully connected layers (two MatMuls). Tensor Parallelism handles them with a brilliant optimization. Split the first FC layer's parameter matrix by columns, and the second FC layer's parameter matrix by rows. If the first layer uses Column Splitting, its output naturally exists as a sequence split by columns ($[Y_1 \mid Y_2]$). This perfectly matches the input requirement for the second layer's Row Splitting. Because the output of Layer 1 flows perfectly into Layer 2, the AllGather communication operation after the first FC layer can be completely omitted/saved. Communication is only needed after the second layer.

**CrossEntropyLoss Splitting**

If the number of classes is very large), storing and computing the final output logit matrix will crash the GPU.
To fix this, the logits are split across the class dimension, and the Softmax/Loss math is executed in three small communication steps

- Find Global Max: Calculate local maximums $x_{\max}$ on each card. Use ncclAllReduce, Max to find the true global maximum to ensure numerical stability.

- Calculate Denominator: Calculate the local exponent sums $e^{x_i - x_{\max}}$. Use ncclAllReduce, Sum to combine them into the global denominator: $\sum_{j} e^{x_j - x_{\max}}$.

- Final Loss: Calculate the local probabilities and partial loss based on the local target labels. Use a final ncclAllReduce, Sum to aggregate the total global CrossEntropyLoss.

Below is how Megatron-LM processes the Row and Column splitting mathematically into map and reduce operations for both forward and backward propagation.

**Column Splitting**

- Map Phase. Forward: The input $X$ is passed identically to all GPUs (identity operation). Backward: The gradient requires an all-reduce operation: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial X_1} + \frac{\partial L}{\partial X_2}$.
- Reduce Phase. Forward: The output $Y = [Y_1, Y_2]$ requires an all-gather operation to concatenate. Backward: The gradient relies on a simple split operation: $[\frac{\partial L}{\partial Y_1}, \frac{\partial L}{\partial Y_2}] = \frac{\partial L}{\partial Y}$.

**Row Splitting**

- Map Phase. Forward: The input requires a split operation: $[X_1, X_2] = X$. Backward: The gradient requires an all-gather operation: $\frac{\partial L}{\partial X} = [\frac{\partial L}{\partial X_1}, \frac{\partial L}{\partial X_2}]$.
- Reduce Phase. Forward: The output requires an all-reduce operation to sum: $Y = Y_1 + Y_2$. Backward: The gradient is passed backward identically: $\frac{\partial L}{\partial Y_i} = \frac{\partial L}{\partial Y}$ (identity operation).

<p align="center">
<img width="455" height="540" alt="b5796146-13bc-4da2-93f1-4ff85d6e0286" src="https://github.com/user-attachments/assets/233f09a3-0697-4931-8c4e-22a90903550b" />
</p>

#### 5.1.3.3 Pipeline Parallelism

While Tensor Parallelism (TP) solves the memory issue of fitting massive parameter matrices onto GPUs, it has limitations. As highlighted in the red text, when using this method on a single machine with 8 cards (e.g., 32GB V100), you can at most train a Dense model of about 10B parameters. Furthermore, because TP involves heavy communication that cannot be overlapped with computation, it is generally unsuited for multi-machine setups. To train even larger models and hide communication latency, the industry introduced Pipeline Parallelism (PP).

Unlike Tensor Parallelism (which slices within a layer), Pipeline Parallelism operates on Inter-layer splitting. It assigns different layers of the neural network to different, designated GPUs. The 5 Core Steps of Pipeline Parallelism:

1. In a pipeline setup, the various layers of a model are split across multiple GPUs.

2. A single batch of data is divided into smaller micro-batches, and execution flows through the pipeline using these micro-batches.

3. The layers of one single model are dispersed across multiple devices.

4. For models with repetitive blocks (like Transformers), each device can be assigned an equal number of Transformer layers.

5. During training, a device executes its designated set of operations, then passes its output down the pipeline to the next device, which executes a different set of operations.

Pipeline parallelism drastically reduces the communication burden.
