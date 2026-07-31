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

Pipeline parallelism drastically reduces the communication burden.  If you apply Tensor Parallelism to an FFN layer, you must slice the fully connected parameters across different cards within the same device node. To integrate the results, you perform an AllReduce sum, which creates a massive total communication parameter volume of $2MK$ (where $M$ is the matrix row dimension and $K$ is the column dimension). When slicing at the FFN layer boundary across different devices, adjacent devices only need to pass intermediate variables and gradients point-to-point. They only need to send or receive a parameter volume of $MK$. Therefore, compared to Tensor Parallelism, Pipeline Parallelism has a much smaller communication parameter volume.

In a Naive Pipeline, only one device computes at a time while the others sit idle, leading to terrible resource utilization. To fix this, engineers split the mini-batch into multiple smaller micro-batches and use advanced scheduling. In the F-then-B Scheduling mode, a device sequentially executes the Forward (F) passes for all micro-batches first, and then executes the Backward (B) passes for all micro-batches. By computing different parts of the model simultaneously, F-then-B can significantly improve device resource utilization. To calculate the backward pass later, the GPU must keep the intermediate activations for every single forward pass alive in its memory. Because this F-then-B mode caches the intermediate variables and gradients of multiple micro-batches, the actual utilization rate of VRAM is not high. To fix the memory bloat of F-then-B, the 1F1B pipeline was created. Here, forward and backward computations are interleaved. Compared to the F-then-B method, the 1F1B method can save 37.5% of peak memory.

Even with micro-batches, pipeline parallelism inherently suffers from idle computing gaps known as bubbles. We can mathematically derive the proportion of these bubbles.

Let:

- $p$ = the number of pipeline compute units (stages).
- $t_f$ = the forward computation time for one micro-batch.
- $t_b$ = the backward computation time for one micro-batch.
- $m$ = the number of micro-batches within one mini-batch.

- Bubble Time Equation: The total idle bubble time ($t_{pb}$) in one mini-batch is:

$$t_{pb} = (p - 1)(t_f + t_b)$$

- Total Computation Time Equation: The actual compute time ($t_{id}$) for the mini-batch is:

$$t_{id} = m(t_f + t_b)$$

- Bubble Ratio Equation: By dividing the two, we get the fraction of time wasted as bubbles:

$$S_{Bubble} = \frac{t_{pb}}{t_{id}} = \frac{p - 1}{m}$$

With a fixed number of pipeline units $p$, the bubble ratio is entirely controlled by $m$. Under the condition that $m \gg p$ (where micro-batches vastly outnumber pipeline stages), you can effectively reduce the bubble proportion

There are two famous implementations of Pipeline Parallelism, each handling batch updates differently.

PipeDream splits the model across machines and allows a machine to start executing the forward computation for a second batch before the backward propagation of the first batch is fully completed. It will cause gradient convergence instability. When a machine performs gradient descent, it must use saved backups of parameters. If a machine computes the forward pass of task 3 using original parameters, but its backward pass uses parameters that were just updated by task 2, there is a fundamental version mismatch. Therefore, this will bring a large amount of error. We must strictly limit how many forward passes can happen before a backward pass to control this uncertainty.

<p align="center">
<img width="393" height="194" alt="f145d983-5dd4-49ab-a3bd-dfdcdaf2b780" src="https://github.com/user-attachments/assets/beebd8f7-d16c-402c-980e-dac920b31cf3" />
</p>

GPipe acts somewhat similarly but is strictly synchronous. It splits a batch into micro-batches, but waits for global gradient synchronization only after the entire global batch is finished. GPipe utilizes Re-materialization (also known as Activation Checkpointing). It intentionally discards intermediate activations and recalculates them during the backward pass. This trades compute time to lower VRAM usage. Overall, GPipe's speed is slower than PipeDream, but its memory footprint is much smaller, and its mathematical convergence is perfectly stable.

<p align="center">
<img width="437" height="195" alt="a552d239-8794-4c22-b7ae-939a80a9507b" src="https://github.com/user-attachments/assets/58c95862-d5eb-461e-8c5d-525c5d371cf6" />
</p>

#### 5.1.3.4 Sequence Parallelism

Colossal-AI Sequence Parallelism focuses on breaking sequence length limits, while Megatron-LM Sequence Parallelism focuses on reducing memory bloat leftover by Tensor Parallelism.

In Colossal-AI Sequence Parallelism, considering that long sequence data will exponentially increase intermediate memory usage, heavily restricting the training capability of the device, the authors propose Sequence Parallelism as a memory-efficient systems-level method to physically break the input sequence length limit, while existing work focuses on reducing time and space complexity purely from an algorithmic perspective. It slices the actual input text sequence vertically across devices. To calculate attention when the sequence is physically cut into pieces, Colossal-AI introduces RSA (Ring Self-Attention). Each GPU calculates its local $Q, K, V$ for its assigned chunk of the sequence. Then, the GPUs pass their $K$ and $V$ matrices to their neighboring device in a continuous ring. By passing the keys around the ring, every query eventually gets to calculate attention against every key, achieving full global attention while only holding a fraction of the sequence in memory at any given time. This method is perfectly compatible with Data Parallelism, Pipeline Parallelism, and Tensor Parallelism. When scaled to 64 NVIDIA P100 GPUs, compared to Tensor Parallelism, RSA achieved 13.7x and 3.0x increases in maximum batch size and sequence length, respectively. Combined with Sparse Attention, it can process sequences exceeding 114K tokens, which is over 27 times longer than traditional single-device sparse attention. Unlike Tensor Parallelism (which is restricted by attention head counts), as long as the sequence length is divisible by the sequence parallel size, sequence parallelism can be used.

Megatron-LM approaches Sequence Parallelism from an entirely different angle. Its primary goal is to handle the massive memory footprint left behind by layers that Tensor Parallelism cannot split. Megatron-LM analyzed exactly how much memory a standard Transformer consumes. Assuming:

- $s$ = Sequence length
- $b$ = Batch size
- $h$ = Hidden dimension
- $a$ = Number of attention heads

Every single layer of a standard Transformer consumes memory equal to:

$$sbh \left(34 + 5\frac{as}{h}\right)$$

hen Tensor Parallelism (size $t$) is activated, the massive Linear and Attention layers are divided among $t$ GPUs. However, the parts that cannot be shared are mainly the input and output of two LayerNorm blocks ($4bsh$); and two dropout mask blocks ($2bsh$); totaling $10bsh$. Because these specific blocks are duplicated identically across every GPU in the Tensor Parallel group, they waste massive amounts of VRAM. Megatron-LM borrows Colossal-AI's concept. Building upon Tensor Parallelism, it slices the inputs of LayerNorm and Dropout in the Transformer layer along the input sequence length dimension, so that each device only needs to perform a fraction of the Dropout and LayerNorm. The computation for LayerNorm and Dropout is evenly spread across devices and reduces compute waste. The intermediate activations generated by LayerNorm and Dropout are distributed, drastically dropping memory usage.

When you slice the sequence dimension alongside the tensor dimension, you break the standard communication flow. Standard Tensor Parallelism relies purely on All-Reduce operations (which sum data across GPUs and return the full sum to all GPUs). For Sequence Parallelism Integration, because the sequence is now physically cut into pieces, All-Reduce is no longer mathematically valid. To fix this, Megatron-LM replaces the standard All-Reduce with two separate, highly optimized operations:

- To collect the results produced by sequence parallelism on each device, an All-Gather operator must be inserted

- And to allow the results produced by tensor parallelism to be passed into the sequence parallel layer, a Reduce-Scatter operator must be inserted.

At first glance, using multiple All-Gather and multiple Reduce-Scatter operators per layer seems like more communication overhead than standard Tensor Parallelism. But this is not the case, because one All-Reduce is mathematically equivalent to one Reduce-Scatter plus one All-Gather, so their total communication volume is exactly the same. Furthermore, during backpropagation, the implementation brilliantly overlaps the Reduce-Scatter communication with the actual gradient calculations, further reducing time and maximizing GPU FLOPS Utilization.

Megatron-LM pairs Sequence Parallelism with a final VRAM optimization trick: Selective Activation Recomputation. The authors noticed that some operations in the Transformer produce massive activation values but require very little computation power to calculate. Instead of saving these massive activations in memory, the model deletes them entirely. When backpropagation occurs, it simply recalculates them on the fly to save space. Other, harder-to-compute activations are kept in memory normally. Sequence Parallelism alone reduces activation memory overhead by ~40%. Selective Activation Recomputation alone reduces activation memory overhead by ~40%. When both features are turned on, the total activation overhead can be reduced by about 80%. While recomputing adds a tiny bit of total math, the massive VRAM savings allow for significantly larger batch sizes, making the overall throughput improvement extremely obvious.

#### 5.1.3.5 Expert Parallelism

Expert Parallelism is a specific parallelization method designed for training MoE (Mixture of Experts) models. The MoE architecture modifies the standard Transformer foundation. Instead of having one massive, dense Feed-Forward Network (FFN) that every single piece of data must pass through, an MoE layer is configured with multiple Expert FFN networks.

While MoE reduces computational FLOPs, it does not reduce memory requirements. The model still possesses a massive number of expert networks, and all of those parameters must be stored in VRAM. If a model has 8 massive experts, storing all of them on a single GPU is likely impossible. Instead of slicing the parameters of a single matrix (like Tensor Parallelism), or slicing the layers of the model (like Pipeline Parallelism), Expert Parallelism simply distributes the separate, whole experts across different physical devices. Training massive LLMs today requires combining multiple parallelization strategies simultaneously to prevent memory and compute bottlenecks. The final crucial point made on the slide is that Expert Parallelism can be combined with 3D Parallelism without any conflicts. Because the routing mechanism of MoE is mathematically independent of how tensors are sliced or how batches are split, EP acts as a highly scalable "4th dimension" of parallelism, allowing MoE models to scale to trillions of parameters seamlessly.

#### 5.1.3.6 3D Parallelism

As deep learning models scale into the hundreds of billions or even trillions of parameters, no single parallelization technique (Data, Tensor, or Pipeline) is sufficient on its own. 3D Parallelism can be said to combine model parallelism, pipeline parallelism, and data parallelism together, and can be used to train models of almost all scales currently available. It is the ultimate composite strategy, forming a three-dimensional grid of computation across massive GPU clusters. 

First, divide the data to form multiple Data Parallel groups. Within each DP group, the model is split by layers into different Pipeline Stages. The tensors within each Pipeline Stage can use Tensor Parallelism.

### 5.1.4 Mixed Precision Training

To understand mixed precision, we must first understand the numeric data types it mixes. Deep learning traditionally relies on floating-point numbers, and the choice of format drastically impacts memory, speed, and accuracy.

**A. float32**

- Concept: Contains 8 bits for the exponent and 23 bits for the fraction/mantissa.
- Pros: It provides very high numerical precision. As highlighted in red, it can effectively avoid numerical underflow and overflow, resulting in high stability.
- Cons: It consumes a massive amount of memory, which can lead to out-of-memory issues when training large models. Computation is also slower because it requires more storage space and bandwidth.
- Application: Usually used for the initial stage of model training or extremely sensitive parameters, like the updates of certain weight matrices.

**B. float16**

- Concept: Contains only 5 bits for the exponent and 10 bits for the mantissa.
- Pros: Its bit-width is half of FP32, meaning weights take up half the space, drastically reducing VRAM requirements. Less data means faster transfer times across GPUs. On hardware like NVIDIA Tensor Cores, FP16 can significantly improve computation efficiency.
- Cons: Because it only has 5 exponent bits, its range is very narrow. It is incredibly prone to underflow and overflow problems. Because it only has 10 mantissa bits, its precision is low. For example, 0.00006666666 in FP32 might be truncated to 0.000067 in FP16. If a gradient update ($\delta$) is smaller than the minimum representable gap of FP16, it is rounded to zero, causing the update to fail.

**C. bfloat16**

- Concept: A hybrid format designed by Google. It contains 8 bits for the exponent (exactly like FP32) and 7 bits for the mantissa
- Pros: Because it shares the 8-bit exponent with FP32, float32 and bfloat16 have the exact same numerical range. This allows BF16 to process the same magnitude of numbers as FP32, essentially eliminating the risk of underflow/overflow, while still providing the storage and speed benefits of a 16-bit format.
- Cons: It has lower precision than FP16 (only 7 mantissa bits), which may cause slight errors in models that are highly dependent on numerical precision.

Mixed Precision Training is a method that combines different numerical precisions (like FP32 and FP16) to train a deep learning model. Use FP16 in memory for storage and multiplication to accelerate computation, and use FP32 for accumulation to avoid rounding errors. The standard MPT iteration for a layer (Figure 1 in the slides) involves a specific dance between FP16 and FP32:

- Weight Backup: A master copy of the weights is permanently kept in FP32.

- Forward Pass (FWD): The FP32 master weights are cast down to FP16 (float2half). The forward pass is calculated entirely in FP16 to utilize Tensor Cores.

- Backward Pass (BWD): The backpropagation is also calculated in FP16 to generate the Activation Gradients and Weight Gradients.

- Weight Update: The FP16 Weight Gradients are cast back up to FP32. They are then added to the FP32 Master-Weights. This step is crucial because gradient updates are often tiny; doing this addition in FP32 entirely prevents the Rounding Error where tiny updates become zero.

While memory is saved, MPT actually requires keeping an extra master copy of the weights, meaning the memory savings are not a perfect 50% reduction, but rather a strategic reduction that allows larger batch sizes. Copying the weights to FP32 increase memory usage. However, during training, dynamic memory (intermediate variables and activations) takes up 3-4 times more space than static memory. Because all dynamic memory is stored in FP16, compared to training the entire network with FP32, the final memory footprint of the model is still essentially halved.

<p align="center">
<img width="450" height="198" alt="4461154f-e58f-4ad5-982d-df45a7df55e4" src="https://github.com/user-attachments/assets/10aca006-4b2b-4402-88f1-f064c6e717d5" />

</p>

While storing the master weights in FP32 solves the Rounding Error, it does not solve the Underflow. During backpropagation, activation gradients are often much smaller than weight gradients. The minimum number FP16 can represent is $2^{-24}$. If a calculated gradient falls below this threshold, it becomes exactly 0. If this happens early in the network, all subsequent layers receive a gradient of 0, and the training fails. To prevent gradients from underflowing into zero, we artificially inflate them so they fit safely within FP16's representable range. Before Backpropagation, the final calculated Loss is manually multiplied by a large scaling factor. Because the loss is scaled up, the derivative (the gradient) is also mathematically scaled up by the exact same factor. Therefore, the gradients will not underflow when calculated in FP16. After Backpropagation, before applying these gradients to update the FP32 master weights, the gradients are divided by the exact same scaling factor - shrink the weight gradients to restore their normal values.

To understand why scale the loss instead of gradients: Because of the chain rule, scaling the loss mathematically scales all subsequent gradients automatically. This is much more cost-effective than scaling every single gradient individually.

- Scale Up Phase: After forward propagation and before backpropagation, the loss is multiplied by $2^K$. The scaled gradients remain safely inside FP16's representable range.
- Scale Down Phase: After backpropagation, the weight gradients are divided by $2^K$ to restore their true values before updating the FP32 master weights.

While you can use a constant scaling factor (like $8$ or $32K$), this is risky. If the gradients naturally grow larger during training, a static multiplier might cause the values to hit the FP16 ceiling ($65504$), causing an Overflow. To fully utilize the FP16 range and mitigate rounding errors, we use Dynamic Loss Scaling to safely use the largest possible multiplier.

1. Start with a very high scale factor, like $2^{24}$.

2. During the training iteration, check for overflow: If there is no gradient overflow, do not change the scale factor. Continue training. If there is gradient overflow, halve the scale factor (divide by 2). Skip the weight update for this step, and try again. Repeat until the gradients fit safely without overflowing.

3. In the later stages of training: Loss convergence stabilizes, and gradient updates become smaller. At this point, we can allow a higher loss scaling factor again to prevent underflow.

4. The F-Multiplier Rule: The algorithm attempts to multiply the loss scale by a factor $F$ every $N=2000$ iterations. If no overflow is detected, the new, higher scale factor is kept.

Even during the FP16 Forward and Backward passes, we can protect against precision loss without abandoning speed. This is done through Precision Accumulation. Use FP16 for matrix multiplication, and use FP32 for addition calculations to make up for the lost precision. Inside Nvidia Volta architectures, the Tensor Core is specifically designed for this. It takes FP16 matrices ($A$ and $B$), multiplies them extremely fast, but accumulates the results into a $C$ and $D$ matrix that is stored in FP32.  Using FP32 in the accumulation phase can drastically reduce precision loss in mixed precision training. 

Modern frameworks handle all this logic automatically. For example, NVIDIA's APEX library provides 4 levels of optimization strategies for mixed precision:

- O0: Pure FP32 training (The baseline).

- O1 (Conservative Mixed Precision): This strategy uses a whitelist/blacklist approach based on Tensor operations. Operations highly friendly to FP16 (like GEMM and CNN Convolutions) have their inputs and weights cast to FP16.  Operations that require high numerical stability (like Softmax and Batch Normalization) are strictly kept in FP32. It automatically includes Dynamic Loss Scaling.

- O2 (Aggressive Mixed Precision): This is the classic MPT architecture we discussed. The model weight parameters and input network data are entirely converted to FP16. Only specific sensitive operations like Batch Normalization remain in FP32. It relies heavily on the Weight Backup (FP32 master weights) and Dynamic Loss Scaling to prevent the rounding errors and underflows caused by moving almost everything to FP16.

- O3: Pure FP16 training (Rarely used because it easily diverges).

Mixed Precision Training is a cornerstone technique in modern deep learning that strategically combines the mathematical stability of 32-bit floating-point (FP32) formats with the blistering speed and efficiency of 16-bit floating-point (FP16) formats. By logically integrating FP32 and FP16, mixed precision drastically elevates training efficiency. This is primarily realized through two massive advantages:

- Faster Computational Speed. Computations performed in FP16 are typically twice as fast, or even faster, compared to those done in FP32. This acceleration is overwhelmingly noticeable on specialized hardware engineered specifically for mixed-precision arithmetic, most notably NVIDIA's Tensor Cores. When these cores are utilized, the mathematical throughput of the network skyrockets.

- Higher Memory Utilization. Because FP16 requires exactly half the bit-width of FP32, it intrinsically uses significantly less memory to store the same number of parameters and activations. This freed-up VRAM is highly valuable: it allows AI researchers to either train much larger model architectures or utilize significantly larger batch sizes on the exact same hardware, directly maximizing training efficiency.

While FP16 offers incredible speed and memory savings, it introduces physical hardware limitations that must be engineered around:

- Numerical Instability. The primary flaw of FP16 is its narrowed dynamic range and lower precision. This makes the training process highly susceptible to numerical underflow or overflow (data disappearing into 0 or exploding to infinity). This is especially dangerous during backpropagation when gradients are exceptionally small. To effectively prevent this network collapse, developers introduce a Loss Scaling mechanism. This algorithm artificially inflates the loss before calculating gradients, ensuring the tiny numbers safely fit inside FP16's limited range without underflowing. Other techniques to prevent underflow or overflow include using bfloat16, choosing stable activation functions such as Leaky ReLU or ELU instead of ReLU to prevent gradients of 0, batch normalization for stable gradient distribution, gradient clipping, adjusting the learning rate, reducing the batch size for smaller gradient, and proper initialization method such as He to prevent abnormal values in the early stage of training.

- Hardware and Software Requirements. Mixed precision is not a pure software trick; it requires a highly specific hardware and software stack to function. It relies on specialized silicon like NVIDIA Tensor Cores.  It requires compatible low-level libraries like CUDA and cuDNN. Fortunately, the deep learning ecosystem has caught up. Mainstream deep learning frameworks, such as PyTorch and TensorFlow, now offer native, out-of-the-box support for mixed precision training.

Mixed precision training is no longer an experimental feature; it is widely deployed as the industry standard for large-scale model training tasks.  It is absolutely vital in scenarios that demand processing massive batches of data through highly complex neural networks. The most prominent examples include the training of foundational Large Language Models (LLMs) like the BERT and GPT series models. On compute-bound hardware, mixed-precision technology unlocks the ability to significantly boost training performance and slash memory overhead. Ultimately, it allows AI engineers to push the boundaries of model scale and dataset size without needing to infinitely scale their physical server resources.

At its core, Mixed Precision achieves the perfect equilibrium: it guarantees the rigorous numerical stability of FP32 while harnessing the unmatched computational efficiency of FP16. This combination brings about profound optimizations in both training velocity and memory consumption, cementing it as an essential, foundational technology in current deep learning. Furthermore, the mandatory introduction of Loss Scaling technology is the linchpin that secures numerical accuracy during FP16 calculations, ensuring that models achieve the ultimate balance of high precision and high performance.

## 5.2 Inference Optimization
 
When generating text using modern Large Language Models, the system struggles against two primary physical limitations: Compute and Storage. The compute bottleneck arises directly from the Autoregressive nature of generative models—also known as the "Next Token Prediction" pattern. This creates a massive problem during the Decode Phase. Because the model can only generate one word at a time, the computation becomes unsaturated. If you are serving a single user (batchsize = 1), the GPU is essentially starving for work. In the Decode phase, the attention mechanism for a single step is written as:

$$ \mathbf{o} = \text{softmax}\left(\frac{\mathbf{q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V} $$

Here is why this equation represents a bottleneck:

- Because we are generating only one token, the query $\mathbf{q}$ is just a single vector, not a matrix. When multiplied by the Keys matrix $\mathbf{K}^T$, the result is a $1 \times L$ vector (where $L$ is the sequence length). This is scaled by $\sqrt{d}$.

- Applying softmax yields an attention weight vector $\alpha \in \mathbb{R}^{1 \times L}$.

- Multiplying this vector by the Values matrix $\mathbf{V} \in \mathbb{R}^{L \times d}$ yields a final output vector $\mathbf{o} \in \mathbb{R}^{1 \times d}$.

On GPUs, we want Matrix-Matrix multiplication, not Matrix-Vector multiplication. Matrix-Vector operations fail to utilize the massive parallel processing power of GPU Tensor Cores.

The Storage Bottleneck must be viewed from two angles:

- Static Storage: Large models have billions of parameters. Simply storing these massive weight matrices takes up a huge amount of GPU VRAM space.

- Dynamic Memory Access / Bandwidth: During the Decode phase, we must use the KV Cache strategy to avoid recalculating past tokens. However, this means for every single new word generated, a massive amount of historical Key and Value data must be read from the slow GPU HBM (High Bandwidth Memory) into the fast shared memory. Frequent HBM access squeezes out the actual time the GPU spends doing real computation.

To accurately pinpoint the bottleneck in a Transformer, we cannot use a broad brush. In Encoder-like Transformers (and LLM Prefill Phase, used in BERT, ViT, or the initial prompt-reading phase of an LLM), we input a sequence of length $\mathcal{O}(n)$ and output a sequence of $\mathcal{O}(n)$. This is highly parallel matrix math. In Transformer-Block with Per-Token Latency as the metric, the bottleneck is the FFN (Feed-Forward Network) module. The FFN simply takes the most time per token. Furthermore, increasing the batch size does not change this latency ratio because the GPU computation is already fully saturated. If the metric is Operation Intensity, the bottlenecks are QKV Projection, Output Projection, and FFN. These are "Skinny MatMuls" (narrow matrix multiplications), which inherently lead to poor utilization of GPU Tensor Cores. In Attention Layer with latency as the metric, the bottleneck lies in non-matmul operations like Softmax. For long sequences, there is constant, heavy memory swapping between the GPU's HBM and SRAM just to calculate softmax. This exact problem led to the invention of FlashAttention. FlashAttention uses "tiling" to load blocks of K, V, and Q into fast on-chip SRAM, computes them there, and writes the output back to HBM. This prevents the slow materialization of the massive $N \times N$ attention matrix on HBM, resulting in a 7.6x speedup. If the metric is Operation Intensity, the bottleneck is in the Activation-Activation Operations, specifically the Logit (L) and Attend (A) operators. Roofline model analysis proves that increasing batch size cannot solve the memory-bound nature of these specific L/A operations. This bottleneck spawned optimizations like Flat Attention.

For Decoder-like LLMs (The Generation Phase), the inference is strictly divided into a Two-Phase KV Cache Paradigm:

- Prefill (Initialization) Phase: The model reads the user's prompt, generates all initial $q, k, v$ matrices, and stores them in the KV Cache. Because this processes all tokens at once (Matrix $\times$ Matrix), it is highly efficient and Compute Bound. Its bottlenecks match the Encoder-like analysis above.

- Decode Phase: The model generates tokens one by one. To do this, it must read the $k, v$ of all previous tokens from the KV Cache for every single step. This makes it severely Memory Bound.

In the decode phase, we no longer look at "operation intensity" because the KV Cache is the absolute undisputed bottleneck. The most important variable here is batch size. In Transformer Block when metric is Per-Token Latency, if the batch size is small, the bottleneck is the FFN. If the batch size is large, the bottleneck shifts to the Attention module. In Attention Layer, the absolute bottleneck is the KV Cache Load Time (moving data from GPU HBM to shared memory). At a batch size of 8, computation latency dominates. However, as batch size increases to 256, computation time shrinks drastically per token, but the kv_cache_latency remains static and becomes the overwhelming majority of the processing time. To optimize LLM inference, one must understand that generation is a tale of two phases. You must ensure the GPU has enough computational throughput for the initial prompt processing (Prefill Phase / Compute Bound), but you must urgently optimize memory bandwidth and KV Cache management (via techniques like PagedAttention or quantization) to survive the token-by-token generation (Decode Phase / Memory Bound), as standard GPU matrix-multiplication hardware is fundamentally starved during this process.

### 5.2.1 Algorithm Optimization

#### 5.2.1.1 Speculative Decoding

Speculative Decoding is a cutting-edge inference optimization technique designed to drastically speed up Large Language Models (LLMs) without altering their final output distribution. By leveraging a smaller, faster model to "guess" tokens and a larger model to "verify" them in parallel, it breaks the compute bottleneck of auto-regressive generation. The fundamental concept of Speculative Sampling (SpS) involves two models:

- Draft Model ($p$): A lightweight, fast auto-regressive model.

- Target Model ($q$): The massive, highly accurate main model.

The process follows three distinct phases:

1. Small Model Auto-regression: The draft model $p$ runs for $K$ steps sequentially, generating a sequence of candidate tokens: $x_1, x_2, \dots, x_K$.

$$p_1(x) = M_p(pf) \rightarrow x_1$$

$$p_2(x) = M_p(pf, x_1) \rightarrow x_2$$

2. Large Model Forward Verification: The target model $q$ takes the entire draft sequence ($pf, x_1, x_2, \dots, x_K$) and runs one single forward pass in parallel to generate its own set of logits for all $K$ positions.


$$q_1(x), q_2(x), \dots, q_K(x) = M_q(pf, x_1, \dots, x_K)$$

3. Sampling and Verification (Sampling): The system compares the probability distributions of the draft model $p(x)$ and the target model $q(x)$ token by token. If $q(x) \ge p(x)$, the token is accepted. If $q(x) < p(x)$, the token is accepted with probability $\frac{q(x)}{p(x)}$. If the token is rejected in Case 2, the system discards it and all subsequent draft tokens. It then resamples the correct token from a normalized difference distribution: $\text{norm}(\max(0, q(x) - p(x)))$. Why do we resample from $(q(x) - p(x))_+$ when a token is rejected? The ultimate objective is to mathematically guarantee that the final generated token follows the exact distribution of the target model $q(x)$. The draft model $p$ sometimes overestimates the probability of certain tokens (the red area where $p > q$). When a token is rejected (because $q(x) < p(x)$ and it fails the probability check), it means we need to "make up" for the parts of the $q$ distribution that the draft model missed or underestimated.  The remaining area needed to perfectly match $q(x)$ is exactly $(q(x) - p(x))_+$.

To understand how to improve Speculative Decoding, we look at the formula for Single Token Latency:


$$L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}$$

- $T_{\text{draft}}$: Time taken by the draft model to generate a block.

- $T_{\text{verify}}$: Time taken by the target model for one verification pass.

- $\tau$: The average number of accepted tokens per verification round (accepted length).

To lower latency, we must: lower $T_{\text{draft}}$, lower $T_{\text{verify}}$, or raise $\tau$.

Existing head-based drafters face a strict trade-off between causality (which increases $\tau$) and efficiency (which lowers $T_{\text{draft}}$):

- Auto-regressive Drafter, e.g., EAGLE: Predicts tokens serially. Every token depends on the previous one. This maintains causality ($\tau$ is high), but $T_{\text{draft}}$ scales linearly with block length. You can only use short blocks and shallow networks to keep it fast.

- Parallel Drafter, e.g., DFlash: Inspired by block diffusion, it generates all tokens in a block simultaneously in one forward pass. $T_{\text{draft}}$ is incredibly low and independent of block length. However, because predictions are independent (no causal constraints), it easily generates contradictory drafts, severely suppressing the acceptance rate $\tau$.

DFlash attempts to solve this by having the target model pre-calculate hidden states, projecting them into context features ($H_{\text{ctx}}$), and injecting them into the draft layers. However, the positions within the block are still independent. For example, a parallel drafter might evaluate the phrase "of course / no problem". Both "course" and "problem" have high individual probabilities, resulting in a cross-pattern collision: "of problem".

DeepSeek's DSpark framework tackles both sides of the equation simultaneously: it uses Semi-autoregressive generation to raise $\tau$, and Confidence scheduling verification to lower effective $T_{\text{verify}}$. To fix the DFlash flaw, DSpark splits draft generation into two stages: Parallel Backbone + Serial Dependency. 

1. Parallel Phase: The DFlash backbone runs a single forward pass to generate base logits $U_k$ for every position $k$.

2. Serial Phase: It adds a Transition Bias ($B_k$) to the base logits. This bias conditions each position on the tokens already sampled within the block, inducing a true auto-regressive distribution:

$$p_k(v \mid x_0, x_{<k}) = \frac{\exp(U_k(v) + B_k(x_0, x_{<k}, v))}{\sum_{u \in \mathcal{V}} \exp(U_k(u) + B_k(x_0, x_{<k}, u))}$$

The Markov Head Implementation:
The simplest way to calculate $B_k$ is using a Markov head, where the bias only depends on the immediately preceding token $B(x_{k-1}, x_k)$. This is brilliantly implemented as a low-rank factorization $B = W_1 W_2$. It requires just one Embedding lookup and one Linear projection. Essentially, at the cost of one low-rank lookup, it adds the auto-regressive dependency of Eagle3 back onto the parallel backbone of DFlash. This retains per-token softmax probabilities and solves the "of problem" collision.

Even with semi-autoregressive drafting, verifying a massive block is expensive. If the draft gets rejected early, verifying the tail end of the block wastes GPU compute and crowds batch capacity. DSpark trains a "Confidence Head" that outputs a scalar $c_k \in (0, 1)$ for every draft position $k$. This models the conditional acceptance rate (the probability that position $k$ passes verification, assuming all prior tokens passed). A hardware-aware prefix scheduler uses these confidence scores and the current system load to decide how much of the draft block to actually verify. It only verifies the high-confidence prefix (e.g., E, F, G). It drops the low-confidence tail (e.g., H). This forces the target model's compute power exclusively toward tokens with a high probability of acceptance, dynamically optimizing $T_{\text{verify}}$ based on the live environment.

Speculative Decoding is an elegant statistical trick to trade cheap compute (drafting) for expensive compute (target verification). However, the evolution from basic Speculative Sampling to DSpark highlights a critical engineering journey: balancing the speed of parallel drafting with the logical necessity of causal, auto-regressive relationships, while dynamically managing hardware load through confidence scoring.

#### 5.2.1.2 Early Exit

Early Exit is an inference optimization strategy for Large Language Models. Its primary purpose on the inference side is to allow the model to make an early exit during the deduction process by deliberately skipping some model layers. By not forcing every single token to pass through every single layer of a massive neural network, the system can significantly speed up response times without needing any extra layers or modules. The essence of early stopping is "allowing tokens to cease computation as soon as their hidden states reach saturation". Essentially, if the model is already highly confident about what the next word should be after processing it through 10 layers, it is a waste of computational resources to push it through the remaining 20 layers.

While the idea is intuitive, applying it to modern, massive LLMs is incredibly difficult. Modern GPUs achieve speed through batching (processing many requests simultaneously). If you allow a single sample within a batch to exit early, it breaks the uniformity of the matrix math and brings immense challenges to scheduling. Existing token-level exit strategies usually rely on ML classifiers to decide when to stop. These classifiers are unpredictable and cannot guarantee worst-case scenarios. The KV Cache Recomputation Problem  is the most severe blocker. In standard LLM decoding, the generation of the next token depends on the Key-Value (KV) cache of the previous token. If token $A$ exits at layer 10, but the next token $B$ needs to compute up to layer 20, token $B$ will suddenly find that the KV cache for token $A$ is missing for layers 11-20. The system would be forced to pause and recompute token $A$'s missing KV cache, entirely destroying any speed gained by exiting early.

To solve these challenges, researchers introduced a new method called SkipDecode. This method is built on a very specific linguistic insight regarding how LLMs generate text. Words towards the end of a sequence are generally easier to predict due to more contextual information. When you are at the end of a long sentence, the context is so rich that the model doesn't need to think deeply (use all its layers) to guess the next word. Based on this insight, SkipDecode proposes two structural changes to how we skip layers.

- Monotonically Decreasing Exit Points. Instead of letting an ML classifier guess when to exit, SkipDecode enforces a strict, mathematical rule: the exit point must strictly decrease (or stay the same) as the sequence gets longer. Early in the prompt (low sequence position), the model uses the max_layer to understand the complex context. As the sequence grows, the computational budget drops linearly toward the min_layer.

- Skipping Lower Layers, Not Top Layers. Standard early exit models stop computation halfway up the network (Early Termination). SkipDecode does the exact opposite. In order not to waste the KV calculated by previous tokens, it will choose to skip the lower layers.

By setting a singular exit point for every token in a batch at each sequence position, the matrix math remains perfectly uniform. All tokens in a column exit at the exact same time. Because the exit layers strictly decrease over time, token $n$ will never need to compute deeper than token $n-1$. This strictly guarantees that the problem of KV cache recomputation does not occur. Furthermore, by skipping the bottom layers and enforcing computation at the top layers, the model implicitly attends to the full, rich computation of previous tokens without needing to recalculate them.

#### 5.2.1.3 Switch Transformer

As deep learning scales, researchers have discovered a clear power-law relationship between model size, dataset size, and computational volume versus overall performance. Following these established scaling laws, the Switch Transformer explores a highly efficient fourth dimension of scaling: Increasing the number of parameters while keeping the floating-point operations (FLOPs) per sample constant. By utilizing a sparse activation model designed for dense matrix multiplication hardware (like GPUs and TPUs), the model scales elegantly. As the number of devices increases, the total weight of the model also grows, while the memory and computational overhead on each device remains manageable.

To understand the Switch Transformer, we must first look at the standard Mixture of Experts (MoE) layer proposed in 2017. In a standard MoE layer, a token $x$ is received and routed to the best $k$ experts out of a total of $N$ experts $\{E_i(x)\}_{i=1}^N$.
First, a routing variable $W_r$ generates logits:


$$h(x) = W_r \cdot x$$

Then, the distribution is normalized by applying a softmax over all $N$ experts in that layer. The gate value (probability) for expert $i$ is calculated as:


$$p_i(x) = \frac{e^{h(x)_i}}{\sum_{j=1}^N e^{h(x)_j}}$$

If $\mathcal{T}$ is the set of indices for the selected top-$k$ experts, the output of the layer is the weighted linear combination of those experts based on their gate values:



$$y = \sum_{i \in \mathcal{T}} p_i(x)E_i(x)$$


Previous works suggested that the routing function needs to route to $k > 1$ experts (usually top-2) to maintain non-trivial gradients and model quality. However, the authors of the Switch Transformer chose to route to ONLY a single expert ($k=1$). They proved that this simplification not only maintains model quality but also reduces routing computation and improves performance. This $k=1$ strategy is called the Switch layer, and it has three major advantages:

1. Routing computation is reduced: Because each token is routed to only one expert.

2. Each expert's capacity can be at least halved: Because each token is assigned to only one expert rather than duplicated across multiple.

3. The routing implementation is simpler, and communication costs are lower.

To efficiently distribute this model, the authors used the MTF (Mesh-TensorFlow) library. They achieved this by abstracting the set of physical cores into a logical processor network, allowing tensors and computations to be easily sharded across dimensions. Because routing decisions are dynamic during training and inference, an important technical problem is how to set expert capacity. Expert capacity is defined by equally dividing the tokens in a batch by the number of experts, and then multiplying by a capacity factor:


$$\text{expert capacity} = \left( \frac{\text{tokens per batch}}{\text{number of experts}} \right) \times \text{capacity factor}$$

- Handling Overflow: If the capacity factor is $>1$, it introduces a buffer. If a certain expert receives too many tokens (here the authors refer to them as 'dropped tokens'), the expert skips computing them. Instead, the token's representation is passed directly to the next layer via a residual connection.

- The Trade-off: While a higher capacity factor prevents dropped tokens, increasing expert capacity also has drawbacks: excessively high values lead to wasted computation and memory.

To prevent all tokens from being routed to just one or two popular experts, the system must enforce load balancing. For each Switch layer, an auxiliary loss is added to the total model loss during training. Given $N$ experts and $T$ tokens in batch $B$, the auxiliary loss is the scaled dot product of vectors $f$ and $P$:



$$\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i$$

$f_i = \frac{1}{T} \sum_{x \in B} \mathbf{1}\{\arg\max p(x) = i\}$ (The actual proportion of tokens assigned to expert $i$).

$P_i = \frac{1}{T} \sum_{x \in B} p_i(x)$ (The average softmax probability routed to expert $i$).

Ideally, routing is uniform, meaning both vectors equal $\frac{1}{N}$.
In this objective function, the $P$ vector is differentiable, while the $f$ vector is non-differentiable. The final loss is multiplied by the number of experts $N$ to keep the loss value stable when the number of experts changes. (Under uniform routing: $\sum_{i=1}^N (f_i \cdot P_i) = \sum (\frac{1}{N} \cdot \frac{1}{N}) = \frac{1}{N}$. Multiplying by $N$ normalizes this back to $1$). Finally, the hyperparameter $\alpha$ is the multiplicative coefficient for these auxiliary losses, typically set to $10^{-2}$ to ensure balance without overwhelming the main cross-entropy objective.

Sparse expert models can be difficult to train due to hard switching decisions, and low-precision formats like BF16 can exacerbate instability in the router's softmax calculations. To fix this, the authors introduced Selective Precision.
Stability can be achieved by selectively converting to FP32 precision only in local regions of the model, while avoiding the high communication cost brought by FP32 tensors.

- The router input is cast to FP32.

- It generates the dispatch and combine tensors used for expert selection and result recombination.

- Crucially, FP32 precision is only used inside the router function. Once the local device computation is done, the output is cast back to BF16 before being broadcasted over the network. This secures the stability of FP32 without the massive network overhead.

Based on the experiments, the Switch Transformer yields three major conclusions:

1. The Switch Transformer outperforms carefully tuned Dense models and MoE Transformers on the speed-quality trade-off.

2. The computational overhead of the Switch Transformer is less than that of the corresponding MoE model.

3. The Switch Transformer performs better under lower capacity factors of 1.0 and 1.25. This reflects that in large model scenarios where memory is scarce, capacity factors should be minimized as much as possible.

To scale models effectively to trillions of parameters, the authors had to implement advanced initialization and regularization techniques to prevent divergence and overfitting. Proper initialization is critical for successful deep learning training. The weight matrices in the Switch Transformer are initialized using a truncated normal distribution with a mean of $\mu = 0$ and a standard deviation of $\sigma = \sqrt{s/n}$ (where $s$ is a scaling hyperparameter and $n$ is the number of input units). As an extra measure against instability, the authors reduced the default Transformer initialization scaling factor $s = 1.0$ by 10 times. The average model quality (measured by negative log perplexity) is significantly improved, and the variance between different training runs is greatly reduced. This initialization scheme allowed them to safely scale from a 223M parameter baseline up to a massive 1T parameter model.

When pre-training on large datasets and fine-tuning on smaller downstream tasks, overfitting becomes a major issue. Because the Switch Transformer has far more parameters than a FLOP-matched dense baseline, it suffers from much more severe overfitting on these small downstream tasks. Simply adding dropout to all layers causes performance degradation. The authors proposed adding a high dropout rate exclusively inside the experts during fine-tuning. If using a smaller dropout rate of 0.1 at non-expert layers, and a larger dropout rate of 0.4 at expert layers, performance improvements are achieved across four small downstream tasks.

The ultimate test of the Switch Transformer is how well it scales. The model was trained on a massive C4 corpus containing over 180B target tokens to ensure it was not restricted by data limits. It's important to note that adding experts keeps the computation costs roughly the same (because only one expert is activated per token), but the router still has to calculate probabilities across all experts, resulting in a minor compute overhead of $\mathcal{O}(d_{model} \times \text{num experts})$. Under the premise of keeping the FLOPs per token constant, more parameters—i.e., more experts—can accelerate training. There is a massive advantage in scaling along this additional dimension of sparse parameters.  Increasing the number of experts can significantly improve the sample efficiency of the model. For example, a Switch-Base 64 expert model reaches the same performance at step 450k that a standard T5-Base model reaches at step 60k. This is equivalent to achieving a 7.5x speedup in terms of step time, meaning it learns much faster when observing the same amount of data.

While step-efficiency is great, sparse models introduce cross-device communication overhead. The ultimate question is: Under a fixed training time and compute budget, should one train a Dense model or a Sparse model?  The Switch Transformer achieves massive real-world speedups. The Switch-Base 64 expert model only needs 1/7 of the wall-clock time required by the T5-Base model to reach similar perplexity. What if we gave those hardware resources to a larger dense model instead of a sparse one? The authors compared the Switch-Base model against the much stronger T5-Large dense model. Although T5-Large uses 3.5 times more FLOPs per token, Switch-Base still has the advantage in sample efficiency, and achieved a 2.5x speedup.

#### 5.2.1.4 SoftMoE

While standard Transformer models excel in vision and language tasks, scaling their performance traditionally requires a massive, linear increase in computational cost.

Standard Sparse Mixture of Experts (MoE) architectures solve this by only activating a small subset of "expert" subnetworks per token. However, because traditional MoE relies on discrete routing mechanisms, it introduces severe challenges: training instability, dropped tokens, unbalanced expert loads, difficulties in scaling expert numbers, and poor performance during fine-tuning. To overcome these dilemmas, researchers introduced Soft MoE, a soft mixture of experts mechanism. The fundamental philosophy of Soft MoE is a shift from "hard" assignments to "soft" assignments. Instead of using hard routing to assign tokens to specific experts, it performs a linear weighted combination of all input tokens, and then distributes them to each expert for processing. This soft allocation method provides a host of benefits: it is fully end-to-end differentiable, trains more stably, and completely avoids token dropping and expert imbalance. Soft MoE maintains the high capacity and efficiency of standard MoE, but easily scales to more experts with minimal computational overhead.

To understand Soft MoE, we must look at how it calculates the weights. Let the input sequence of tokens be represented as $X \in \mathbb{R}^{m \times d}$, where $m$ is the number of tokens and $d$ is the dimensionality.
The Soft MoE layer contains $n$ expert functions $\{f_i: \mathbb{R}^d \rightarrow \mathbb{R}^d\}_{1:n}$.
Each expert processes $p$ "slots". Every slot corresponds to a $d$-dimensional parameter vector. All these parameter vectors together form a massive matrix $\Phi \in \mathbb{R}^{d \times (n \cdot p)}$. The input slots $\tilde{X} \in \mathbb{R}^{(n \cdot p) \times d}$ are constructed as convex combinations of the original $m$ input tokens $X$:

$$D_{ij} = \frac{\exp((X\Phi)_{ij})}{\sum_{i'=1}^m \exp((X\Phi)_{i'j})}, \quad \tilde{X} = D^\top X$$

- The matrix $D$ represents the Dispatch Weights. It is calculated by applying the softmax function to every column of the $X\Phi$ logits.

- Because of this, every single input slot is a weighted average of all input tokens.

Next, the corresponding expert function is applied to each slot (each row of $\tilde{X}$) to generate the output slots:

$$\tilde{Y}_i = f_{\lfloor i/p \rfloor}(\tilde{X}_i)$$

Finally, the output tokens $Y$ are generated as a convex combination of all $n \cdot p$ output slots $\tilde{Y}$:

$$C_{ij} = \frac{\exp((X\Phi)_{ij})}{\sum_{j'=1}^{n \cdot p} \exp((X\Phi)_{ij'})}, \quad Y = C\tilde{Y}$$

The matrix $C$ represents the Combine Weights. It is calculated by applying the softmax function to every row of the $X\Phi$ logits.

Following standard MoE design, the Soft MoE module is typically used to replace a portion of the MLP modules in a Transformer (usually the latter half).

The transition from discrete to soft routing unlocks five massive advantages:

- Fully Differentiable. Traditional sparse MoE algorithms solve token allocation using non-differentiable approximations (like Top-$k$ or Expert Choice). In contrast, all operations within the Soft MoE layer are continuous and completely differentiable. The weighted average generated by softmax weights is interpreted as a "soft assignment" rather than a "hard assignment."

- No Token Dropping. Classic routing mechanisms face severe issues with tokens being dropped (unassigned) or expert load imbalance (some experts receiving vastly more tokens). Soft MoE completely avoids these problems because every slot is filled with a weighted average of ALL input tokens, ensuring no token is dropped and every expert receives information.

- Highly Efficient. The computational cost of a Soft MoE layer is purely determined by the total number of slots. As long as the total number of slots is the same, whether it's a few experts with many slots, or many experts with few slots, the computational cost is identical. Furthermore, Soft MoE completely avoids slow sorting operations or Top-$k$ selections, making it noticeably faster and more hardware-friendly than traditional sparse MoE methods.

- Combines Sparse and Dense Traits. Soft MoE is technically not sparse, because every input token fractionally activates all model parameters via the weighted average slots. However, it isn't a standard Dense MoE either, because each physical expert only computes a fraction of the slots, maintaining high efficiency.

- Sequence-level Determinism. Traditional MoE forces tokens into fixed-size groups for load balancing. If a group contains tokens from different sequences, they compete for buffer spots, causing non-determinism. Because Soft MoE doesn't require this grouping mechanism, it achieves complete sequence-level determinism.

Assuming the cost of an expert function per token is $O(k)$, the time complexity of a Soft MoE layer is $O(mnpd + npk)$. If we set the number of slots per expert $p = O(m/n)$ (the ratio of tokens to experts), the total complexity simplifies to $O(m^2d + mk)$. Because each expert has independent parameters, increasing the number of experts $n$ (and adjusting $p$ accordingly) allows you to scale up the total model parameters WITHOUT increasing the time complexity. Even though there is an $m^2d$ term, it is equivalent to the compute cost of standard Self-Attention, meaning Soft MoE will not become the bottleneck of the Transformer. As shown in experiments, increasing experts from 8 to 4096 leaves Soft MoE throughput virtually unchanged.

Because MoE replaces the Feed-Forward Network in a Transformer, its input is usually layer-normalized. However, when the model dimension $d$ is very large, a mathematical stability issue occurs: As $d \rightarrow \infty$, the softmax output will tend to approach a one-hot vector. To fix this, Soft MoE applies an L2 Normalization across specific axes to both the input $X$ and the parameter matrix $\Phi$, multiplied by a trainable scale scalar. This maintains the "soft" property of the softmax and allows for larger dimensions and higher learning rates. When the number of experts scales massively, the model cannot fit on a single device. Just like standard large-scale MoE training, Soft MoE perfectly supports standard model parallelism techniques to distribute experts across multiple GPUs/TPUs.

#### 5.2.1.5 DeepSeekMoE

As Large Language Models (LLMs) continue to scale, the Mixture of Experts (MoE) architecture has become a prominent method to drastically increase model capacity while keeping computational costs (FLOPs) manageable. However, traditional MoE routing has inherent flaws regarding how experts specialize and share knowledge. DeepSeekMoE was introduced to solve these structural inefficiencies. By redesigning how experts are segmented and how general knowledge is routed, it creates a highly specialized and vastly more efficient sparse model.

In a standard dense Transformer, each layer consists of a Self-Attention module and a Feed-Forward Network (FFN). A traditional MoE layer modifies this by replacing the single FFN with multiple expert subnetworks. For a given token, a routing mechanism (Gating Network) selects only the Top-$K$ experts to activate. The output hidden state $\mathbf{h}_t^l$ for token $t$ at layer $l$ is calculated as:


$$\mathbf{h}_t^l = \sum_{i=1}^N (g_{i,t} \cdot \text{FFN}_i(\mathbf{u}_t^l)) + \mathbf{u}_t^l$$

The gating value $g_{i,t}$ is determined by a softmax affinity score $s_{i,t}$, and is strictly zero for experts not in the Top-$K$:

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} \mid 1 \le j \le N\}, K) \\ 0, & \text{otherwise} \end{cases}$$

$$s_{i,t} = \text{Softmax}_i((\mathbf{u}_t^l)^\top \mathbf{e}_i^l)$$

(Where $N$ is the total number of experts, $\mathbf{e}_i^l$ is the centroid vector of the $i$-th expert, and $\mathbf{u}_t^l$ is the token's hidden state after Self-Attention).

This guarantees sparse activation, meaning only $K$ out of $N$ gating values are non-zero.

To improve upon this, DeepSeekMoE introduces two massive architectural shifts: Fine-Grained Expert Segmentation and Shared Expert Isolation. These two strategies can elevate the degree of expert specialization.

In standard MoE, a token assigned to an expert might cover a wide variety of knowledge domains. Therefore, this expert tends to learn widely differing knowledge in its parameters, which is hard to utilize effectively at the same time. If each token can be routed to more experts, different types of knowledge have the potential to be decoupled and learned in different experts. DeepSeekMoE achieves this by taking standard experts and slicing their intermediate hidden dimensions into $m$ smaller fragments, shrinking them to $\frac{1}{m}$ of their original size. To keep the computational cost perfectly identical, the number of activated experts is multiplied by $m$. The total experts become $mN$, and the activated experts become $mK$:



$$\mathbf{h}_t^l = \sum_{i=1}^{mN} (g_{i,t} \cdot \text{FFN}_i(\mathbf{u}_t^l)) + \mathbf{u}_t^l$$

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} \mid 1 \le j \le mN\}, mK) \\ 0, & \text{otherwise} \end{cases}$$

This fine-grained slicing drastically increases routing flexibility. Assume a baseline of $N=16$ experts using a Top-$2$ routing strategy. This yields $\binom{16}{2} = 120$ possible routing combinations. If we slice each expert into 4 smaller experts ($m=4$), we now have $64$ total experts and route to Top-$8$. This yields $\binom{64}{8} = \mathbf{4,426,165,368}$ possible combinations!
This massive surge in flexibility allows the model to capture highly targeted, precise knowledge.

In traditional routing, tokens sent to different experts often still require baseline context or common knowledge. As a result, multiple experts might converge to learn the same shared knowledge in their own parameters, leading to parameter redundancy across experts. If we set up dedicated shared experts to capture and consolidate general knowledge across different contexts, this parameter redundancy is mitigated, leaving the routed experts to focus entirely on specialized tasks.

The Final DeepSeekMoE Equation:
DeepSeekMoE isolates $K_s$ experts specifically as "Shared Experts." Every single token is deterministically passed through these shared experts. To keep the total compute constant, the number of dynamically routed experts is simply reduced by $K_s$.

$$\mathbf{h}_t^l = \sum_{i=1}^{K_s} \text{FFN}_i(\mathbf{u}_t^l) + \sum_{i=K_s+1}^{mN} (g_{i,t} \cdot \text{FFN}_i(\mathbf{u}_t^l)) + \mathbf{u}_t^l$$

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} \mid K_s+1 \le j \le mN\}, mK - K_s) \\ 0, & \text{otherwise} \end{cases}$$

Automatically learned routing risks a fatal flaw known as Routing Collapse, where the model only ever picks a few favorite experts, leaving the rest untrained. Furthermore, unbalanced routing across distributed GPUs creates massive communication bottlenecks. DeepSeekMoE implements two levels of balance loss to fix this. To prevent routing collapse, the model calculates the load distribution across experts:

$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i$$

- $f_i = \frac{N'}{K'T} \sum_{t=1}^T \mathbb{1}(\text{Token } t \text{ select Expert } i)$ (The fraction of times expert $i$ is selected)

- $P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}$ (The average routing probability for expert $i$)
(Where $N' = mN - K_s$ and $K' = mK - K_s$)

There is no need to impose strict balance constraints at the expert level because overly strong load-balancing constraints will damage model performance. Instead, we just need to ensure the total computation happening on each physical GPU is balanced. If all routed experts are divided into $D$ groups $\{E_1, E_2, \dots, E_D\}$ corresponding to physical devices, the device-level loss is:

$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^D f_i' P_i'$$

- $f_i' = \frac{1}{|E_i|} \sum_{j \in E_i} f_j$

- $P_i' = \sum_{j \in E_i} P_j$

The Hyperparameter Strategy (Purple Text):
DeepSeekMoE sets a smaller expert-level balance factor to mitigate routing collapse risk, while simultaneously setting a larger device-level balance factor to promote computation balance across devices. This ensures high hardware utilization without stifling the model's natural ability to assign specialized tasks unevenly when mathematically necessary.

#### 5.2.1.6 KV Cache

During the inference phase of Large Language Models (LLMs) based on the Transformer architecture, the system generates text auto-regressively—one token at a time. This process is inherently slow and resource-heavy. To combat this, the industry standard is to use a KV Cache (Key-Value Cache).

In a standard Self-Attention mechanism, generating a new token requires calculating the attention scores between the current Query (Q) and the Keys (K) and Values (V) of all previous tokens in the sequence.

- Without Cache (The Problem): Every time a new token is generated, the model recalculates the Keys and Values for all historical tokens from scratch. This redundant computation causes the computational complexity to grow quadratically ($O(n^2)$), severely slowing down inference.

- With KV Cache (The Solution): The model saves the historical Keys and Values in GPU memory. When generating the next token, the model only needs to calculate the Q, K, and V for the new token. It then interacts with the cached historical K and V data.

The core purpose of KV Cache is to avoid repeated calculations, thereby greatly improving efficiency. By trading memory for compute, the mathematical complexity of generation is reduced from quadratic to linear. To truly understand KV Cache, we must look at the two distinct stages of LLM inference: Prefill and Decode. Modern optimization heavily relies on PD Separation, which decouples these two tasks physically or logically.

The Prefill stage is responsible for parsing the user's input Prompt.

- Characteristics: It is a Compute-bound process. Because the input prompt is static and known, it heavily relies on dense matrix multiplications (GEMM), making massive demands on GPU computing power (FLOPs). It has a Static feature, allowing prompts to be processed in batches for high throughput.

- The Output: Once computation is complete, this stage generates the initial Key-Value Cache (KV Cache), which is provided for repeated use in the subsequent decoding stage to avoid repeated calculations.

The Decode stage is the auto-regressive, token-by-token generation phase.

- Characteristics: It is a Memory-bound process. Generating each new token requires reading the massive KV Cache stored in the GPU's High Bandwidth Memory (HBM). Its Dynamic feature makes it incredibly difficult to batch efficiently compared to the Prefill stage.

- The Benefit: Because of the cache, the model doesn't need to recalculate Key and Value for all historical tokens, and can directly reuse Prefill stage results, drastically cutting down compute overhead.

Advanced architectures utilize Level 3 PD Separation, routing the Compute-bound Prefill tasks to heterogeneous hardware like CPUs or FPGAs, while dedicating GPUs strictly to the Memory-bound Decode tasks.

While KV Cache saves compute time, it requires an astronomical amount of VRAM (Video RAM). The memory requirement of KV Cache expands linearly with the sequence length and batch size, often growing to multiple times the size of the model weights themselves.

The exact total size of the KV Cache is determined by the following formula:

$$\text{KVCache Size} = 2 \times \text{precision} \times n_{layers} \times d_{model} \times seqlen \times batch$$

Formula Breakdown:

- $2$: We must cache two vectors: one for Key and one for Value.

- $\text{precision}$: The byte size of the data format (e.g., FP16 = 2 bytes).

- $n_{layers}$: The number of layers in the model (every attention mechanism in every layer caches its own K and V).

- $d_{model}$: The hidden dimension of the model (often equal to $\text{numheads} \times \text{headdimension}$).

- $seqlen$: The sequence length ($s + n$, where $s$ is input length and $n$ is generated length).

- $batch$: The batch size (number of sequences processed simultaneously).

Case Study: GPT-3 (175B Parameters)
Let's assume we are running GPT-3 with FP16 precision ($2$ bytes), a batch size of $4$, an input sequence length of $4096$, $96$ layers, and a $d_{model}$ of $12288$ ($96$ heads $\times 128$ dim).



$$\text{KV Cache Size} = 2 \times 2 \times 96 \times 12288 \times 4096 \times 4$$

$$\text{KV Cache Size} = 77,317,495,808 \text{ bytes} \approx \mathbf{72 \text{ GB}}$$

There are  three reasons why managing KV Cache is the most critical hurdle in modern AI deployment:

- The LLM Context Window Trend: The length of context windows is constantly growing. OpenAI's API frequently sees dynamic lengths around 10k tokens (including prefill prompts and conversation history). There is a direct contradiction between the ever-growing context demands and the limited physical GPU VRAM.

- Consumer GPU Limitations: For cost-effective consumer GPUs like the RTX 4090, VRAM is severely limited (24GB). As a result, the massive footprint of KV Cache reduces the model's batch size. to prevent out-of-memory errors.

- Crucial for AIGC & Multi-modal Generation: This issue isn't just for text. Video and image generation models like Sora or Stable Diffusion have relatively small parameters ($<10B$) but incredibly long generation sequences (approaching $1000k$ patches). In these models, KV cache can account for 90% or more of the total VRAM usage.

Because KV Cache dominates VRAM, researchers are attacking the problem through four main avenues:

- Storage Compression: Moving away from FP16 and using highly efficient, lower bit-width numeric representations like FP8, cutting the memory requirement exactly in half.

- Paged/Tiered Caching: Introducing multi-tier memory systems (like swapping data between fast GPU VRAM and slower CPU Host Memory) to securely store massive caches without crashing the GPU.

- Block Storage: Chunking long sequence tasks to reduce peak memory demands (similar to techniques like FastGen).

- Inference Algorithm Optimization: Exploiting the low-rank or sparse nature of attention mechanisms to permanently drop or ignore historical keys/values that are mathematically deemed unimportant, shrinking the cache size dynamically.

To fully grasp why Inference Algorithm Optimization is critical, we must understand the physical constraints of GPU hardware during the decoding phase. In an ideal scenario, computation dictates speed. However, when performing matrix multiplications during decoding, the raw floating-point performance (FLOPS) of CUDA or Tensor Cores is often vastly greater than the data throughput the GPU's memory can provide. This creates a Memory-bound situation:

- Compute Units Idle: The compute units are frequently in a state of "idling" or "waiting for data", unable to operate at full load.

- Massive Data Transfer: Because parameters and intermediate activations must be read from Video RAM (VRAM) for every single step, massive amounts of data must be moved continuously.

Therefore, from a hardware perspective, we can approximate that:

$$\text{Inference Speed} \propto \frac{\text{Memory Bandwidth}}{\text{Data required per generated token}}$$

To understand this physically, we can construct a rough estimation formula for single-step latency ($t_{\text{model}}$). If we only consider the time it takes to read the model's parameters once:



$$t_{\text{model}} = \frac{\text{ModelSize}}{\text{BW (Memory Bandwidth)}}$$

Example: If a GPU has a bandwidth of 1 TB/s ($1000 \text{ GB/s}$) and the model size is 14.2 GB, the theoretical minimum latency per token is $14.2 \text{ GB} / 1000 \text{ GB/s} = 0.0142 \text{ s} \approx \mathbf{14.2 \text{ ms}}$.

(Note: This is a highly idealized theoretical lower limit. It ignores instruction scheduling, thread switching, parallel efficiency, and kernel launch delays).

While model weights require a constant read time per token, KV Cache introduces a Linearly Increasing Overhead.

- Step 1: The model reads the prefix context (Prompt).

- Step 2: Generating token 2 requires reading 1 set of KV Cache.

- Step $n$: Generating token $n$ requires reading $(n-1)$ sets of KV Cache! Because deep networks must visit this KV cache at every single layer, the total memory access volume scales aggressively.

For a standard 7B model, loading 14 GB of weight data completely overshadows the ~130 MB of KV cache in early generation stages. However, in much larger models (30B, 70B, 100B+) or in multi-turn chat scenarios where the context length is incredibly long, the proportion of latency caused by reading the KV cache continuously grows.

How many milliseconds or seconds a single inference takes is usually directly and highly correlated with these three items: 'Model Size', 'Context Length', and 'GPU Bandwidth'. This fundamental physical limitation is why aggressive engineering optimizations—such as quantization, Tensor Parallelism, Pipeline Parallelism, and Algorithm Optimizations (dropping unimportant keys)—are absolutely mandatory to survive in modern LLM deployment.

### 5.2.2 System Optimization

#### 5.2.2.1 Iteration-Level Batching

To understand the massive optimization brought about by Iteration-Level Batching (also referred to as Dynamic Batching or continuous batching), we must first understand the fundamental mechanics of Large Language Model (LLM) inference and the severe limitations of traditional scheduling. For Causal Decoder-only models, the inference process is strictly divided into two distinct phases:

- Prefill Stage: The model processes the user's initial prompt all at once.

- Decode Stage: The LLM generates the output text auto-regressively—one token at a time—until it encounters an <EOS> (End of Sentence) token or hits the maximum length limit.

LLM inference is Memory I/O bound, not Compute bound. Loading 1MB of weights into the GPU takes longer than actually performing the math on that 1MB of data. Therefore, the throughput of an LLM is heavily dependent on how much batch data you can cram into the high-speed GPU memory. However, GPU memory consumption increases with both model size and token length. If you limit sequence length to 512, you might fit 28 sequences in a batch. If you raise it to 2048, you might only fit 7. When processing multiple user requests (sequences) simultaneously, the traditional approach is Static Batching. The fundamental flaw of Static Batching is that the LLM doesn't know when a sequence will finish generating its <EOS> token.

When you batch multiple different requests together, some requests will naturally finish much earlier than others. Because it is a static batch, the GPU cannot release the computational resources for shorter sequences until the longest sequence is completely finished. The white empty squares after the red END markers represent the GPU sitting completely idle doing nothing. This means the GPU is severely underutilized. Traditional static batching cannot utilize the white idle time. If you are running a chat application where answer lengths vary wildly, this static method ruins GPU efficiency.

To solve the massive waste of GPU cycles seen in Static Batching, the industry moved to Dynamic Batching (often referred to as continuous batching or iteration-level batching). The concept is beautifully simple: "Once a certain sequence in a batch finishes generating and produces an <EOS> token, you can insert a new sequence in its place to continue generating tokens". By dynamically swapping in new requests at the exact iteration (token generation step) that an old request finishes, the system completely eliminates the "blank cells." This achieves a much higher GPU utilization rate than static batching. 

Reality is more complicated than this simplified model. Because the Prefill stage requires different computation patterns than the Decode stage, it cannot be easily batched together with token generation. This specific challenge leads to even more advanced techniques, such as Chunked Prefill.

#### 5.2.2.2 Chunked Prefill (SARATHI)

Building upon the concepts of Iteration-Level Batching (Dynamic Batching), Chunked Prefill addresses a major flaw in how modern Large Language Models handle the stark differences between processing a prompt and generating a response. Standard Iteration-Level Batching processes requests token-by-token. However, this process fails to consider the fundamental differences between the prefill and decode stages. When we analyze the specific characteristics of these two stages, two critical conclusions emerge:

- Prefill saturates the GPU easily, Decode starves it. Because the Prefill stage processes all the tokens in the user's input prompt in parallel, a very small batch size will max out GPU utilization. Conversely, in the decode stage (with KV Cache enabled), every auto-regressive step only generates exactly one token. Therefore, GPU utilization is very low.

- During Decode, the computational cost of generating a single token is significantly higher than processing a single token during the Prefill stage. As you increase the batch size, the per-token cost of Prefill remains almost completely constant. The per-token cost of Decode drops dramatically. The prefill stage saturates GPU efficiency at a very small batch size, while the decode stage only saturates it at a very large batch size. Prefill takes a long input sequence $L$, resulting in heavy compute (Compute-bound). Decode takes an input length of $1$ but must repeatedly read the massive KV Cache, resulting in heavy Memory IO overhead (Memory-bound).

Because the Prefill stage for a long prompt takes a significantly longer time than a single Decode step, throwing them both into a standard pipeline creates massive computational waste. Due to the existence of the long-latency prefill stage, computation bubbles exist when performing pipeline parallelism. To eliminate these massive bubbles, the industry introduced Chunked Prefill, as demonstrated in the SARATHI schedule. Chunked Prefill introduces two revolutionary steps:

- Splitting the Prompt: It takes long, variable-length prompts and breaks them apart into uniformly sized, smaller "chunks" for prefilling.

- Piggybacking Decodes: Because these chunks are smaller, they leave tiny gaps (bubbles) in the computation schedule. The system inserts (piggybacks) the quick Decode requests of other sequences into these gaps.

You cannot simply chop up a prompt without consequences. Chunked prefill requires special handling of the attention mask because a single prefill is split into multiple times. Furthermore, Chunked Prefill slightly increases the overall overhead of the Prefill stage because the system must repeatedly load the KV Cache of previous chunks from GPU Memory into the Kernel to compute the current chunk.

If Chunked Prefill increases the overhead of the Prefill stage, why do we use it? We do it because piggybacking decode requests onto prefill chunks is overwhelmingly beneficial for the Decode stage. During decoding, the GPU memory overhead comes from fetching the KV Cache and fetching the massive model weights (parameters). When we piggyback a decode task alongside a prefill chunk, it can directly reuse the model parameters fetched during the prefill stage. Doing so can almost convert decode from a memory-bound operation to a compute-bound operation. 

#### 5.2.2.3 FlashAttention

FlashAttention-1 computes the same dense, softmax-based attention as ordinary attention, but it computes it in a much more GPU-friendly order. It does not replace attention with an approximation. Instead, it:

- Divides Q, K, and V into small blocks.
- Loads those blocks from relatively slow GPU HBM into very fast on-chip SRAM.
- Fuses score calculation, masking, softmax, dropout, and multiplication by V.
- Never writes the full N×N score matrix S or probability matrix P to HBM.
- Keeps only a small running maximum, a running softmax denominator, and the partial output for each query row.
- Recomputes S and P during backpropagation instead of saving them.

The key philosophy is: Doing a little more arithmetic can be faster when it avoids moving a great deal of data.

FlashAttention therefore leaves the dense-attention arithmetic complexity at approximately O(N2d), but drastically reduces memory traffic and removes the quadratic-size saved intermediates. It is mathematically exact, apart from ordinary floating-point differences caused by changing the order of operations.

| Memory level             |                             Approximate capacity shown in the slides | Approximate bandwidth shown | Main role                                                       |
| ------------------------ | -------------------------------------------------------------------: | --------------------------: | --------------------------------------------------------------- |
| CPU main memory, or DRAM |                                                   **More than 1 TB** |               **12.8 GB/s** | Very large, but far from the GPU                                |
| GPU HBM                  |                                                         **40–80 GB** |            **1.5–2.0 TB/s** | Holds model activations, (Q,K,V,O), and normal GPU tensors      |
| On-chip GPU SRAM         | **192 KB per SM**, across **108 SMs**, approximately **20 MB total** |   Approximately **19 TB/s** | Small, extremely fast workspace for currently executing kernels |

<p align="center">
<img width="521" height="377" alt="image" src="https://github.com/user-attachments/assets/6215782f-8802-4269-a944-2dcc05d042a3" />
</p>

The smaller a memory level is, the faster it tends to be. On the A100, on-chip SRAM is roughly an order of magnitude faster than HBM, but its capacity is thousands of times smaller. The A100 figures in the original FlashAttention paper are 40–80 GB of HBM at 1.5–2.0 TB/s, and 192 KB of combined on-chip storage for each of 108 streaming multiprocessors, with estimated aggregate bandwidth around 19 TB/s. A hardware nuance is that the quoted 192 KB is the combined L1, texture-cache, and shared-memory structure. CUDA exposes at most about 164 KB of shared memory per A100 SM, and a single thread block can use slightly less than that. Thus, “SRAM” in the FlashAttention explanation is an abstraction covering fast shared memory, caches, and registers—not one globally accessible 20 MB pool.

A GPU operation is usually implemented as a kernel. A kernel generally:
1. Loads input data from HBM.
2. Places active data in registers and SRAM.
3. Performs calculations.
4. Writes results back to HBM.

Because modern GPUs have become extremely fast at matrix multiplication, many operations are limited not by arithmetic throughput but by how quickly data can be moved. The relevant metric is arithmetic intensity:

$\text{Arithmetic intensity} = \frac{\text{arithmetic operations}}{\text{bytes moved}}$

A high value means the GPU performs substantial computation for every byte loaded. A low value means it spends much of its time moving data.

For a compute-bound operation, arithmetic dominates runtime, while HBM access is a relatively small part of the cost. Examples are matrix multiplication with a large inner dimension, and convolution with many channels. For a memory-bound operation, runtime is determined mainly by memory reads and writes, while the arithmetic itself is comparatively cheap. Examples include: Activation functions, Dropout, Sum and other reductions, Softmax, Batch Normalization, Layer Normalization, etc. This distinction explains an apparent paradox in attention: most mathematical operations may be in matrix multiplications, but masking, softmax, dropout, and other elementwise operations can consume a surprisingly large fraction of wall-clock time because they repeatedly read and write the large attention matrix.

A conventional framework may execute attention as several kernels:

- Matrix multiplication
- Masking
- Softmax
- Dropout
- Another matrix multiplication

Each kernel may read its input from HBM and write its output back to HBM. Therefore, even inexpensive operations such as masking and dropout can be expensive when they operate on an N×N matrix. Although the two matrix multiplications contain most of the FLOPs, masking, softmax, and dropout occupy a substantial part of the PyTorch runtime. FlashAttention replaces the stack with a fused computation that avoids materializing the large intermediate attention matrix in HBM. The original paper reported a 7.6× speedup for the attention computation in the particular GPT-2 microbenchmark shown in that figure.

For one attention head, let

$$
Q, K, V \in \mathbb{R}^{N \times d}
$$

where:

- N is sequence length.
- d is the dimension of one attention head.

In a normal Transformer, the complete expression is usually closer to

$$
S = \frac{QK^\top}{\sqrt{d}} + \text{mask}
$$

$$
P = softmax_{\text{row}}(S)
$$

$$
O = Dropout(P)V
$$

The scaling factor, mask, and dropout do not change the central FlashAttention idea; they can be incorporated into the fused tile computation.

The standard implementation proceeds as follows.

1. Store Q, K, and V in HBM. The input matrices are normal GPU tensors and reside in the large HBM.

2. Compute the score matrix. Load blocks of Q and K from HBM to SRAM, compute

$$
S = QK^\top
$$

Write the complete score matrix

$$
S \in \mathbb{R}^{N \times N}
$$

to HBM.

3. Compute softmax. Read \(S\) back from HBM and compute

$$
P = \softmax(S)
$$

Then write the complete probability matrix

$$
P \in \mathbb{R}^{N \times N}
$$

to HBM.

4. Compute the output. Read blocks of \(P\) and \(V\) from HBM and compute

$$
O = PV
$$

Then write \(O\) to HBM. The problem is not merely that the matrices are large. It is that they are repeatedly transferred between HBM and on-chip memory.

Both \(S\) and \(P\) contain \(N^2\) elements for every batch item and every attention head.For batch size \(B\) and \(H\) attention heads, each of these intermediate tensors contains approximately

$$
BHN^2
$$

elements. As \(N\) grows:

- Arithmetic grows approximately as \(N^2d\).
- The storage required for \(S\) and \(P\) grows as \(N^2\).
- HBM traffic grows rapidly because \(S\) and \(P\) are written and read several times.
- Masking, softmax, and dropout also process \(N^2\) elements.

This is why attention is often largely **memory-bound**. Communication between HBM and SRAM becomes a major limiter of efficiency.

FlashAttention divides the matrices into blocks small enough to fit in fast on-chip memory. Within a tile, it fuses:

- \(QK^\top\)
- Scaling
- Masking
- Safe softmax
- Dropout
- Multiplication by \(V\)

This prevents the complete \(S\) and \(P\) matrices from ever being written to HBM. Ordinary training saves intermediate tensors because backpropagation needs them. FlashAttention instead saves only compact softmax statistics and recomputes the required \(S\) and \(P\) blocks during the backward pass. This resembles **gradient checkpointing**, but there is an important difference. Ordinary checkpointing often trades speed for lower memory usage. FlashAttention’s recomputation can still be faster because recomputing a tile on-chip may be cheaper than reading a huge saved matrix from HBM.

Suppose

$$
Q,K,V\in\mathbb{R}^{4\times3}.
$$

Divide \(Q\) into two row blocks:

$$
Q_1,Q_2\in\mathbb{R}^{2\times3}.
$$

Divide \(K\) and \(V\) into corresponding blocks:

$$
K_1,K_2,V_1,V_2\in\mathbb{R}^{2\times3}.
$$

Equivalently,

$$
K_1^\top,K_2^\top\in\mathbb{R}^{3\times2}.
$$

Ignoring softmax for a moment, one output block can be accumulated as

$$
O_i =
(Q_iK_1^\top)V_1
+
(Q_iK_2^\top)V_2.
$$

The loop structure is:

- Outer loop: fix one $\(K_j,V_j\)$ block.
- Inner loop: visit the different $\(Q_i\)$ blocks.
- Compute a partial output block.
- Add contributions from successive $\(K,V\)$ blocks.
- Concatenate the completed output row blocks to obtain $\(O\)$.

Every tile calculation contributes directly to part of the final output.

The complete $\(4\times4\)$ score matrix does not need to be retained.

Without softmax, partial matrix products can simply be added. With softmax, each query row must be normalized across all keys:

$$
P_{ij} =
\frac{e^{S_{ij}}}
{\sum_{k=1}^{N}e^{S_{ik}}}.
$$

A softmax computed independently inside each tile would use a different denominator for every tile. Those independently normalized results cannot simply be added. FlashAttention therefore needs a method for computing softmax incrementally while seeing only one block at a time. For a vector

$$
X=[x_1,x_2,\ldots,x_N],
$$

ordinary softmax is

$$
softmax(x_i) =
\frac{e^{x_i}}
{\sum_{j=1}^{N}e^{x_j}}.
$$

The largest finite value representable by FP16 is 65504. However,

$$
e^{12}\approx162755,
$$

which already exceeds that limit. Directly exponentiating moderately large scores can therefore overflow. The stable solution is safe softmax. Define

$$
m(X)=\max(x_1,x_2,\ldots,x_N).
$$

Then compute

$$
softmax(x_i) =
\frac{e^{x_i-m(X)}}
{\sum_{j=1}^{N}e^{x_j-m(X)}}.
$$


Subtracting the same constant from every score does not change softmax because the common exponential factor cancels. Choosing the maximum is especially useful because

$$
x_i-m(X)\leq0,
$$

so every exponential is at most \(1\). This operation is better described as a maximum shift than as conventional input normalization.

The safe-softmax process can be written as follows:

$$
X=[x_1,x_2,\ldots,x_N],
$$

$$
m(X)=\max(x_1,x_2,\ldots,x_N),
$$

$$
p(X) =
\left[
e^{x_1-m(X)},
\ldots,
e^{x_N-m(X)}
\right],
$$

$$
\ell(X)=\sum_i p(X)_i,
$$

$$
softmax(X) =
\frac{p(X)}{\ell(X)}.
$$

Here:

- $\(m(X)\)$ is the row maximum.
- $\(p(X)\)$ is the unnormalized, maximum-shifted exponential vector.
- $\(\ell(X)\)$ is the softmax denominator after maximum shifting.

Suppose one row is split into two blocks: 

$$
X=[X^1,X^2].
$$

For each block, independently calculate

$$
m_1=m(X^1),
\qquad
m_2=m(X^2),
$$

$$
p_1=e^{X^1-m_1},
\qquad
p_2=e^{X^2-m_2},
$$

$$
\ell_1=\sum p_1,
\qquad
\ell_2=\sum p_2.
$$

The global maximum is

$$
m=\max(m_1,m_2).
$$

The first block was expressed relative to $\(m_1\)$, while the second block was expressed relative to $\(m_2\)$. To put both blocks on the common global scale \(m\), rescale them:

$$
p(X) =
\left[
e^{m_1-m}p_1,
\;
e^{m_2-m}p_2
\right].
$$

The combined denominator is

$$
\ell(X) =
e^{m_1-m}\ell_1
+
e^{m_2-m}\ell_2.
$$

Therefore,

$$
softmax(X) =
\frac{p(X)}{\ell(X)}.
$$

This is the essential online softmax rule. A complete row can be processed piece by piece while retaining only a maximum and a sum for that row. FlashAttention applies this rule simultaneously to many rows in a query block.

FlashAttention does not need to preserve the complete vector \(p(X)\). It only needs the softmax-weighted value sum. For one query row, define the state after some key blocks have been processed:

$$
m =
\text{largest score seen so far},
$$

$$
\ell =
\sum_{\text{seen keys}}e^{s_k-m},
$$

$$
z =
\sum_{\text{seen keys}}e^{s_k-m}v_k,
$$

$$
o =
\frac{z}{\ell}.
$$

Here, \(o\) is the current normalized attention output. For a new block, calculate local statistics

$$
\widetilde m,
\qquad
\widetilde\ell,
\qquad
\widetilde z.
$$

Then merge the old and new states. The new maximum is

$$
m^{\text{new}} =
\max(m,\widetilde m).
$$

The new denominator is

$$
\ell^{\text{new}} =
e^{m-m^{\text{new}}}\ell
+
e^{\widetilde m-m^{\text{new}}}\widetilde\ell.
$$

The new weighted numerator is

$$
z^{\text{new}} =
e^{m-m^{\text{new}}}z
+
e^{\widetilde m-m^{\text{new}}}\widetilde z.
$$

The new normalized output is

$$
o^{\text{new}} =
\frac{z^{\text{new}}}
{\ell^{\text{new}}}.
$$

This is the mathematical heart of FlashAttention-1.

When a later block contains a larger score, the reference maximum changes.

The old contribution is not discarded. It is multiplied by the appropriate exponential correction so that it is expressed relative to the new maximum.

Complete FlashAttention-1 forward algorithm:

| Symbol | Meaning |
|---|---|
| \(N\) | Sequence length |
| \(d\) | Dimension of one attention head |
| \(M\) | Usable local on-chip memory in the paper’s abstract model |
| \(B_r\) | Number of query rows in one query block |
| \(B_c\) | Number of key/value rows in one key/value block |
| \(T_r=\lceil N/B_r\rceil\) | Number of query blocks |
| \(T_c=\lceil N/B_c\rceil\) | Number of key/value blocks |
| \(m_i\) | Running row maximum for query block \(i\) |
| \(\ell_i\) | Running softmax denominator for query block \(i\) |
| \(O_i\) | Running normalized output for query block \(i\) |

The pseudocode assumes one batch item and one attention head.

1. Inputs are stored in HBM. The matrices

$$
Q,K,V\in\mathbb{R}^{N\times d}
$$

reside in HBM. HBM is measured in gigabytes, so storing these $\(N\times d\)$ matrices is generally manageable. The larger problem is storing the $\(N\times N\)$ matrices $\(S\)$ and $\(P\)$.

2. Select row and column block sizes. The paper’s simplified rule is

$$
B_c =
\left\lceil
\frac{M}{4d}
\right\rceil,
$$

$$
B_r =
\min
\left(
\left\lceil
\frac{M}{4d}
\right\rceil,
d
\right).
$$

The factor $\(4d\)$ provides the intuition that on-chip memory must accommodate roughly four \(d\)-wide working blocks:

- A $\(Q\)$ block
- A $\(K\)$ block
- A $\(V\)$ block
- An $\(O\)$ block

The temporary score tile and softmax state must also fit, which is one reason $\(B_r\)$ is capped. Production kernels also choose hardware-friendly tile sizes instead of mechanically applying this theoretical expression. In addition, $\(M\)$ represents memory available to one executing work unit. It is not the aggregate on-chip memory distributed across the entire GPU.

3. Initialize the output and softmax statistics

Initialize the following values in HBM:

$$
O=0\in\mathbb{R}^{N\times d},
$$

$$
\ell=0\in\mathbb{R}^{N},
$$

$$
m=-\infty\in\mathbb{R}^{N}.
$$

Their meanings are:

- $\(O\)$: accumulated normalized attention output
- $\(\ell\)$: accumulated softmax denominator
- $\(m\)$: largest score encountered so far in each row

Initializing $\(m\)$ to $\(-\infty\)$ guarantees that the first real score becomes the running maximum.

4. Divide $\(Q\)$, $\(K\)$, and $\(V\)$ into blocks. Split \(Q\) into

$$
T_r =
\left\lceil
\frac{N}{B_r}
\right\rceil
$$

blocks:

$$
Q_1,\ldots,Q_{T_r},
\qquad
Q_i\in\mathbb{R}^{B_r\times d}.
$$

Split \(K\) and \(V\) into

$$
T_c =
\left\lceil
\frac{N}{B_c}
\right\rceil
$$

blocks:

$$
K_1,\ldots,K_{T_c},
\qquad
V_1,\ldots,V_{T_c},
$$

with

$$
K_j,V_j\in\mathbb{R}^{B_c\times d}.
$$

5. Divide $\(O\)$, $\(\ell\)$, and $\(m\)$

Partition \(O\) into blocks:

$$
O_1,\ldots,O_{T_r},
\qquad
O_i\in\mathbb{R}^{B_r\times d}.
$$

Divide $\(\ell\)$ and $\(m\)$ into corresponding vectors of length $\(B_r\)$.

These vectors contain the current state for the query rows in that block.

6. Begin the outer loop over $\(K,V\)$ blocks.

For

$$
j=1,\ldots,T_c,
$$

process one key/value block.

The outer loop moves across the $\(K\)$ and $\(V\)$ blocks.

7. Load $\(K_j,V_j\)$ from HBM into SRAM. Transfer the selected key and value blocks into fast on-chip memory. They can then be reused while processing multiple query blocks. Approximately 50% of SRAM remains available at this moment and is reserved for $\(Q\)$ and $\(O\)$. This is a conceptual allocation:

- One portion holds $\(K,V\)$.
- Another portion holds $\(Q,O\)$ and related state.

The exact fraction in a real kernel depends on tile shapes, score storage, registers, numerical precision, and occupancy.

8. Begin the inner loop over $\(Q\)$ blocks.  For

$$
i=1,\ldots,T_r,
$$

process one query block against the fixed $\(K_j,V_j\)$ block.

The inner loop moves across the $\(Q\)$ blocks.

9. Load the current query block and state.

Load

$$
Q_i,
\qquad
O_i,
\qquad
\ell_i,
\qquad
m_i
$$

from HBM into on-chip memory.

- $\(Q_i\)$ and $\(O_i\)$ occupy most of this part of the workspace.
- $\(\ell_i\)$ and $\(m_i\)$ are small vectors of length $\(B_r\)$.
- These small statistics can often be stored in registers.
  
10. Compute one score tile.

Compute on-chip:

$$
S_{ij} =
Q_iK_j^\top.
$$

The shapes are

$$
(B_r\times d)(d\times B_c) =
B_r\times B_c.
$$

Only this small score tile exists at one time.

The complete $\(N\times N\)$ score matrix is never constructed in HBM.

11. Compute local softmax statistics. For every row in \(S_{ij}\), compute the local maximum:

$$
\widetilde m_{ij} =
rowmax(S_{ij})
\in
\mathbb{R}^{B_r}.
$$

Compute local unnormalized probabilities:

$$
\widetilde P_{ij} =
\exp
\left(
S_{ij} -
\widetilde m_{ij}
\right)
\in
\mathbb{R}^{B_r\times B_c}.
$$

The row maximum is broadcast across every element in the corresponding row.

Then calculate the local row sums:

$$
\widetilde\ell_{ij} =
rowsum
\left(
\widetilde P_{ij}
\right)
\in
\mathbb{R}^{B_r}.
$$

All of these calculations remain on-chip.

12. Merge the current block with previous blocks. Update the running maximum:

$$
m_i^{\text{new}} =
\max
\left(
m_i,
\widetilde m_{ij}
\right).
$$

The maximum is calculated elementwise across the \(B_r\) rows.

Update the denominator:

$$
\ell_i^{\text{new}} =
e^{m_i-m_i^{\text{new}}}\ell_i
+
e^{\widetilde m_{ij}-m_i^{\text{new}}}
\widetilde\ell_{ij}.
$$

13. Update the output. The pseudocode writes the output update as

$$
O_i
\leftarrow
diag
\left(
\ell_i^{\text{new}}
\right)^{-1}
\left[
diag
\left(
\ell_i
\right)
e^{m_i-m_i^{\text{new}}}O_i
+
e^{\widetilde m_{ij}-m_i^{\text{new}}}
\widetilde P_{ij}V_j
\right].
$$

A clearer rowwise form is

$$
O_i
\leftarrow
\frac{
e^{m_i-m_i^{\text{new}}}\ell_iO_i
+
e^{\widetilde m_{ij}-m_i^{\text{new}}}
\widetilde P_{ij}V_j
}{
\ell_i^{\text{new}}
}.
$$

All vector multiplication and division in this expression is rowwise. The old $\(O_i\)$ is already normalized:

$$
O_i =
\frac{
\text{old weighted numerator}
}{
\ell_i
}.
$$

Therefore,

$$
\ell_iO_i
$$

recovers the old unnormalized weighted numerator.
If the maximum changes from $\(m_i\)$ to $\(m_i^{\text{new}}\)$, the old numerator must be rescaled by

$$
e^{m_i-m_i^{\text{new}}}.
$$

The current tile’s weighted numerator is

$$
\widetilde P_{ij}V_j.
$$

However, it was calculated relative to $\(\widetilde m_{ij}\)$, so it must be rescaled by

$$
e^{\widetilde m_{ij}-m_i^{\text{new}}}.
$$

After adding the two corrected numerators, divide by the new denominator

$$
\ell_i^{\text{new}}.
$$

The diagonal-matrix notation does not imply an expensive general matrix operation.

It is simply a mathematical way of expressing rowwise scaling and division.

14. Write the compact state back to HBM. Store

$$
O_i,
\qquad
\ell_i^{\text{new}},
\qquad
m_i^{\text{new}}
$$

back to HBM.

The two softmax-statistics vectors each contain only \(B_r\) elements.

This is tiny compared with a

$$
B_r\times B_c
$$

score or probability tile, and vastly smaller than an

$$
N\times N
$$

matrix.

15. Finish the loops and return $\(O\)$. Repeat the inner loop for every $\(Q\)$ block.

Repeat the outer loop for every $\(K,V\)$ block.

After every key/value block has been processed, $\(O\)$ contains

$$
O =
softmax
\left(
QK^\top
\right)V,
$$

with scaling, masking, and dropout included as required.

The complete score and probability matrices were never materialized in HBM.

The simplified algorithm omits several ordinary Transformer details. The complete FlashAttention forward kernel can combine:

1. Score calculation $\(QK^\top\)$
2. The $\(1/\sqrt d\)$ scaling factor
3. Padding or causal masking
4. Row-maximum calculation
5. Exponentiation
6. Row summation
7. Dropout
8. Multiplication by $\(V\)$
9. Output accumulation

<p align="center">
<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/9e22a655-bc59-4a08-95a2-fe4051d4aa57" />
</p>


This is more powerful than merely fusing a few elementwise operations. The attention algorithm itself is reorganized so that the large intermediate tensors never need to exist outside the on-chip tile.

For dropout, the original implementation saves the random-number-generator state instead of saving an $\(N\times N\)$ dropout mask. The same dropout mask can then be regenerated during backpropagation.

In a real model, tensors usually have a shape similar to

$$
[
\text{batch size},
\text{number of heads},
N,
d
].
$$

Each batch-head pair performs an independent attention computation. Extending FlashAttention-1 to batch_size > 1 and num_heads > 1 is conceptually straightforward. FlashAttention-1’s main coarse-grained parallelism is across

batch_size×num_heads

Its original execution design used approximately one CUDA thread block for one attention head, producing

batch_size×num_heads

independent thread blocks. A thread block is scheduled onto one streaming multiprocessor, or SM. The A100 has 108 SMs, so utilization is generally good when the number of independent batch-head work units is comparable to or greater than the available number of SMs. When that product is small, some SMs may remain idle. This limited sequence-dimension parallelism was one of the issues later addressed by FlashAttention-2. An SM may sometimes host multiple resident thread blocks, depending on register use, shared-memory use, and occupancy constraints. The one-block-per-SM explanation is a scheduling-level mental model.

The backward pass requires information derived from S and P. A standard attention implementation therefore tends to retain these N×N intermediate matrices. This is particularly expensive for long sequences.

FlashAttention saves compact information such as:

- $\(Q, K, V\)$
- The output $\(O\)$
- Final row maxima $\(m\)$
- Final row denominators $\(\ell\)$
- The dropout random-number-generator state, when dropout is enabled

It does not save the complete $\(S\)$ and $\(P\)$ matrices.

During backpropagation, it reloads blocks of $\(Q\)$, $\(K\)$, and $\(V\)$, and recomputes

$$
S_{ij} = Q_iK_j^\top.
$$

It then reconstructs the corresponding probability block from the final normalization statistics:

$$
P_{ij} =
\frac{
e^{S_{ij}-m_i}
}{
\ell_i
}.
$$

Because the final maximum and denominator are already known, the backward pass does not need to repeat the forward pass’s incremental online-softmax merging process. This is why backward can be considered conceptually simpler because there is no softmax-rescaling recurrence. Its implementation is still more complicated because it must calculate several gradients and hold more working values in SRAM.


Ignoring dropout for clarity, let $\(dO\)$ be the incoming gradient of the attention output.

A simplified backward calculation includes the following operations.

1. Recompute the scores

$$
S = QK^\top.
$$

This is matrix multiplication 1.

2. Calculate the gradient with respect to values

$$
dV = P^\top dO.
$$

This is matrix multiplication 2.

3. Calculate the gradient with respect to probabilities

$$
dP = dOV^\top.
$$

This is matrix multiplication 3.

4. Calculate the softmax gradient

For each row,

$$
dS =
P \odot
\left(
dP -
rowsum
\left(
dP \odot P
\right)
\right).
$$

An equivalent efficient form uses the identity

$$
rowsum
\left(
dP \odot P
\right) =
rowsum
\left(
dO \odot O
\right).
$$

This avoids materializing another large temporary matrix.

5. Calculate the gradient with respect to queries

$$
dQ = dSK.
$$

This is matrix multiplication 4.

6. Calculate the gradient with respect to keys

$$
dK = dS^\top Q.
$$

This is matrix multiplication 5.

Therefore:

- Forward: two major matrix multiplications, \(QK^\top\) and \(PV\)
- Backward: approximately five matrix multiplications, including score recomputation

The backward kernel must also handle:

- Masks
- Dropout regeneration
- Row reductions
- Gradient accumulation
- Additional on-chip state

This is why backward is mathematically simpler in one narrow respect but more complicated to implement overall.

Recomputation adds FLOPs. However, modern GPUs are extremely efficient at matrix multiplication. Reading and writing a huge $N \times N$ tensor from HBM can take more time than recomputing a small tile in SRAM.

- 10–20× memory savings, depending on sequence length
- Attention memory that is linear in \(N\), rather than quadratic in \(N\)
- 2–4× faster backward propagation** due to reduced memory traffic

Recomputation can therefore improve both memory usage and execution speed.

FlashAttention-2 preserves the central idea of FlashAttention-1—compute exact attention in SRAM-sized tiles without materializing the full N×N attention matrix—but reorganizes the arithmetic and GPU work assignment to make the hardware substantially more efficient.  Its three main improvements are:

- Reduce expensive non-matrix-multiplication FLOPs.
- Add parallelism along the sequence-length dimension, so a single attention head can use multiple GPU thread blocks.
- Partition work among warps more efficiently, reducing communication through shared memory.

These changes make FlashAttention-2 roughly twice as fast as FlashAttention-1 in the paper’s benchmarks, reaching about 50–73% of the A100’s theoretical peak throughput, compared with roughly 25–40% for FlashAttention-1.

Modern GPUs contain specialized units, such as Tensor Cores, designed to execute matrix multiplications extremely quickly. FlashAttention-1 already reduced HBM traffic, but it still reached only about 25–40% of the theoretical device throughput. Profiling showed that the remaining inefficiency came largely from suboptimal work division among thread blocks and warps, together with unnecessary non-matmul operations and shared-memory traffic.

FlashAttention-2 still computes exact dense attention:

$$
O =
softmax
\left(
\frac{QK^\top}{\sqrt d}
+
\text{mask}
\right)V.
$$

It does not use a sparse, low-rank, linear-attention, or other approximate replacement. It still:

- Divides $\(Q\)$, $\(K\)$, and $\(V\)$ into tiles.
- Loads tiles from HBM into fast on-chip SRAM.
- Computes one small score tile at a time.
- Uses online softmax to combine successive key blocks.
- Avoids storing the full score matrix $\(S\)$.
- Avoids storing the full probability matrix $\(P\)$.
- Recomputes attention probabilities during backpropagation.
- Uses $\(O(N)\)$ additional memory rather than storing $\(O(N^2)\)$ intermediates.

Its arithmetic complexity remains

$$
O(N^2d).
$$

The improvement is not a reduction in the number of query-key pairs. It is a more efficient mapping of the same mathematical work onto GPU hardware.

The first major change is a modification to the online-softmax update. Suppose attention is processed in key/value blocks. After processing the first block, FlashAttention-1 has a normalized partial output \(O^{(1)}\). When it processes the second block, the running softmax maximum and denominator may change. Conceptually, the normalized update has the form

$$
O^{(2)} =
diag
\left(
\frac{\ell^{(1)}}{\ell^{(2)}}
\right)
e^{m^{(1)}-m^{(2)}}O^{(1)}
+
diag
\left(
\ell^{(2)}
\right)^{-1}
e^{S^{(2)}-m^{(2)}}V^{(2)}.
$$

Both terms are scaled by the new denominator $\(\ell^{(2)}\)$. This means FlashAttention-1 performs repeated:

- Divisions
- Rowwise scaling
- Exponentiation-based corrections
- Normalization of the partial output

These are non-matmul operations. FlashAttention-2 instead maintains an unnormalized accumulated numerator. Let

$$
\widetilde O^{(j)}
$$

denote the unnormalized output after processing key/value blocks $\(1,\ldots,j\)$. For the second block, it updates

$$
\widetilde O^{(2)} =
e^{m^{(1)}-m^{(2)}}
\widetilde O^{(1)}
+
e^{S^{(2)}-m^{(2)}}V^{(2)}.
$$

Only after all key/value blocks have been processed does it normalize:

$$
O =
diag
\left(
\ell^{(\text{last})}
\right)^{-1}
\widetilde O^{(\text{last})}.
$$

In plain language: Do not repeatedly divide the partial output by the current softmax denominator. Keep the numerator unnormalized, and divide only once at the end.

This removes many repeated non-matmul FLOPs while preserving the exact same result. Suppose there are two score blocks:

$$
S^{(1)}
\quad\text{and}\quad
S^{(2)}.
$$

For the first block, compute the row maximum:

$$
m^{(1)} =
rowmax
\left(
S^{(1)}
\right)
\in
\mathbb{R}^{B_r}.
$$

Compute the rowwise exponential sum:

$$
\ell^{(1)} =
rowsum
\left(
e^{S^{(1)}-m^{(1)}}
\right)
\in
\mathbb{R}^{B_r}.
$$

Compute the first unnormalized weighted value sum:

$$
\widetilde O^{(1)} =
e^{S^{(1)}-m^{(1)}}V^{(1)}
\in
\mathbb{R}^{B_r\times d}.
$$

Now process the second score block. The new global maximum is

$$
m^{(2)} =
\max
\left(
m^{(1)},
rowmax
\left(
S^{(2)}
\right)
\right).
$$

If there are only two blocks, this final maximum can also be denoted by

$$
m=m^{(2)}.
$$

The updated denominator is

$$
\ell^{(2)} =
e^{m^{(1)}-m^{(2)}}\ell^{(1)}
+
rowsum
\left(
e^{S^{(2)}-m^{(2)}}
\right).
$$

Equivalently,

$$
\ell^{(2)} =
rowsum
\left(
e^{S^{(1)}-m}
\right)
+
rowsum
\left(
e^{S^{(2)}-m}
\right).
$$

The updated unnormalized output is

$$
\widetilde O^{(2)} =
e^{m^{(1)}-m^{(2)}}
\widetilde O^{(1)}
+
e^{S^{(2)}-m^{(2)}}V^{(2)}.
$$

Because

$$
\widetilde O^{(1)} =
e^{S^{(1)}-m^{(1)}}V^{(1)},
$$

the first term becomes

$$
e^{m^{(1)}-m}
\widetilde O^{(1)} =
e^{S^{(1)}-m}V^{(1)}.
$$

Therefore,

$$
\widetilde O^{(2)} =
e^{S^{(1)}-m}V^{(1)}
+
e^{S^{(2)}-m}V^{(2)}.
$$

Finally,

$$
O^{(2)} =
diag
\left(
\ell^{(2)}
\right)^{-1}
\widetilde O^{(2)}.
$$

This is exactly the same result as applying softmax to the concatenated score blocks and multiplying by the corresponding values.

FlashAttention-1 typically retained two rowwise statistics for backward:
- The final maximum $\(m\)$
- The final shifted exponential sum $\(\ell\)$

FlashAttention-2 combines them into the rowwise log-sum-exp value:

$$
L =
m+\log(\ell).
$$

For one row,

$$
L_i =
\log
\left(
\sum_j e^{S_{ij}}
\right).
$$

To see why, recall that

$$
\ell_i =
\sum_j e^{S_{ij}-m_i}.
$$

Then

$$
m_i+\log(\ell_i) =
m_i
+
\log
\left(
\sum_j e^{S_{ij}-m_i}
\right).
$$

Pulling \(e^{-m_i}\) out of the sum gives

$$
m_i+\log(\ell_i) =
\log
\left(
\sum_j e^{S_{ij}}
\right).
$$

This single vector is sufficient to reconstruct probabilities during backpropagation:

$$
P_{ij} =
e^{S_{ij}-L_i}.
$$

Thus FlashAttention-2 stores

$$
L\in\mathbb{R}^{N}
$$

instead of separately storing both

$$
m\in\mathbb{R}^{N}
\quad\text{and}\quad
\ell\in\mathbb{R}^{N}.
$$

This reduces persistent state and simplifies the backward formula.

Let

$$
Q,K,V\in\mathbb{R}^{N\times d}.
$$

Let:

- $\(B_r\)$: number of query rows in a query tile
- $\(B_c\)$: number of key/value rows in a key/value tile
- $\(T_r=\lceil N/B_r\rceil\)$: number of query tiles
- $\(T_c=\lceil N/B_c\rceil\)$: number of key/value tiles

The complete FlashAttention-2 forward pass is:

1. Partition the input matrices. Divide $\(Q\)$ into row blocks:

$$
Q_1,\ldots,Q_{T_r},
\qquad
Q_i\in\mathbb{R}^{B_r\times d}.
$$

Divide $\(K\)$ and $\(V\)$ into blocks:

$$
K_1,\ldots,K_{T_c},
\qquad
V_1,\ldots,V_{T_c},
$$

where

$$
K_j,V_j\in\mathbb{R}^{B_c\times d}.
$$

Divide the output into

$$
O_1,\ldots,O_{T_r},
\qquad
O_i\in\mathbb{R}^{B_r\times d}.
$$

Divide the log-sum-exp vector into

$$
L_1,\ldots,L_{T_r},
\qquad
L_i\in\mathbb{R}^{B_r}.
$$

2. Make the query-block loop outermost. For each query block

$$
i=1,\ldots,T_r,
$$

load

$$
Q_i
$$

from HBM into SRAM. This loop ordering differs from the simplified FlashAttention-1 presentation in which a $\(K,V\)$ block was often loaded and reused across many $\(Q\)$ blocks. FlashAttention-2 keeps one $\(Q_i\)$ resident while it walks through all key/value blocks. This makes each query-row block an independent unit of forward work, which later enables sequence-length parallelism across thread blocks.

3. Initialize the on-chip state. For the current query block, initialize

$$
O_i^{(0)} =
0
\in
\mathbb{R}^{B_r\times d},
$$

$$
\ell_i^{(0)} =
0
\in
\mathbb{R}^{B_r},
$$

$$
m_i^{(0)} =
-\infty
\in
\mathbb{R}^{B_r}.
$$

Here \(O_i^{(0)}\) is an unnormalized numerator accumulator, despite the pseudocode naming it $\(O\)$. 

4. Loop over key/value blocks. For

$$
j=1,\ldots,T_c,
$$

load

$$
K_j,V_j
$$

from HBM into SRAM.

Compute the current score tile:

$$
S_i^{(j)} =
Q_iK_j^\top
\in
\mathbb{R}^{B_r\times B_c}.
$$

In a complete Transformer implementation, the score also includes scaling and possibly a mask:

$$
S_i^{(j)} =
\frac{Q_iK_j^\top}{\sqrt d}
+
M_i^{(j)}.
$$

5. Update the running maximum. Compute

$$
m_i^{(j)} =
\max
\left(
m_i^{(j-1)},
rowmax
\left(
S_i^{(j)}
\right)
\right).
$$

This is an elementwise maximum over the $\(B_r\)$ rows.

6. Compute the local shifted exponentials. Compute

$$
\widetilde P_i^{(j)} =
\exp
\left(
S_i^{(j)}-m_i^{(j)}
\right)
\in
\mathbb{R}^{B_r\times B_c}.
$$

These are not yet final normalized probabilities. They are unnormalized exponential weights expressed relative to the latest running row maximum.

7. Update the denominator. The old denominator was expressed relative to \(m_i^{(j-1)}\). It must be converted to the scale of \(m_i^{(j)}\):

$$
\ell_i^{(j)} =
e^{m_i^{(j-1)}-m_i^{(j)}}
\ell_i^{(j-1)}
+
rowsum
\left(
\widetilde P_i^{(j)}
\right).
$$

8. Update the unnormalized output. Update

$$
O_i^{(j)} =
diag
\left(
e^{m_i^{(j-1)}-m_i^{(j)}}
\right)
O_i^{(j-1)}
+
\widetilde P_i^{(j)}V_j.
$$

A clearer rowwise form is

$$
O_i^{(j)} =
e^{m_i^{(j-1)}-m_i^{(j)}}
\odot
O_i^{(j-1)}
+
\widetilde P_i^{(j)}V_j.
$$

The exponential vector is broadcast across the $\(d\)$ output dimensions. Notice what is missing: there is no division by $\(\ell_i^{(j)}\)$ inside the loop.

9. Normalize only after the loop. After all $\(T_c\)$ key/value blocks have been processed, compute

$$
O_i =
diag
\left(
\ell_i^{(T_c)}
\right)^{-1}
O_i^{(T_c)}.
$$

Equivalently, rowwise:

$$
O_i =
\frac{
O_i^{(T_c)}
}{
\ell_i^{(T_c)}
}.
$$

This is the only final normalization required for the query block.

10. Compute log-sum-exp. Compute

$$
L_i =
m_i^{(T_c)}
+
\log
\left(
\ell_i^{(T_c)}
\right).
$$

Then write

$$
O_i
\quad\text{and}\quad
L_i
$$

to HBM.

After all query blocks are complete, the algorithm returns the exact output

$$
O =
softmax
\left(
QK^\top
\right)V,
$$

using

$$
O(N^2d)
$$

FLOPs and only

$$
O(N)
$$

extra storage beyond the inputs and outputs.

In autoregressive attention, token $\(i\)$ must not attend to a future token $\(j>i\)$. Therefore,

$$
S_{ij} =
-\infty
\qquad
\text{when}
\qquad
j>i.
$$

A naive implementation could calculate every score and then apply an elementwise causal mask. FlashAttention-2 can do better because attention is already divided into tiles. If every key-column index in a tile is greater than every query-row index in that tile, the entire tile is masked.

The kernel can skip the tile completely:

- Do not calculate $\(Q_iK_j^\top\)$.
- Do not perform softmax work.
- Do not load unnecessary values.
- Do not execute elementwise masking.

This is much cheaper than computing the tile and filling it with $\(-\infty\)$. With square tiles, most tiles are either:

- Completely valid
- Completely invalid and skipped

Only tiles crossing the causal diagonal contain a mixture of allowed and disallowed elements. Therefore, each query-row tile generally needs detailed causal masking in only one diagonal key tile. Optimized causal attention can be approximately 1.7–1.8 times faster than unmasked attention in relevant comparisons because a large portion of the upper-triangular work is skipped. The exact ratio depends on shape and implementation. Conceptually, causal attention performs roughly half as many score computations for long square sequences.

The FlashAttention-2 backward pass receives:

$$
Q,K,V,O,dO
\in
\mathbb{R}^{N\times d}
$$

and

$$
L\in\mathbb{R}^{N}.
$$

Its goal is to produce:

$$
dQ,
\quad
dK,
\quad
dV.
$$

The main difference from FlashAttention-1 is that probabilities are reconstructed using only the log-sum-exp vector $\(L\)$:

$$
P_{ij} =
\exp
\left(
S_{ij}-L_i
\right).
$$

There is no need to separately load a row maximum and exponential sum.

Let

$$
P =
softmax(S),
\qquad
O =
PV.
$$

Given \(dO\), first compute

$$
dP =
dOV^\top.
$$

The rowwise softmax derivative is

$$
dS =
P
\odot
\left(
dP -
rowsum
\left(
dP\odot P
\right)
\right).
$$

FlashAttention uses the identity

$$
rowsum
\left(
dP\odot P
\right) =
rowsum
\left(
dO\odot O
\right).
$$

Define

$$
D =
rowsum
\left(
dO\odot O
\right)
\in
\mathbb{R}^{N}.
$$

Then the score gradient can be written as

$$
dS =
P
\odot
\left(
dP-D
\right),
$$

where $\(D\)$ is broadcast across each row.

The pseudocode screenshot appears to label $\(D\)$ as belonging to $\(\mathbb{R}^{d}\)$, but the rowwise quantity has one value per query row, so the intended shape is

$$
D\in\mathbb{R}^{N}.
$$

Below is the step-by-step backward algorithm:

1. Partition all tensors.

Partition $\(Q\)$, $\(O\)$, $\(dO\)$, $\(dQ\)$, $\(L\)$, and $\(D\)$ according to query-row tiles.

Partition $\(K\)$, $\(V\)$, $\(dK\)$, and $\(dV\)$ according to key/value-row tiles.

2. Initialize $\(dQ\)$. Initialize

$$
dQ =
0
\in
\mathbb{R}^{N\times d}
$$

in HBM.

The gradient blocks $\(dK_j\)$ and $\(dV_j\)$ will be accumulated while processing one key/value block.

3. Precompute $\(D\)$. Compute

$$
D =
rowsum
\left(
dO\odot O
\right).
$$

Write $\(D\)$ to HBM and divide it into blocks

$$
D_1,\ldots,D_{T_r},
\qquad
D_i\in\mathbb{R}^{B_r}.
$$

4. Outer loop over key/value blocks. For each

$$
j=1,\ldots,T_c,
$$

load

$$
K_j,V_j
$$

from HBM into SRAM.

Initialize

$$
dK_j=0,
\qquad
dV_j=0
$$

in SRAM.

5. Inner loop over query blocks. For each query block $\(i\)$, load

$$
Q_i,
\quad
O_i,
\quad
dO_i,
\quad
dQ_i,
\quad
L_i,
\quad
D_i
$$

from HBM into SRAM.

6. Recompute the score tile

Compute

$$
S_i^{(j)} =
Q_iK_j^\top
\in
\mathbb{R}^{B_r\times B_c}.
$$

This recomputation avoids storing the complete $\(N\times N\)$ score matrix during the forward pass.

7. Reconstruct the probability tile. Using the saved log-sum-exp vector:

$$
P_i^{(j)} =
\exp
\left(
S_i^{(j)}-L_i
\right)
\in
\mathbb{R}^{B_r\times B_c}.
$$

The entries are already correctly normalized because

$$
L_i =
\log
\left(
\sum_k e^{S_{ik}}
\right).
$$

8. Accumulate the value gradient. Since

$$
O=PV,
$$

the value gradient is

$$
dV =
P^\top dO.
$$

For the current tiles:

$$
dV_j
\leftarrow
dV_j
+
\left(
P_i^{(j)}
\right)^\top
dO_i.
$$

Its shape is

$$
(B_c\times B_r)(B_r\times d) =
B_c\times d.
$$

9. Compute the probability gradient

Compute

$$
dP_i^{(j)} =
dO_iV_j^\top
\in
\mathbb{R}^{B_r\times B_c}.
$$

10. Compute the score gradient. Compute

$$
dS_i^{(j)} =
P_i^{(j)}
\odot
\left(
dP_i^{(j)}-D_i
\right).
$$

Here $\(D_i\)$ is broadcast across the $\(B_c\)$ columns of each row.

If attention scores include the usual scaling

$$
S =
\frac{QK^\top}{\sqrt d},
$$

then the corresponding $\(1/\sqrt d\)$ factor must also be included when propagating into $\(Q\)$ and $\(K\)$.

11. Accumulate the query gradient. The query gradient is

$$
dQ =
dSK.
$$

For the current tiles:

$$
dQ_i
\leftarrow
dQ_i
+
dS_i^{(j)}K_j.
$$

Because multiple $\(j\)-blocks$ contribute to the same $\(dQ_i\)$, the algorithm must repeatedly:

- Load $\(dQ_i\)$ from HBM.
- Update it in SRAM.
- Write it back to HBM.

This shared accumulation becomes important when backward work is parallelized across key-column blocks.

12. Accumulate the key gradient. The key gradient is

$$
dK =
dS^\top Q.
$$

For the current tiles:

$$
dK_j
\leftarrow
dK_j
+
\left(
dS_i^{(j)}
\right)^\top
Q_i.
$$

13. Write $\(dK_j,dV_j\)$ to HBM

After all query blocks have contributed to the current key/value block, write

$$
dK_j
\quad\text{and}\quad
dV_j
$$

back to HBM.

Finally return

$$
dQ,
\quad
dK,
\quad
dV.
$$

FlashAttention-2 explicitly supports:

- MQA: Multi-Query Attention
- GQA: Grouped-Query Attention

In ordinary multi-head attention, every query head generally has its own key and value head. In MQA, many query heads share one key head and one value head. In GQA, query heads are divided into groups, and all query heads in one group share a key/value head. This reduces the KV cache required during autoregressive inference. A naive implementation might physically duplicate the shared $\(K\)$ and $\(V\)$ tensors so that every query head appears to have a matching key/value head.

FlashAttention-2 does not need to do this. Instead, it uses indexing logic to map several query heads to the same key/value head:

$$
\text{KV head index} =
f
\left(
\text{query head index}
\right).
$$

The shared $\(K,V\)$ data is reused without explicit tensor replication. During backpropagation, however, multiple query heads contribute gradients to the same shared $\(K\)$ and $\(V\)$. Therefore, the corresponding gradients must be summed:

$$
dK_{\text{shared}} =
\sum_{h\in\text{group}}dK_h,
$$

$$
dV_{\text{shared}} =
\sum_{h\in\text{group}}dV_h.
$$

FlashAttention-2 added support for MQA and GQA, together with support for head dimensions up to 256. FlashAttention-1 primarily parallelized over:

$$
\text{batch size}
\times
\text{number of attention heads}.
$$

That gives approximately

$$
B\times H
$$

independent thread blocks.

A thread block is scheduled onto an SM, or streaming multiprocessor.

The A100 has 108 SMs. When

$$
B\times H
$$

is reasonably large—one slide uses roughly 80 or more as a practical example—the GPU can use most of its SMs efficiently.

However, long-sequence workloads often use:

- Small batches
- Fewer attention heads
- Large sequence length

For example, if

$$
B=1,
\qquad
H=16,
$$

there may be only 16 coarse-grained attention work units for 108 SMs. Much of the GPU can remain idle even though the attention matrix itself is enormous.

FlashAttention-2 parallelizes not only across batch items and heads but also across blocks of the sequence dimension. In the forward pass, different query-row blocks are independent. For a fixed attention head:

- Worker 1 can process query-row block 1.
- Worker 2 can process query-row block 2.
- Worker 3 can process query-row block 3.
- And so on.

Each worker owns one row block of the conceptual attention matrix. It scans through the required $\(K,V\)$ blocks and produces the corresponding rows of $\(O\)$. These thread blocks do not need to communicate with one another because different output rows are independent. The number of forward work units becomes approximately

$$
B\times H\times T_r,
$$

instead of merely

$$
B\times H.
$$

This raises occupancy when batch size and head count are small but the sequence is long.

In backward, the slides illustrate assigning workers by column blocks of the attention matrix. For a fixed key/value block $\(j\)$, one worker can accumulate:

$$
dK_j
\quad\text{and}\quad
dV_j
$$

across all query blocks.

Different $\(j\)-workers$ own different $\(dK_j,dV_j\)$ outputs, so these quantities do not conflict. The complication is $\(dQ\)$. Every key block contributes to the same query gradient:

$$
dQ_i =
\sum_j
dS_i^{(j)}K_j.
$$

Therefore, different workers may attempt to update the same $\(dQ_i\)$. The slides identify this as the main shared computation among column-parallel workers:

1. Load $\(dQ_i\)$ from HBM.
2. Add the worker’s contribution.
3. Write $\(dQ_i\)$ back to HBM.

When several thread blocks update the same $\(dQ_i\)$, atomic additions or another reduction mechanism are required to combine their contributions correctly. Despite this coordination, adding sequence-dimension parallelism substantially improves occupancy in long-sequence, low-batch, low-head-count settings.

A CUDA thread block contains several warps. A warp is normally a group of 32 threads that execute instructions together. Even after choosing the right number of thread blocks (e.g. 4 or 8), the work inside each block must be divided efficiently.

In FlashAttention-1’s forward pass:

- $\(Q\)$ was visible to all warps.
- $\(K\)$ was split among the warps.
- $\(V\)$ was split among the warps.

With four warps:

- Warp 1 handled one slice of $\(K,V\)$.
- Warp 2 handled another slice.
- Warp 3 handled another slice.
- Warp 4 handled another slice.

Each warp computed a partial contribution to

$$
QK^\top.
$$

After multiplication by its corresponding $\(V\)$ slice, the partial output contributions had to be summed across warps. This is called a $split-\(K\)$ decomposition because the reduction or inner dimension of the matrix multiplication is divided among workers.

Why $split-\(K\)$ is costly here? Each warp produces only a partial result. To combine those partial results, the warps must:

1. Write intermediate values to shared memory.
2. Synchronize.
3. Read the other partial values.
4. Add them together.

These shared-memory reads and writes are not free. They add communication and synchronization around the extremely fast matrix multiplications. The result is that Tensor Cores may spend time waiting for data exchange rather than continuously performing matmuls. 

FlashAttention-2 reverses the partitioning:

- $\(Q\)$ is divided among the warps.
- $\(K\)$ is visible to all warps.
- $\(V\)$ is visible to all warps.

For example, with four warps:

- Warp 1 owns one group of query rows.
- Warp 2 owns another group.
- Warp 3 owns another group.
- Warp 4 owns another group.

Every warp independently computes the output rows corresponding to its own $\(Q\)$ slice. After calculating its part of

$$
QK^\top,
$$

the warp multiplies by the shared $\(V\)$ tile and produces the final contribution for its own output rows.

Why this avoids communication? Different query rows correspond to different output rows. Therefore:

- Warp 1 writes one output-row region.
- Warp 2 writes another.
- Warp 3 writes another.
- Warp 4 writes another.

Their results do not have to be summed together. Consequently, warps do not need to exchange partial output accumulators through shared memory. This reduces:

- Shared-memory writes
- Shared-memory reads
- Synchronization
- Inter-warp communication

This improved warp partitioning is one of FlashAttention-2’s central performance gains.

FlashAttention-2 assigns different portions of $\(Q\)$ to different warps while sharing $\(K\)$ and $\(V\)$. This works naturally because output rows are independent. The backward pass is more complicated because it involves:

$$
Q,
\quad
K,
\quad
V,
\quad
O,
\quad
dO,
\quad
dQ,
\quad
dK,
\quad
dV
$$

and several dependencies between them. Even so, FlashAttention-2 avoids the earlier $split-\(K\)$ arrangement where possible. This reduces shared-memory traffic and improves performance again.

Some synchronization remains necessary because gradients can be shared across computation partitions, particularly when several workers contribute to the same $\(dQ\)$, $\(dK\)$, or $\(dV\)$. The important distinction is: FlashAttention-2 does not eliminate every synchronization. It eliminates an unnecessarily communication-heavy work partition used by FlashAttention-1.

Larger tiles often reduce the number of shared-memory loads and stores because more computation is performed per tile. However, making a tile larger also increases:

- Register usage
- Shared-memory usage
- The number of live intermediate values
- Pressure on occupancy

If register demand becomes too high, values can spill from registers into slower local memory. Register spilling can cause a substantial performance loss. If shared-memory demand exceeds the capacity available to a thread block, the kernel cannot be launched with that configuration. The best choice depends on:

- Head dimension $\(d\)$
- GPU architecture
- Available shared memory
- Register-file capacity
- Forward versus backward
- Causal versus noncausal attention
- Desired occupancy

Therefore, “use the largest possible tile” is not a correct universal strategy. The real goal is: Choose a tile large enough to reuse data efficiently, but small enough to avoid excessive register pressure, shared-memory consumption, or reduced occupancy.

FlashAttention-3 keeps the same mathematical attention operation as FlashAttention-1 and FlashAttention-2, but redesigns the GPU kernel specifically around NVIDIA’s Hopper architecture, especially the H100. Its goal is no longer only to reduce HBM traffic. FlashAttention-3 also tries to keep Hopper’s different hardware units busy at the same time:

- One group of warps loads data.
- Another group performs Tensor Core matrix multiplications.
- Softmax runs concurrently whenever dependencies allow.
- FP8 is used for much faster matrix multiplication, with extra techniques to control its numerical error.

The paper summarizes its three principal ideas as:

1. Warp specialization to overlap data movement and computation.
2. Asynchronous pipelining to overlap matrix multiplication and softmax.
3. FP8 block quantization and incoherent processing for higher throughput with lower numerical error.

On H100, the paper reports approximately 1.5–2.0× speedup over FlashAttention-2, up to about 740 TFLOPs/s in FP16 and close to 1.2 PFLOPs/s in FP8. It also reports that the improved FP8 method has about 2.6× lower numerical error than a straightforward FP8 baseline. FlashAttention-3 still calculates exact dense attention:

$$
O =
softmax
\left(
\frac{QK^\top}{\sqrt d}
+
M
\right)V.
$$

Here:

- $\(Q\)$ is the query matrix.
- $\(K\)$ is the key matrix.
- $\(V\)$ is the value matrix.
- $\(M\)$ may contain a causal or padding mask.
- $\(d\)$ is the head dimension.

It still uses the fundamental FlashAttention strategy:

- Divide $\(Q\)$, $\(K\)$, and $\(V\)$ into tiles.
- Keep only small tiles in on-chip memory.
- Avoid storing the full $\(N\times N\)$ score matrix.
- Use online softmax.
- Accumulate the output tile by tile.
- Save only compact rowwise normalization statistics.

The mathematical complexity remains approximately

$$
O(N^2d).
$$

The improvement comes from scheduling and hardware utilization, not from skipping arbitrary query-key pairs or approximating attention. Like FlashAttention-2, FlashAttention-3 has high-level parallelism across:

- Batch items
- Attention heads
- Query-sequence tiles

Suppose:

- $\(N\)$ is the sequence length.
- $\(d\)$ is the head dimension.
- $\(B_r\)$ is the number of query rows in one tile.

Then $\(Q\)$ is divided into

$$
T_r =
\left\lceil
\frac{N}{B_r}
\right\rceil
$$

tiles:

$$
Q_1,Q_2,\ldots,Q_{T_r},
\qquad
Q_i\in\mathbb{R}^{B_r\times d}.
$$

One CUDA thread block, also called a CTA or Cooperative Thread Array, processes one query tile $\(Q_i\)$ and produces the corresponding output tile

$$
O_i\in\mathbb{R}^{B_r\times d}.
$$

Therefore, the algorithm can be understood at two levels.

Different CTAs process different combinations of

$$
\text{batch item}
\times
\text{attention head}
\times
\text{query tile}.
$$

The CTA walks through all required $\(K,V\)$ tiles and calculates the output for one $\(Q_i\)$. FlashAttention-3’s major innovations mainly concern this second level: how the warps inside one CTA cooperate.

Hopper provides hardware features that were not fully exploited by the FlashAttention-2 kernel structure. The most important are:

- TMA, or Tensor Memory Accelerator
- WGMMA, or warp-group matrix multiply-accumulate
- Asynchronous execution and synchronization
- Dynamic register allocation among warp groups
- High-throughput FP8 Tensor Core operations

FlashAttention-3 combines these features so that memory transfers, matrix multiplication, and softmax can overlap instead of running almost entirely one after another.

TMA is a Hopper hardware mechanism that moves multidimensional tensor tiles between global memory and shared memory. Instead of having many normal CUDA threads manually calculate addresses and issue many load instructions, a small number of threads can describe a tensor transfer and let TMA carry it out asynchronously.

In FlashAttention-3, TMA loads

$$
Q_i
$$

and successive pairs

$$
K_j,\;V_j
$$

from HBM into shared memory. The important property is that issuing a TMA load does not necessarily block subsequent independent work. A producer can request a transfer and continue preparing later transfers while the hardware completes the earlier one. FlashAttention-3 uses a shared-memory pipeline with $\(s\)$ stages. Conceptually, shared memory contains slots such as

$$
0,1,\ldots,s-1.
$$

Tile $\(j\)$ is placed in stage

$$
j\bmod s.
$$

Before reusing a stage, the producer waits until the consumer has finished using the old data in that stage.

The sequence is:

1. Wait until buffer stage $\(j\bmod s\)$ becomes free.
2. Start loading $\(K_j,V_j\)$ into that stage.
3. Signal that the data is ready.
4. Let the consumer use it.
5. Release the stage after consumption.
6. Reuse it for a later tile.

Rather than making every warp perform every kind of work, FlashAttention-3 assigns different roles to different warp groups. A simplified division is:

1. Producer warp group. Responsible mainly for:

- Issuing TMA loads
- Moving $\(Q\)$, $\(K\)$, and $\(V\)$ from HBM to shared memory
- Managing pipeline stages
- Signaling when data is ready

2. Consumer warp group or groups. Responsible mainly for:

- Computing $\(QK^\top\)$
- Applying online softmax
- Computing $\(PV\)$
- Accumulating the output

The producer and consumers communicate using asynchronous barriers.

This is analogous to a factory:

- The producer delivers raw material.
- Consumers perform the calculations.
- The next delivery happens while the current material is being processed.

Loading data with TMA does not require the producer to hold a large number of intermediate values in registers. Matrix multiplication and softmax, however, need many registers. Hopper provides mechanisms that allow a warp group to relinquish some registers while another warp group receives more. The slides refer to this using operations such as `setmaxnreg`.

Conceptually:

1. The producer gives up registers that it does not need.
2. Consumer warps receive a larger register allocation.
3. Consumers use those registers for score tiles, accumulators, and softmax statistics.

This is important because registers are a limited on-chip resource shared by all warps in the CTA. Warp specialization therefore concerns not only instruction roles but also **resource specialization**. This is a circular producer-consumer pipeline.

GEMM means **General Matrix Multiplication**. A common form is

$$
C =
\alpha AB+\beta C.
$$

Here:

- $\(A,B,C\)$ are matrices.
- $\(\alpha,\beta\)$ are scalar coefficients.
- They are often $\(0\)$ or $\(1\)$.

Attention contains two important GEMMs:

$$
S=QK^\top
$$

and

$$
O=PV.
$$

For convenience, the slides call them:

- GEMM0: $\(QK^\top\)$
- GEMM1: $\(PV\)$

WGMMA is Hopper’s warp-group matrix multiply-accumulate instruction family. WGMMA is not merely a single-warp operation. A normal CUDA warp contains 32 threads. A Hopper warp group normally consists of four cooperating warps, or 128 threads. WGMMA lets this warp group execute matrix-multiply-accumulate work using Hopper Tensor Cores.

Depending on the instruction form, matrix operands can come from:

- Shared memory
- Registers

Two shorthand labels:

- SS-GEMM: both relevant matrix operands are supplied from shared memory.
- RS-GEMM: one operand is held in registers and another comes from shared memory.

For attention, the score operation may resemble an SS form:

$$
S_i^{(j)} =
Q_iK_j^\top,
$$

while the output update may use a register/shared-memory arrangement:

$$
O_i
\leftarrow
O_i+\widetilde P_i^{(j)}V_j.
$$

Let

$$
Q_i\in\mathbb{R}^{B_r\times d},
$$

$$
K,V\in\mathbb{R}^{N\times d},
$$

and let the key/value block size be \(B_c\). Then the number of key/value blocks is

$$
T_c =
\left\lceil
\frac{N}{B_c}
\right\rceil.
$$

The producer performs approximately the following process.

1. Set up the pipeline. Create synchronization objects and an $\(s\)-stage$ circular shared-memory buffer.

2. Release unneeded registers. The producer relinquishes a predetermined number of registers so that consumers can use them.

3. Load $\(Q_i\)$. Issue an asynchronous TMA transfer:

$$
Q_i:
\text{HBM}
\rightarrow
\text{shared memory}.
$$

When it completes, signal the consumer.

4. Stream $\(K,V\)$ blocks. For

$$
j=0,\ldots,T_c-1,
$$

the producer:

1. Waits until stage $\(j\bmod s\)$ is free.
2. Issues TMA loads for $\(K_j,V_j\)$.
3. Signals consumers when the loads are complete.

Because TMA is asynchronous, the producer does not need to wait for every previous transfer to finish before issuing unrelated later work, provided the pipeline has space. During the first $\(s\)$ iterations, the producer can often fill different buffer stages without waiting because those slots are initially empty. The consumer obtains more registers and initializes

$$
O_i=0,
$$

$$
\ell_i=0,
$$

$$
m_i=-\infty.
$$

For each $\(K_j,V_j\)$ tile:

1. Wait for the tile. Wait until the producer signals that $\(K_j\)$ is ready.

2. Compute the score tile

$$
S_i^{(j)} =
Q_iK_j^\top.
$$

3. Update online softmax statistics. Save the old maximum:

$$
m_i^{\text{old}} =
m_i.
$$

Update the running maximum:

$$
m_i =
\max
\left(
m_i^{\text{old}},
rowmax
\left(
S_i^{(j)}
\right)
\right).
$$

Calculate unnormalized local probabilities:

$$
\widetilde P_i^{(j)} =
\exp
\left(
S_i^{(j)}-m_i
\right).
$$

Update the denominator:

$$
\ell_i =
e^{m_i^{\text{old}}-m_i}\ell_i
+
rowsum
\left(
\widetilde P_i^{(j)}
\right).
$$

4. Wait for $\(V_j\)$. Wait until the corresponding value tile is ready.

5. Update the output accumulator. The old output must be rescaled if the running maximum changed:

$$
O_i
\leftarrow
diag
\left(
e^{m_i^{\text{old}}-m_i}
\right)O_i
+
\widetilde P_i^{(j)}V_j.
$$

This accumulator remains unnormalized.

6. Release the shared-memory stage. After $\(K_j,V_j\)$ are no longer needed, the consumer marks the stage as free so that the producer can reuse it.

After all $\(K,V\)$ tiles have been processed:

$$
O_i
\leftarrow
diag
\left(
\ell_i
\right)^{-1}
O_i.
$$

The log-sum-exp vector is

$$
L_i =
m_i+\log(\ell_i).
$$

The CTA writes

$$
O_i
\quad\text{and}\quad
L_i
$$

to HBM. This is still the same online-softmax strategy used in FlashAttention-2. The new contribution is that the loading and calculation paths are asynchronous and specialized.

Without overlap, one iteration might look like:

1. Load $\(K_j,V_j\)$.
2. Wait.
3. Calculate $\(QK_j^\top\)$.
4. Calculate softmax.
5. Calculate $\(\widetilde P_jV_j\)$.
6. Start loading the next tile.

Many hardware units are idle at different moments.

With producer-consumer overlap:

- The producer loads tile $\(j+1\)$.
- The consumer performs GEMMs and softmax for tile $\(j\)$.
- The data-transfer engine and Tensor Cores operate concurrently.

This hides much of the memory latency behind useful computation.

Warp specialization can also divide compute work between two consumer warp groups. This is **ping-pong scheduling**. Suppose there are:

- Consumer warp group 1
- Consumer warp group 2

The operations for one tile include:

- GEMM0: $\(QK^\top\)$
- Softmax
- GEMM1: $\(PV\)$

The problem is that these operations have different hardware characteristics. On an H100 SXM5:

- FP16 Tensor Core matrix multiplication has a theoretical peak around

$$
989\ \text{TFLOPs/s}.
$$

- Special functions such as exponentials have dramatically lower throughput, with the slides using approximately

$$
3.9\ \text{TFLOPs/s}
$$

as an illustrative comparison. For head dimension $\(128\)$, the slides state that the matmul FLOP throughput is about $\(512\)$ times the exponential throughput, while exponential hardware throughput is about $\(256\)$ times lower. This imbalance means softmax can occupy a surprisingly large fraction of runtime despite requiring far fewer nominal operations. The aim is therefore: Run softmax on one warp group while another warp group keeps Tensor Cores busy with GEMM.

A synchronization mechanism such as a barrier can prioritize work so that one warp group begins GEMM while another performs softmax. A simplified schedule is:

1. Warp group 1

- GEMM0
- Softmax
- GEMM1
- Yield the compute role

2. Warp group 2

- GEMM0
- Softmax
- GEMM1
- Yield the compute role

Their execution is offset in time. When warp group 1 is doing softmax, warp group 2 may issue a GEMM. Later they exchange roles. This resembles ping-pong:

- One group handles non-matmul work.
- The other feeds Tensor Cores.
- Then they switch.

For an FP16 forward pass with:

- Head dimension \(128\)
- Sequence length \(8192\)

the slides report an increase from approximately

$$
570\ \text{TFLOPs/s}
$$

to approximately

$$
620\text{–}640\ \text{TFLOPs/s}.
$$

This illustrates how overlapping softmax with Tensor Core operations improves utilization. For MQA and GQA, FlashAttention-3 follows the same broad indexing strategy as FlashAttention-2: shared \(K,V\) heads are accessed through adjusted indices rather than physically duplicating $\(K,V\)$ in HBM.

Producer-consumer overlap hides memory movement, but there are also dependencies inside the consumer loop. For tile $\(j\)$:

1. Softmax needs the score result

$$
S_j =
QK_j^\top.
$$

2. GEMM1 needs the probability tile

$$
\widetilde P_j.
$$

So the straightforward sequence is

$$
\text{GEMM0}_j
\rightarrow
\text{softmax}_j
\rightarrow
\text{GEMM1}_j.
$$

If every operation waits for the previous one to finish, the pipeline becomes mostly serial. FlashAttention-3 therefore constructs a multi-stage pipeline across different iterations.

The two-stage schedule overlaps:

- GEMM1 for tile $\(j\)$
- GEMM0 for tile $\(j+1\)$
- Softmax for tile $\(j+1\)$, as dependencies permit

A useful way to see it is

$$
\text{GEMM0}_{j+1}
\quad\parallel\quad
\text{GEMM1}_{j}
$$

followed by

$$
\text{softmax}_{j+1}.
$$

The actual algorithm maintains two score buffers:

- $\(S_{\text{cur}}\)$
- $\(S_{\text{next}}\)$

Load $\(Q_i\)$ and $\(K_0\)$, then calculate

$$
S_{\text{cur}} =
Q_iK_0^\top.
$$

Wait for it to finish and compute the corresponding:

- Running maximum
- Probability tile
- Softmax denominator
- Output rescaling

For each later tile $\(j\)$:

1. Start the next score GEMM asynchronously.

$$
S_{\text{next}} =
Q_iK_j^\top.
$$

Commit it without immediately waiting.

2. Perform the current value GEMM. Using the previous probability tile:

$$
O_i
\leftarrow
O_i+
\widetilde P_{\text{cur}}V_{j-1}.
$$

Commit this asynchronously as well.

3. Wait for the new score. Once \(S_{\text{next}}\) is ready, compute:

- New row maximum
- New exponentials
- New denominator

4. Finish and rescale the old output. Wait for the previous $\(PV\)$ operation, then apply the output rescaling required by the updated maximum.

5. Rotate buffers. Set

$$
S_{\text{cur}}
\leftarrow
S_{\text{next}}.
$$

The process repeats.

In the steady state, the second WGMMA of iteration $\(j\)$ can overlap with the softmax of iteration $\(j+1\)$.

WGMMA is asynchronous. The kernel can:

1. Issue the operation.
2. Commit the operation.
3. Continue with independent work.
4. Wait only when the result is actually needed.

If the kernel calls a wait immediately after every WGMMA, it destroys the benefit of asynchronous execution. The two-stage algorithm carefully postpones waits to expose overlap.

The pseudocode represents an ideal schedule. However, the compiler, such as NVCC, may reorder instructions while optimizing the program.

That can cause problems:

- A WGMMA may be scheduled later than intended.
- Softmax instructions may move.
- Operations designed to overlap may become serialized.
- Unexpected dependencies may appear.

Therefore, performance-oriented implementations must inspect the generated SASS machine code and carefully control dependencies, barriers, and instruction ordering. An algorithmically correct pipeline is not automatically a well-scheduled hardware pipeline.

The two-stage pipeline must preserve more intermediate state than a simple serial loop. In particular, it may need an extra score tile:

$$
S_{\text{next}}
\in
\mathbb{R}^{B_r\times B_c}.
$$

If this tile is held in FP32-sized storage, its space requirement is approximately

$$
B_rB_c
\cdot
sizeof
\left(
\text{float}
\right).
$$

The kernel must also retain:

- Current score or probability data
- Output accumulators
- Row maxima
- Denominators
- Scaling values
- WGMMA accumulators

This creates a trade-off.

For Larger tile:

Advantages:

- Higher arithmetic efficiency
- Better data reuse
- Fewer loop iterations

Disadvantages:

- More registers
- More shared memory
- Greater risk of spilling
- Lower occupancy

For Deeper pipeline:

Advantages:

- More overlap
- Higher potential Tensor Core utilization

Disadvantages:

- More live intermediate state
- Greater register demand
- More complicated scheduling

FlashAttention-3 also explores a three-stage variant. The intention is to overlap:

- GEMM0 for iteration $\(j+2\)$
- Softmax for iteration $\(j+1\)$
- GEMM1 for iteration $\(j\)$

Conceptually:

$$
\text{GEMM0}_{j+2}
\quad\parallel\quad
\text{softmax}_{j+1}
\quad\parallel\quad
\text{GEMM1}_{j}.
$$

This exposes even more theoretical concurrency and could increase Tensor Core utilization.

Why the three-stage version may be worse? The practical implementation is often inferior to the two-stage version. The compiler may not issue instructions in the intended order. The desired overlap might be:

$$
\text{softmax}
\quad
\text{with both}
\quad
\text{GEMM0 and GEMM1}.
$$

But generated machine code may overlap softmax only with the first WGMMA, while the second WGMMA remains serialized. The additional stage then produces complexity without the expected utilization gain.

Besides, a three-stage pipeline must hold additional objects, such as:

- Another $\(\widetilde P\)$ tile
- Additional output scaling data
- More WGMMA accumulators
- More pipeline context

Additional storage are similar to

$$
B_rB_c
\cdot
sizeof
\left(
\text{input type}
\right)
+
B_r
\cdot
sizeof
\left(
\text{float}
\right).
$$

To prevent register overflow, smaller tiles may be required. Smaller tiles reduce Tensor Core efficiency and data reuse, potentially erasing the gains from the deeper pipeline. The broader lesson is: More pipeline stages do not automatically mean better performance. Pipeline depth must be balanced against tile size, registers, shared memory, compiler behavior, and occupancy.

Hopper Tensor Cores have much higher throughput for FP8 than for FP16/BF16. The theoretical FP8 matrix-multiplication throughput is roughly twice the FP16 throughput on relevant Hopper configurations. However, FP8 creates two large challenges:

1. Memory-layout compatibility
2. Numerical accuracy

FlashAttention-3 addresses both. Compared with FP16 or BF16, FP8 has far fewer precision bits. This creates two problems. Nearby real values may be mapped to the same FP8 number because the mantissa is short. Large models often contain outlier values whose magnitudes are far larger than most values in a tensor. If one scale factor is chosen for the entire tensor, the outlier may force a large quantization range. Most normal values then use only a small fraction of the available FP8 representation levels. This can substantially increase attention error. The tensors $\(Q,K,V\)$ are typically stored with the head dimension contiguous. That means adjacent memory values usually correspond to adjacent elements of the feature dimension $\(d\)$. However, the FP8 WGMMA used for the second matrix multiplication can require the \(V\) tile to follow a different major ordering, with sequence-position elements contiguous in the required layout. In simplified terms:

- The original $\(V\)$ storage is convenient for the standard tensor layout.
- GEMM1 wants a transposed or differently strided $\(V\)$ representation.

TMA can efficiently copy tiles, but it does not arbitrarily change the contiguous dimension during the transfer. Therefore, FlashAttention-3 needs a layout-conversion strategy.

There are two possible $\(V\)-layout$ strategies.

1. pretranspose \(V\) in global memory. This can be done by: Fuse the transpose into an earlier operation (For example, an earlier positional-encoding or tensor-production kernel could write $\(V\)$ directly in the desired layout.); Launch a separate preprocessing kernel. The preprocessing kernel exchanges the sequence and head-dimension strides. Disadvantages are:

- Fusion may be hard to integrate into a general-purpose library.
- A separate preprocessing kernel adds work and extra memory traffic.
- In memory-limited inference, maintaining another representation can be wasteful.

2. transpose $\(V\)$ inside the attention kernel

FlashAttention-3 chooses this strategy for its FP8 implementation. The tile is loaded into shared memory and then rearranged on-chip before GEMM1. Hopper provides instructions such as:

- `ldmatrix`
- `stmatrix`

These allow threads in a warp to cooperatively move matrix fragments between shared memory and registers. The benefits are:

- Efficient register use
- Layout-aware matrix movement
- No additional HBM copy
- Ability to hide the transpose behind WGMMA execution

After the first iteration, the kernel can begin transposing the next $\(V\)$ tile while Tensor Cores are processing the current $\(K,V\)$ work. Thus, the on-chip transpose becomes part of the asynchronous pipeline rather than a separate visible stage.

FP8 WGMMA also has a register-layout issue. The FP32 accumulator layout produced by the first WGMMA is not necessarily arranged in the register order expected for the second WGMMA. That means the probability or score fragment held in registers must be reorganized. For example, a logical ordering may be changed to something like

$$
\{
d_0,d_1,d_4,d_5,d_2,d_3,d_6,d_7
\}.
$$

This pattern repeats every eight bytes. Conceptually, the transformation rearranges columns of the probability tile. The slides give the example that a column ordering such as

$$
0,1,8,9
$$

may become the first four logical columns required by the next operation. The on-chip $\(V\)$ transpose must use a matching row arrangement so that the second WGMMA still calculates the correct output. Thus, two layout transformations are coordinated:

1. Reorder the probability accumulator in registers.
2. Reorder the $\(V\)$ tile in shared memory or register fragments.

Together they present the operands in the layout required by FP8 GEMM1. A simple FP8 implementation may use one scale factor per tensor:

$$
Q_{\text{FP8}} =
quantize
\left(
Q;s_Q
\right),
$$

$$
K_{\text{FP8}} =
quantize
\left(
K;s_K
\right),
$$

$$
V_{\text{FP8}} =
quantize
\left(
V;s_V
\right).
$$

This is vulnerable to outliers. FlashAttention-3 instead uses block quantization. Divide the tensors into the same kinds of tiles already used by the attention algorithm:

$$
Q_i
\in
\mathbb{R}^{B_r\times d},
$$

$$
K_j,V_j
\in
\mathbb{R}^{B_c\times d}.
$$

Give each tile its own scale:

$$
s_{Q_i},
\qquad
s_{K_j},
\qquad
s_{V_j}.
$$

Then quantize each block independently. For example:

$$
\widehat Q_i =
quantize
\left(
\frac{Q_i}{s_{Q_i}}
\right).
$$

Because each tile has a narrower local range than the complete tensor, its FP8 values use the available numerical levels more effectively.

FlashAttention already works tile by tile. Therefore:

- Quantization can be fused with earlier tile-related work.
- Scale factors naturally align with kernel tiles.
- Each score tile can be corrected using its corresponding scales.
- Little or no additional asymptotic computation is introduced.

Quantization may be fused with memory-bandwidth-bound operations such as rotary positional encoding, so the extra arithmetic may be hidden under existing memory movement.

Block quantization reduces variation between different regions, but a single block can still contain a few large outliers. FlashAttention-3 therefore uses incoherent processing. Before quantization, transform $\(Q\)$ and $\(K\)$ using a structured orthogonal matrix $\(M\)$:

$$
Q'=QM,
$$

$$
K'=KM.
$$

Because $\(M\)$ is orthogonal,

$$
MM^\top =
I.
$$

Therefore:

$$
(QM)(KM)^\top =
QMM^\top K^\top =
QK^\top.
$$

So the exact full-precision attention score matrix is unchanged. Each element of $\(QM\)$ is a mixture of many elements from the original query vector. Similarly, each element of $\(KM\)$ is a mixture of many key features. A very large outlier concentrated in one original feature is spread over multiple transformed features. This lowers the peak magnitude and makes the transformed values more uniform. FP8 quantization then has an easier task because:

- The dynamic range is smaller.
- Outliers are less concentrated.
- More FP8 representational levels are useful for normal values.

The term incoherent refers to spreading energy across coordinates so that it is not concentrated in a small number of dimensions.

$\(M\)$ is a product of:

- A random $\(\pm1\)$ diagonal matrix
- A Hadamard matrix

A Hadamard transform can be computed in approximately

$$
O(d\log d)
$$

operations rather than dense

$$
O(d^2)
$$

matrix multiplication.

The transformation can also be fused with preceding operations such as rotary positional encoding. This means that the accuracy improvement need not introduce a large separate kernel cost.

The combination of block quantization and incoherent processing can reduce FP8 numerical error by up to approximately

$$
2.6\times.
$$

The two techniques solve different parts of the problem. Block quantization adapts the scale to local regions. Incoherent processing spreads large outliers across dimensions before quantization. Together they make FP8 attention substantially more accurate than simply converting the full tensors using one global scale. 

The backward pass also uses warp specialization and asynchronous execution. As in earlier FlashAttention versions, it recomputes attention score and probability tiles rather than saving complete $\(N\times N\)$ intermediates. A new challenge is accumulation into $\(dQ\)$. For one query block,

$$
dQ_i =
\sum_j
dS_{ij}K_j.
$$

Different CTAs or work partitions may independently calculate contributions to the same global $\(dQ_i\)$. This creates a memory race:

- Multiple thread blocks may try to update the same output location.
- A normal write would lose some contributions.
- Atomic accumulation or an explicit reduction mechanism is required.

FlashAttention-3 adds another specialized role:

- Producer warps load data.
- Consumer warps calculate gradients.
- A dedicated writer warp performs global $\(dQ\)$ accumulation.

The writer issues the necessary updates asynchronously. This is valuable because a global or atomic write can have significant latency. Without specialization, all compute warps might have to wait for the write to complete. With a dedicated writer:

1. Consumer warps produce a $\(dQ\)$ contribution.
2. The writer warp commits it to global $\(dQ\)$.
3. The other warps continue processing the next tile.
4. Synchronization is needed only when a buffer or dependency requires it.

This extends the producer-consumer idea from forward data loading to backward gradient output.

FlashAttention is an efficient and numerically stable implementation of exact self-attention. It improves performance mainly by reorganizing how attention is computed and how data moves through GPU memory. Its three main advantages are:

- Lower memory usage. FlashAttention divides the attention matrix into small tiles and processes one tile at a time. It does not store the complete S or P matrix in GPU global memory. Because it avoids quadratic intermediate storage, FlashAttention can process sequences containing thousands or even tens of thousands of tokens much more efficiently.
- Higher computational efficiency. FlashAttention does not substantially reduce the mathematical amount of dense-attention computation. It still evaluates the relevant query-key pairs, so its arithmetic complexity remains approximately O(L2d). Its speed advantage comes from reducing expensive memory traffic.
- Better numerical stability. FlashAttention uses the stable softmax identity. This greatly reduces the risk of numerical overflow.

#### 5.2.2.4 PagedAttention

PagedAttention is an attention and memory-management method designed for efficient LLM inference. Its main purpose is to manage the KV cache more flexibly and reduce memory waste when many requests with different prompt lengths and output lengths are being served at the same time. The core idea is very similar to virtual memory in an operating system: A sequence’s KV cache is divided into fixed-size logical blocks, and those logical blocks can be mapped to physically noncontiguous blocks in GPU memory. This allows vLLM to allocate KV-cache memory only when needed, reuse freed blocks, share blocks across related sequences, and support dynamic batching more efficiently.

When a user sends a prompt to an autoregressive LLM, the model generates output tokens one at a time. Suppose the prompt is

$$
x_1,\ldots,x_n
$$

and the model generates

$$
x_{n+1},\ldots,x_{n+T}.
$$

At every generation step, the prediction of the next token depends on:

- All prompt tokens
- All previously generated output tokens

For attention, the model needs the key and value vectors of all previous tokens. Recomputing those key and value vectors at every step would be wasteful, so they are cached. This cache is called the KV cache. At each decoding step:

1. The model receives the newest token.
2. It computes only that token’s new key and value vectors.
3. It appends them to the KV cache.
4. It uses all previous cached keys and values to predict the next token.

LLM inference can be divided into two main phases. The prompt phase processes the entire user prompt:

$$
x_1,\ldots,x_n.
$$

It computes the probability of the first generated token:

$$
P(x_{n+1}\mid x_1,\ldots,x_n).
$$

During this phase, the system must calculate the key and value vectors of every token in the prompt. This phase can be highly parallelized because the full prompt is already known. Matrix multiplication can make good use of GPU parallelism.

After the prompt phase, the model generates one token at a time. Suppose $\(t\)$ output tokens have already been generated. At step $\(t+1\)$, the model receives the newest token and uses:

- The KV cache of the prompt
- The KV cache of all previously generated output tokens

to predict the next token. Only the newest token’s key and value vectors need to be computed and appended to the cache. Generation continues until:

- A maximum output length is reached, or
- The model generates the end token `<eos>`

This phase depends on all previously generated content, so it cannot be fully parallelized. Most work proceeds token by token. As a result, autoregressive decoding often becomes a bottleneck, especially when:

- Sequences are long
- Many requests are active
- Different requests have very different lengths

LLM requests do not all arrive at the same time, and they do not all have the same length. Two important problems appear. Requests arrive asynchronously. If the server waits until a batch is full before processing it, early requests wait too long. If it does not wait, GPU resources may be underutilized. Batch efficiency may fall. Besides, different requests may have very different: 

- Prompt lengths
- Expected output lengths

If all requests are padded to the same maximum length, the system wastes:

- Memory
- Computation

This is particularly serious when one request is much longer than the others.

To reduce these problems, LLM serving systems can use more fine-grained batching methods, such as:

- Cellular batching
- Iteration-level scheduling

Instead of waiting for an entire request to finish, the batch is updated after each decoding iteration. After one token-generation iteration:

- Completed requests are removed.
- New requests are added.
- Active requests continue.

This makes the batch dynamic at token granularity. Benefits include:

- Less padding waste
- Shorter queueing time
- More flexible request mixing
- Better GPU utilization

Specialized GPU kernels can also process sequences of different lengths without excessive padding. However, even with fine-grained batching, the number of simultaneous requests is still constrained by GPU memory, especially by the KV cache.

The size of the KV cache grows rapidly with:

- Number of requests
- Sequence length
- Number of layers
- Hidden size
- Data precision

Since modern GPUs often have only tens of gigabytes of memory, only a limited number of requests may fit simultaneously if memory is used inefficiently. Also, GPU compute speed has increased faster than GPU memory capacity. For example, moving from A100 to H100 greatly increases FLOPs, while maximum memory may still remain around $\(80\)$ GB. Therefore, memory becomes an increasingly important bottleneck.

Traditional KV-cache allocation can waste a large amount of memory. A request may reserve space for future tokens that are never generated. For example, a request may reserve many slots based on an assumed maximum output length, even though the sequence ends early. If a request is allocated a large contiguous region and uses only part of it, unused space inside that reservation is wasted. Free memory may exist, but it may be split into small noncontiguous regions. A large new request may fail to find one sufficiently large contiguous block even though the total free memory is adequate. There are also additional memory-management difficulties:

1. Large KV cache. KV-cache size grows quickly with request count and sequence length. Poor memory management reduces batch size and overall throughput.

2. Different decoding algorithms have different sharing patterns. Memory-management complexity depends on the decoding algorithm. For example, In parallel sampling, multiple outputs share the same prompt KV cache. In beam search, beams may share not only the prompt but also later intermediate prefixes. The sharing structure may change dynamically during decoding. The memory system must support these patterns efficiently.

Prompt lengths vary widely, and output lengths are not known in advance. As decoding continues, a request’s KV cache keeps growing. It may consume memory that could otherwise be used for:

- New requests
- Other active prompts
- Larger batches

The system may need to:

- Evict KV-cache blocks
- Swap them to CPU memory
- Recompute them later

The authors introduced **PagedAttention** and built the vLLM serving engine around it. vLLM uses a centralized scheduler to coordinate distributed GPU workers. A KV-cache manager manages physical KV-cache memory on GPU workers according to commands from the centralized scheduler. The system contains ideas such as:

- Scheduler
- KV-cache manager
- CPU block allocator
- GPU block allocator
- Per-worker cache engine
- Model shards

The scheduler decides which requests are processed, while the KV-cache manager maintains the mapping between logical KV blocks and physical memory blocks.

PagedAttention is inspired by virtual memory in operating systems.

In an operating system:

- Memory is divided into fixed-size pages.
- A program sees logical pages.
- Logical pages are mapped to physical pages.
- Logically adjacent pages do not need to be physically adjacent.

PagedAttention applies the same idea to the KV cache. A sequence’s KV cache is divided into fixed-size KV blocks. Each block contains the key and value vectors for a fixed number of tokens. Let the block size be $\(B\)$. The $\(j\)-th$ key block is

$$
K_j =
\left(
k_{(j-1)B+1},
\ldots,
k_{jB}
\right),
$$

and the corresponding value block is

$$
V_j =
\left(
v_{(j-1)B+1},
\ldots,
v_{jB}
\right).
$$

The logical KV blocks of one sequence can be stored in physically noncontiguous locations. That is the key difference from ordinary contiguous KV-cache allocation. 

PagedAttention rewrites attention in block form. For query $\(q_i\)$, the attention score vector for KV block $\(j\)$ is

$$
A_{ij} =
\frac{
\exp
\left(
q_i^\top K_j/\sqrt d
\right)
}{
\sum_{t=1}^{\lceil i/B\rceil}
\exp
\left(
q_i^\top K_t/\sqrt d
\right)
}.
$$

The output is

$$
o_i =
\sum_{j=1}^{\lceil i/B\rceil}
V_jA_{ij}^{\top}.
$$

Here:

- $\(A_{ij}\)$ is the attention-score row vector for block $\(j\)$.
- It contains the attention weights for all token positions inside that block.
- The kernel identifies and retrieves the necessary physical KV blocks during attention.

The blocks do not need to be physically contiguous. The KV-cache manager separates:

- Logical KV blocks
- Physical KV blocks

A request sees its KV cache as a sequence of logical blocks. As tokens are generated, the logical blocks are filled from left to right. Only the unused positions in the final logical block are reserved for future tokens. On the GPU worker:

- The block engine reserves a large physical GPU-memory region.
- That region is divided into fixed-size physical KV blocks.
- A block table records how logical blocks map to physical blocks.

Each block-table entry records:

- Which physical block corresponds to a logical block
- How many token positions in the logical block are currently filled

Because logical and physical blocks are separated, the KV cache can grow without reserving memory for every future token in advance. This eliminates most of the waste caused by large contiguous reservations. Suppose two requests, A and B, have different logical sequences. Their logical KV blocks may be mapped to physical blocks such as:

- Request A logical block 0 → physical block 7
- Request A logical block 1 → physical block 1
- Request A logical block 2 → physical block 3
- Request B logical block 0 → physical block 5
- Request B logical block 1 → physical block 2

Adjacent logical blocks do not need to be adjacent in GPU memory. This means:

- Free physical blocks can be used immediately.
- The system does not need one large contiguous region per request.
- Blocks released by completed requests can be reused by others.
- External fragmentation is greatly reduced.

One sentence from the slides summarizes it: PagedAttention allows KV blocks to be stored in physically noncontiguous memory, enabling flexible paged memory management in vLLM.

At each decoding iteration, vLLM performs several steps.

1. Select candidate sequences. vLLM chooses a group of active sequences for the next batch. It allocates physical blocks for any new logical blocks they require.

2. Build the model input. vLLM concatenates the input tokens for all requests in the current iteration. These include:

- All prompt tokens during the prompt phase
- The most recent token during the generation phase

The combined tokens are sent to the LLM.

3. Read old KV cache and write new KV cache. During model execution:

- The block table is used to locate previous KV-cache blocks.
- Newly generated key and value vectors are written to physical KV blocks.

Because one KV block contains several token positions, the PagedAttention kernel can process multiple positions inside a block in parallel. This improves hardware utilization and reduces latency. However, very large blocks may increase internal fragmentation.

vLLM allocates a new physical block only when needed. Blocks are filled from left to right. A new block is allocated only after all previous blocks are full. Therefore, for each request, unused memory is limited to at most one partially filled block.

This gives two major benefits:

- Efficient use of nearly all available memory
- More simultaneous requests in a batch

When a request finishes, all of its physical blocks are released and can be reused.

The block size creates a trade-off. For larger blocks, Advantages are:

- More token positions can be processed together
- Better parallel processing inside the attention kernel
- Potentially better hardware utilization
- Lower metadata overhead

Disadvantages are:

- More internal fragmentation in the final block
- More wasted space if a sequence ends early

For smaller blocks, advantages are:

- Less internal fragmentation
- Finer-grained allocation

Disadvantages are:

- More block-table entries
- More metadata
- Potentially lower kernel efficiency

PagedAttention must balance these competing effects. In parallel sampling, one input prompt generates several output sequences. All output sequences share the same prompt. Therefore, their prompt KV cache is identical and can be shared. The prompt portion may account for roughly $\(12\%\)$ of the entire KV-cache memory in an example. Sharing it reduces memory use. PagedAttention supports sharing because multiple logical blocks can map to the same physical block. Each physical block has a reference count. 

Once two sequences diverge, they may need to write different tokens into a block that was previously shared. vLLM uses **copy-on-write** at block granularity. Suppose samples A1 and A2 share a physical block.

If A1 needs to modify the last logical block:

1. vLLM checks the block’s reference count.
2. If the reference count is greater than 1, the block is shared.
3. vLLM allocates a new physical block.
4. It copies the old block’s content into the new block.
5. A1 is remapped to the new block.
6. The original block’s reference count decreases.
7. A2 continues using the original block.

This resembles process forking in an operating system. The advantage is that shared prompt blocks remain shared, while only the block being modified must be copied. For long prompts, this can significantly reduce memory overhead.

Beam search keeps the top $\(k\)$ most promising candidate sequences at every decoding step. Unlike simple parallel sampling, beam candidates may share:

- The original prompt
- Several later blocks
- Different parts of their decoding histories

The sharing structure changes dynamically over time, like a tree of process forks. After ranking the next beam candidates:

- Some previous candidates are no longer selected.
- Their logical blocks are released.
- Physical blocks whose reference counts drop to zero are freed.
- New physical blocks are allocated for the surviving new candidates.

Traditional systems might frequently copy large parts of the KV cache between beam candidates. vLLM reduces this copying through physical block sharing and copy-on-write. Most blocks remain shared. Only the newly modified shared block is copied.

Many users may submit prompts that share a common prefix, such as:

- System instructions
- Task description
- Few-shot examples
- Standard input/output demonstrations

The full request prompt may consist of:

$$
\text{shared prefix}
+
\text{user-specific prompt}.
$$

The service can precompute and store the KV cache of the shared prefix. When a new request includes that prefix:

- Its logical blocks are mapped to the already cached physical blocks.
- Only the user-specific part needs prompt-phase computation.
- The last block can be marked copy-on-write if it may be modified.

This is similar to sharing a library across operating-system processes. It reduces repeated prompt computation and lowers memory use.

Different requests may use different decoding methods at the same time. Examples include:

- Greedy decoding
- Parallel sampling
- Beam search
- Shared-prefix workloads

These methods have different KV-cache sharing patterns. vLLM hides this complexity behind a common logical-to-physical block mapping layer. The model and attention kernel only need to know:

- The physical block IDs belonging to each sequence

They do not need to directly manage the sharing relationships among sequences. This allows more requests with different decoding strategies to be batched together, increasing overall throughput. When request volume exceeds system capacity, vLLM must choose which requests to process. The **first-come-first-served**, or FCFS, policy, is used to:

- Preserve fairness
- Prevent starvation

When preemption is necessary:

- Earlier requests receive higher priority.
- More recently arrived requests are preempted first.

This is particularly important because:

- Prompt lengths vary greatly.
- Output lengths are unknown in advance.
- Active KV caches keep growing.
- GPU memory can eventually run out of free physical blocks.

Instead of evicting individual blocks from a sequence, vLLM either:

- Evicts all blocks belonging to a sequence, or
- Keeps all of them

The reason is that all blocks of a sequence are typically accessed together. For decoding algorithms with several related sequences, such as beam search:

- The whole sequence group is scheduled together.
- The whole group is preempted or resumed together.

This is useful because members of the group may share physical blocks.

To restore preempted sequences, there are two strategies.

1. Swapping. This is similar to operating-system swapping. When vLLM needs free GPU blocks:

1. It selects some sequences for preemption.
2. Their KV-cache blocks are copied from GPU memory to CPU RAM.
3. GPU blocks are freed for active requests.
4. Later, the blocks can be copied back to GPU memory.
5. The preempted sequences resume.

vLLM includes:

- A GPU block allocator
- A CPU block allocator

The CPU allocator manages physical blocks stored in CPU RAM. Once some requests have been preempted, vLLM may temporarily stop accepting new requests until the preempted sequences finish. When an active request finishes:

- Its GPU blocks are released.
- Swapped sequences can be restored.

The number of CPU-swapped blocks never exceeds the total physical KV blocks that can be allocated in GPU memory, so the CPU swap requirement remains bounded relative to the GPU KV-cache capacity.

2. Recomputation. Instead of swapping KV-cache blocks to CPU memory, the system can discard them and recompute them later. When a preempted sequence is rescheduled:

- Its generated tokens are concatenated with the original user prompt.
- The combined sequence is treated as a new prompt.
- The entire KV cache is regenerated during a prompt phase.

Recomputation latency may be significantly lower than the original generation latency because:

- Prompt processing is parallel.
- Autoregressive generation is sequential.

Therefore, regenerating the KV cache in one prompt pass may be cheaper than generating the same tokens one by one again.

Many LLMs are too large to fit on one GPU. They must be distributed across several GPUs using model parallelism. There is a Megatron-LM-style tensor-parallel strategy. The system follows an SPMD model:

$$
\text{SPMD} =
\text{Single Program Multiple Data}.
$$

In this approach:

- Linear layers are split across GPUs.
- Matrix multiplication is partitioned.
- GPUs use all-reduce operations to synchronize intermediate results.
- Attention heads are divided across SPMD processes.
- Each process handles a subset of attention heads.

Even under model parallelism, every model shard processes the same set of input tokens. Therefore, each shard must have the corresponding KV cache for its assigned attention heads. vLLM maintains one copy of the logical KV-cache manager in the centralized scheduler. Different GPU workers share:

- The same logical-to-physical block mapping
- The same physical block IDs

However, each worker stores only the KV-cache data for the attention heads handled by that model shard. The shared mapping allows all workers to interpret the same request in the same way.

The distributed execution flow is as below:

1. Scheduler prepares control information. For each request in the batch, the scheduler prepares:

- Input token IDs
- The request’s block table

2. Broadcast to GPU workers. The scheduler broadcasts this control information to all GPU workers. Each worker begins model execution using the same input token IDs. At the attention layer, each GPU worker reads KV cache according to the supplied block table. During model execution, workers synchronize intermediate results using all-reduce. The scheduler does not need to coordinate the workers during this phase.

3. Return sampled token. At the end of the decoding step, the GPU workers return the sampled token to the scheduler. Workers do not need to synchronize KV-cache memory-management decisions among themselves. They receive all memory-management information at the beginning of each iteration.

#### 5.2.2.5 SGLang

SGLang is a programming and runtime system for building complex LLM workflows.

It is designed for tasks that go beyond a single prompt followed by a single model response. These tasks may involve:

- Multiple model calls
- Control flow
- Parallel branches
- Structured inputs and outputs
- Images or videos
- Repeated prompt prefixes
- Multiple rounds of reasoning
- External API models
- Different models working together

Examples include:

- Few-shot learning
- Self-consistency
- Chain-of-thought and tree-of-thought workflows
- Multi-turn chat
- Structured JSON generation
- Multi-model pipelines
- Agent-style planning and interaction

Treat an LLM workflow as a program, then optimize both how that program is expressed and how it is executed. These programs are called LM programs. A simple chatbot may make one model call:

- Send a prompt.
- Receive an answer.

More advanced applications are different. They may need to:

1. Call an LLM several times.
2. Insert those calls into conditional logic.
3. Branch into several parallel generations.
4. Merge the results.
5. constrain the final output to JSON or another schema.
6. Reuse common prompt prefixes and KV cache.
7. Interact with external tools or environments.

LM programs generally have two common properties:
- They contain multiple LLM calls embedded in a control flow.
- They accept structured inputs and produce structured outputs.

Modern LLM inference is more than inference for one isolated model. It often involves combinations of models running within a particular task workflow. This creates two major problems:

- Users must write complicated prompts and orchestration logic.
- Repeated model calls often recompute the same KV cache, wasting memory and computation.

SGLang addresses both problems:

- It provides programming primitives for expressing workflows.
- It provides a runtime that automatically optimizes their execution.

SGLang consists of two main parts:

1. SGLang Client, the frontend
2. SGLang Runtime, the backend

Between them is an interpreter. A simplified flow is:

SGLang program
    ↓
Interpreter
    ↓
API server
    ↓
SGLang Runtime
    ↓
Tokenizer → request queue → scheduler → GPU workers
    ↓
Detokenizer → result

The frontend focuses mainly on:

- Expressing LM programs
- Prompt construction
- Control flow
- Parallelism
- Structured output
- Multimodal input

The backend focuses mainly on:

- Efficient inference
- Scheduling
- KV-cache reuse
- Faster structured decoding
- Optimizing repeated API calls

SGLang provides a domain-specific language, or DSL, embedded in Python. It includes primitives for:

- Parallel execution
- Control flow
- Nested model calls
- External calls
- Structured generation
- Image and video inputs

Its goal is to make LM programs easier to write while allowing the interpreter, compiler, and runtime to optimize them. The frontend manages the scheduling and control structure of the LM program. The simplest way to run an SGLang program is through its interpreter. The prompt is treated as an asynchronous stream. This allows Python code to continue running without waiting for every generation to finish immediately. Each prompt has a background stream executor that manages execution. The program blocks only when it actually needs a generated result. This is similar to launching an asynchronous CUDA kernel:

- Work is submitted.
- Independent code can continue.
- Synchronization occurs only when the result is required.

SGLang programs can also be compiled into computation graphs and executed through a graph executor for further optimization and reduction of redundant work. 

### 5.2.3 Optimization Frameworks

#### 5.2.3.1 DeepSpeed

DeepSpeed is an open-source deep learning optimization system developed by Microsoft. It is designed for Large-model training, Distributed training, Inference, and Model compression. Its most important training optimization is ZeRO, short for Zero Redundancy Optimizer. The central problem DeepSpeed addresses is: In distributed training, memory efficiency and communication overhead often conflict with each other. Ordinary data parallelism is computationally efficient, but it duplicates a large amount of model state on every GPU. ZeRO changes this by progressively partitioning those states across devices.

There are two common ways to distribute neural-network training across multiple GPUs. In data parallelism, every GPU stores a complete copy of the model. Data parallelism usually has high computational efficiency because every GPU can work on a different subset of the data at the same time. However, every GPU stores the entire model, gradients, and optimizer states. This creates a large amount of memory redundancy. In model parallelism, the model itself is divided across multiple GPUs. This is necessary when the entire model cannot fit into one GPU.

The key challenge in data parallelism is ensuring that all model replicas remain identical. For every process to end with identical parameters, two things must be true:

- Every process begins with the same initial parameters W0.
- Every process uses the same gradient ΔW at each update.

Two common approaches are used in synchronizing initial paramters. The first one is that every process uses the same random seed and initializes parameters in the same order. The second one is that, one process initializes the model, then broadcasts the parameters to all other processes. Each process independently runs the forward pass on its local micro-batch. Because the input data differs, each process obtains a different local loss. Each process independently computes gradients from its own local loss. Therefore, different processes initially produce different local gradients. Before updating the model, these gradients must be synchronized. Gradient synchronization is commonly implemented with an AllReduce sum. All local gradients are added together across processes. The result is then divided by the number of data-parallel processes to obtain the average gradient. After synchronization, every process has the same global gradient. Each process can then update its local model independently. The main difference from single-GPU training is therefore: In data parallelism, gradients must be synchronized across all processes before the parameter update.

Modern language-model training commonly uses Mixed-precision training and the Adam optimizer. Mixed-precision training stores and computes some values in FP16 while keeping more numerically sensitive values in FP32. 

DeepSpeed separates GPU memory into two broad categories: Model states and Residual states. Model states include:

- Model parameters
- Gradients
- Optimizer states

For mixed-precision Adam training, the major model states are as follows. 

1. FP16 model parameters. Each parameter is stored in FP16. Each value requires 2 bytes. For Φ parameters, the memory requirement is 2Φ bytes.

2. FP16 gradients. The number of gradient values equals the number of parameters. Each gradient also uses 2 bytes. The memory requirement is 2Φ bytes.

3. Adam optimizer states. Adam maintains several FP32 values for every parameter. Adam often maintains an FP32 copy of each model parameter for more accurate updates. Memory cost is 4Φ bytes. Adam stores a first-moment estimate, or momentum, for every parameter. For this part, memory cost is 4Φ bytes. Adam also stores a second-moment estimate, or variance, for every parameter. This part is also 4Φ bytes. Therefore, Adam optimizer states require 12Φ bytes.

The complete model-state memory is 16Φ bytes. The Adam-related FP32 states account for 75%. Therefore, in mixed-precision Adam training, optimizer states consume approximately 75% of model-state memory. This is why optimizer-state partitioning is the first and most valuable ZeRO step.

In addition to model states, training consumes memory for:

- Activations
- Temporary buffers
- Memory fragmentation
- Other intermediate tensors

ZeRO mainly targets redundancy in the model states. Activation-memory optimization is a separate concern, although DeepSpeed has other mechanisms for it. In ordinary distributed data parallelism, every GPU stores a complete copy of:

- Model parameters
- Gradients
- Optimizer states

This is highly redundant. ZeRO partitions these states across GPUs so that the distributed system collectively maintains one complete copy, rather than every GPU independently maintaining a full copy. If there are N GPUs, each GPU ideally stores only 1/N of a partitioned state. ZeRO has four conceptual stages:

- Stage 0: ordinary DDP with no ZeRO memory optimization

- Stage 1: partition optimizer states

- Stage 2: partition optimizer states and gradients

- Stage 3: partition optimizer states, gradients, and model parameters

If no stage is specified in deepspeed_config, DeepSpeed defaults to Stage 0. Stage 0 is standard distributed data parallelism. Each GPU stores complete copies of parameters, gradients, and optimizer states. The memory requirement per GPU is 16Φ bytes. There is no ZeRO partitioning.

Stage 1 partitions the optimizer states across all devices. Each GPU stores only 1/N of of the optimizer states. However, every GPU still stores a complete copy of FP16 parameters and FP16 gradients. The memory per GPU becomes 2Φ+2Φ+12Φ​/N. As N becomes large, this approaches 4Φ. That is about one quarter of the original 16Φ. Each GPU is responsible for updating only the parameter partition associated with its optimizer-state shard. After local updates, the updated parameter pieces are gathered so that all GPUs again have a complete parameter copy. Stage 1 has approximately the same communication volume as ordinary data parallelism.

Stage 2 additionally partitions gradients. Now each GPU stores:

- A complete copy of parameters
- Only 1/N of gradients
- Only 1/N of optimizer states

The memory per GPU becomes 2Φ+ (2Φ+12Φ​) / N. As N grows, this approaches 2Φ. That is about one eighth of the original 16Φ. Instead of every GPU keeping the full synchronized gradient, each GPU ultimately retains only the gradient partition for which it is responsible. The global synchronization can be implemented using ReduceScatter. ReduceScatter reduces gradient contributions across all GPUs, and distributes different pieces of the reduced result to different GPUs. Therefore, each GPU receives only its assigned averaged-gradient partition. Stage 2 has approximately the same communication volume as ordinary data parallelism.

Stage 3 further partitions the model parameters. Each GPU stores only:

- 1/N of parameters
- 1/N of gradients
- 1/N of optimizer states

The model-state memory per GPU becomes 16Φ​ / N. As N becomes large, per-GPU model-state memory approaches zero relative to the unpartitioned model. This enables the training of extremely large models even when no single GPU can store the full model state.

Although parameters are partitioned at rest, a layer needs its full parameters when that layer is computed. Therefore, the required parameter shards must be communicated before forward and backward computation. This introduces extra communication, commonly through operations such as Broadcast and AllGather. After the layer no longer needs the gathered parameters, the temporary full copy can be released. Stage 3 increases communication volume to approximately 1.5× the data-parallel baseline. The extra communication occurs because model parameters are also partitioned and must be reconstructed when needed.

| Stage   |  Parameters |   Gradients | Optimizer states | Approximate per-GPU model-state memory |
| ------- | ----------: | ----------: | ---------------: | -------------------------------------: |
| Stage 0 |  Replicated |  Replicated |       Replicated |                               $(16\Phi)$ |
| Stage 1 |  Replicated |  Replicated |      Partitioned |               $(4\Phi+\frac{12\Phi}{N})$ |
| Stage 2 |  Replicated | Partitioned |      Partitioned |               $(2\Phi+\frac{14\Phi}{N})$ |
| Stage 3 | Partitioned | Partitioned |      Partitioned |                     $(\frac{16\Phi}{N})$ |

The progression is:

- Remove redundant optimizer states.
- Remove redundant gradients.
- Remove redundant parameters.

When to use each ZeRO stage:

1. ZeRO-1. Appropriate when:

- The model fits on one GPU.
- Optimizer states cause out-of-memory errors.
- Maximum training speed is important.
- Communication overhead should remain similar to baseline data parallelism.

2. ZeRO-2. Appropriate for:

- Large models, including models below roughly 70B parameters 
- Situations requiring a balance between memory reduction and training speed

If ZeRO-2 memory consumption exceeds approximately 75% of GPU capacity, upgrade to ZeRO-3.
  
3. ZeRO-3. ZeRO-3 can be combined with offload and DeepSpeed Infinity to extend training to even larger scales. It is appropriate for:

- Models with hundreds of billions or trillions of parameters
- Cases in which parameters themselves cannot fit on one GPU
- GPT-4-scale model training scenarios

DeepSpeed can move some training states from GPU memory to CPU memory. The configuration can include offload_optimizer and offload_param. This reduces GPU-memory usage and makes larger models trainable. However, there are more communication between CPU and GPU, increased data-transfer latency, and potentially slower the training.

Distributed training relies on collective communication primitives. There are five important operations.

1. Reduce. Reduce aggregates data from multiple processes onto one destination process. For example, it may sum the values from all GPUs and place the result on one root GPU. The data flow is from many devices to one device. Typical use is to combine gradients or statistics onto one processor.

2. Broadcast. Broadcast sends data from one source process to every process. Data flow is from one device to all devices. Typical use is to broadcast initialized model parameters from one root process.

3. AllGather. AllGather collects each process’s local data and sends the complete combined result to every process. Afterward, every GPU has the same concatenated result. Data flow is that every device communicates with every device, so that every device receives the complete data. Typical uses include reconstructing complete model parameters from shards or distributing parameter partitions.

4. AllReduce. AllReduce combines data across all processes and distributes the aggregated result to all processes. For gradients, it typically reduces local gradients by summing them, and broadcasts the result to all GPUs. After division by the number of processes, every GPU has the same average gradient. Typical use is gradient synchronization in ordinary data parallelism. A common Ring AllReduce can be implemented as ReduceScatter and AllGather.

5. ReduceScatter. ReduceScatter combines values across all processes and distributes a different partition of the reduced result to each process. Each GPU receives only one shard of the globally reduced data. The data flow is every device sends and receives data, and each device ends with a different partition. Typical uses include grdient partitioning in ZeRO Stage 2 and synchronizing only the gradient shard owned by each GPU.

In ordinary data parallelism, every GPU computes a complete gradient tensor. AllReduce is used to obtain the global average gradient. The communication volume per GPU is 2Φ bytes for the gradient synchronization process, assuming FP16 gradients. In Stage 2, every GPU keeps only 1/N of the gradients and optimizer states. For a given GPU, the required gradient partition must be reduced across all N GPUs. For one partition, the total reduction communication is Φ. The system uses a bucket strategy so that gradient communication can overlap with backward computation. Gradients are communicated as soon as suitable buckets are ready instead of waiting for the entire backward pass to finish. Once a GPU has the averaged gradient for its partition, it updates its local optimizer states. At the end, the updated parameter partitions are gathered. Globally, this corresponds to ReduceScatter for gradients and AllGather for updated parameters.  The total communication is approximately the same as ordinary data parallelism. Suppose the final two layers’ gradients belong to GPU 0. Other GPUs do not need to permanently keep these gradients. They can:

1. Compute a gradient bucket.
2. Send the bucket while continuing to backpropagate through earlier layers.
3. Delete the local gradients after the communication and dependent computation are complete.

This overlaps communication with computation and reduces the peak gradient-memory requirement.

In Stage 3, each GPU stores only 1/N of the model parameters. Therefore, before computing a layer, the full layer parameters must be made available. This requires extra communication during forward propagation and backward propagation. This involves Broadcast operations, although equivalent implementations may use collective gathering of parameter shards. Because parameter communication happens in both forward and backward passes, Stage 3 increases total communication compared with Stage 1 and Stage 2.

Traditional thinking says:

- Data parallelism is computationally efficient but duplicates memory.
- Model parallelism saves memory but adds communication.

ZeRO introduces another option:

- Keep the data-parallel execution model.
- Partition model states to remove redundancy.
- Communicate only the required state shards at the appropriate time.

This allows DeepSpeed to preserve much of the efficiency of data parallelism while greatly reducing memory usage.

Here we also introduce DeepSpeed-MII as a system for low-latency inference and high-throughput inference. 

- Prompt caching. The model can be run once for a fixed prompt, and the resulting model state can be cached. Future requests using the same static prompt can reuse the cached state.

- Dynamic memory utilization. DeepSpeed-MII includes cache allocators on both CPU and GPU. Memory utilization can change dynamically according to request size while still meeting performance requirements.

#### 5.2.3.2 vLLM

vLLM is an LLM serving system that sits between incoming API requests and the model’s GPU kernels. It does not change the model’s probability distribution. Instead, it continuously decides:

- Which requests should run in the current iteration
- How many tokens each request should compute
- Where each request’s KV cache should be stored
- When requests should enter, leave, pause, or resume
- How outputs should be streamed back to the caller

The best way to understand vLLM is to treat one request as a complete lifecycle:

Request arrival
→ protocol parsing and tokenization
→ scheduling
→ model execution
→ sampling and output processing
→ streaming response
→ resource release

Its core abstractions are not a fixed training-style batch. They are a dynamically changing set of tokens to compute, the KV-cache blocks that store token state, and a scheduler that rebuilds the executable batch every iteration. Training usually operates on preassembled batches with stable boundaries including forward pass, backward pass, and optimizer step. Online serving is much less regular.

A serving system must deal with:

- Prompts of different lengths
- Unknown output lengths
- Requests arriving at arbitrary times
- Requests ending at different times
- Streaming one token at a time
- Cancellations and timeouts
- Limited KV-cache capacity
- Different sampling and multimodal configurations

Therefore, simply loading a model onto a GPU does not complete the serving system. vLLM is the control layer between HTTP requests and model kernels. It decides which tokens run during every engine iteration and manages their associated state. Autoregressive inference has two computation phases. Prefill processes multiple prompt tokens at once. For every layer, it computes Q, K, V and MLP operations. The K and V values for all prompt positions are written into the KV cache. Because tokens within a prompt can be processed in parallel, prefill commonly has larger matrix multiplications, better GPU compute utilization, and a workload closer to compute-bound execution.

Decode normally advances each active sequence by one or a small number of tokens per iteration. For every new query token, attention reads the cached K and V values of all preceding positions. Decode therefore tends to have very thin matrices for each request, large KV reads, strong memory-bandwidth pressure, more kernel-launch and scheduling overhead, and lower arithmetic intensity than prefill. Prefill and decode use the same model, but they have very different execution shapes.

Assume a Transformer has:

- $\(L\)$ layers
- $\(n_{kv}\)$ KV heads
- Head dimension $\(d_h\)$
- $\(b\)$ bytes per KV element
- Sequence length $\(T\)$

The theoretical KV-cache memory required by one sequence is approximately:

$$
M_{KV}(T) = 2LTn_{kv}d_hb.
$$

The factor $\(2\)$ represents:

- Keys
- Values

For multi-head attention, the number of KV heads may equal the number of query heads.

GQA and MQA reduce $\(n_{kv}\)$, so they reduce the KV-cache memory required for the same context length.

For an online batch, total KV memory is not a fixed rectangle based on the longest sequence. It is approximately:

$$
\sum_i M_{KV}(T_i),
$$

and it grows as active sequences generate more tokens.

In training, a batch is usually a fixed collection of samples. In vLLM, the core scheduling object is closer to a changing collection of tokens together with the KV blocks carrying their state. Each engine iteration consists approximately of three stages:

1. Schedule
2. Model execution
3. Postprocessing

The scheduler chooses tokens from waiting and running requests. The worker executes the model using selected token IDs, slot mappings, block tables, and attention metadata. After sampling, the engine updates stopping conditions and streams results. New requests can join while older requests are still generating, and completed requests immediately release their resources. This is the basis of continuous batching.

A mixed prefill/decode schedule tries to combine the large-matrix efficiency of prefill, the concurrency and throughput of decode, and the avoidance of waiting for the slowest sequence in a static batch. However, prefill and decode compete for the same GPU. Too much prefill can delay currently streaming requests, increase inter-token latency, and stretch the time between decode iterations; And too much decode can delay newly arrived requests, increase time to first token, and build a long admission queue. Therefore, serving optimization is not simply maximizing tokens per second. The scheduler must find a working point among admission rate, KV-cache capacity, per-iteration token budget, time-to-first-token targets, and inter-token-latency targets.

Online workloads are dynamic. The following may change continuously: Sequence lengths; Prefill/decode ratio; Sampling branches; Request arrival rate; Output lengths; and KV-cache use. The same request concurrency may correspond to very different numbers of scheduled tokens, KV-memory consumption, and kernel shapes. A reproducible serving benchmark should therefore fix at least the model, the data type, the input-length distribution, output-length distribution, the arrival process, the concurrency limit, the sampling parameters, and the hardware topology. During training, larger batch size could be better, but this is unreliable during inference.

Offline inference is often summarized by tokens per second. Online serving requires separate measurements of queueing and token experience. It may include queueing, admission, tokenization, prefill, initial sampling, and network transmission. Its common bottlenecks include admission delay, prefill work, and tokenization. A good average TTFT alone can hide a long tail for large prompts.

ITL is the interval between consecutive streamed output tokens. Average ITL is often called TPOT, or time per output token. Common sources of poor ITL include decode-bandwidth limits, preemption, interference from prefill, and poor decode batching. A good average TPOT does not guarantee that a request’s output stream is smooth.

For $\(N_o\)$ output tokens, a useful approximation is:

$$
T_{e2e} \approx T_{TTFT} + (N_o - 1)T_{TPOT}.
$$

This is not an exact identity because streaming, network transfer, detokenization, queueing, and tail processing also contribute.

Request throughput measures completed requests per unit time. It depends strongly on input and output lengths, scheduling, and KV-cache capacity. It cannot be compared fairly across workloads with different output-length distributions.

Token throughput measures the total number of processed tokens per unit time. It is influenced by tokens scheduled per batch, kernel efficiency, and batch shape. It may improve by sacrificing individual-request latency.

Goodput measures useful throughput that satisfies an explicit service-level objective. It depends on tail latency, error rate, timeout rate, cancellation, preemption, and defined TTFT and ITL targets. For online systems, the objective should usually be to maximize goodput subject to latency constraints. It should not be unconstrained peak throughput.

Production evaluation should inspect percentiles such as P50, P90, and P99. It should also include successful request throughput, timeouts, cancellations, preemptions, recomputation, and queue growth. A good stress test should increase request arrival rate gradually until the SLO first becomes unsatisfied. At that point, record request mix, KV-cache utilization, preemption frequency, running and waiting request counts, prompt and generation token counts, and engine iteration time. Testing only one request’s latency ignores continuous batching. Testing only offline throughput ignores queueing and tail latency.

Separating metrics by serving phase makes tuning more diagnostic.

- High TTFT: inspect queueing, admission, tokenization, and prefill.
- High ITL: inspect decode batch shape, communication, and preemption.
- Low throughput: inspect kernels, quantization, tensor parallelism, and batch formation.
- Poor goodput: inspect tail latency, error rate, and SLO definitions.

Monitoring should include running requests, waiting requests, KV-cache usage, cache-hit rate, preemption count, prompt-token count, generation-token count, and engine iteration time.

System capacity is constrained jointly by three main boundaries.

1. Model-weight capacity and bandwidth. The model weights must fit, and they must be read efficiently during inference. Quantization primarily changes weight size and some GEMM characteristics.

2. KV-cache capacity. The KV cache determines how many active token states can be retained. KV quantization primarily changes bytes per cached token. GQA changes the number of KV heads and KV bytes per token.

3. Scheduler token budget. The scheduler must form a sufficiently large and efficient executable batch under a per-iteration token budget. PagedAttention primarily changes KV allocation efficiency. These techniques affect different accounting categories. One should not use a vague statement such as “memory usage decreased” to substitute for measuring each effect separately.

At low concurrency, the GPU may be underutilized because:

- GEMMs are too small
- Kernel launches are frequent
- Batches are too thin

As concurrency rises, batching improves throughput. At still higher concurrency, the system may become limited by KV-cache capacity, queueing, network saturation, scheduling, and prermption. P99 can then worsen sharply. The correct objective is the maximum goodput satisfying the latency constraint, not the highest unconstrained throughput. This is also why the same configuration may not serve equally well:

- Short conversational requests
- Long-document prefill
- Very long reasoning outputs

Their token shapes and ideal budgets are different. 

vLLM V1 separates user-facing request handling, scheduling and KV management, and GPU model execution into clearer process boundaries. An offline LLM class drives synchronous requests through an LLMEngine. An OpenAI-compatible API server uses an AsyncLLMEngine. It creates asynchronous channels among HTTP handling, tokenization, streaming output, and engine iterations. The Engine Core is responsible for per-iteration scheduling, KV-cache management, and maintaining request state. The GPU Worker is responsible for loading the model and executing model kernels. This separation lets API concurrency and GPU execution scale more independently.

In a multi-GPU configuration:

- Each data-parallel rank corresponds to an Engine Core.
- Each Engine Core controls a number of workers equal to:

$$
TP \times PP,
$$

where:

- $\(TP\)$ is the tensor-parallel degree.
- $\(PP\)$ is the pipeline-parallel degree.

If:

- $\(A\)$ is the number of API-server processes,
- $\(D\)$ is the data-parallel degree,

then the total process count is described as:

$$
N_{\text{proc}} =
A + D + D(TP \times PP) + \mathbf{1}[D > 1].
$$

The final term represents an additional data-parallel coordinator when $\(D > 1\)$.

Data parallelism in serving is not training gradient synchronization. Each serving replica handles different requests independently. It does not perform per-step gradient AllReduce. Tensor parallelism and pipeline parallelism distribute one model execution across multiple GPUs, so they enter the latency path. A useful deployment principle is, first make a single replica large enough to hold and run the model. Then use data parallelism to scale request throughput. Data parallelism, tensor parallelism, and pipeline parallelism should not be treated as interchangeable decompositions of the same GPU count.

The API server and Engine Core communicate through ZMQ. When multiple data-parallel replicas are present, a coordinator collects load information from ranks and helps route requests. Each replica processes different requests independently. Tensor- and pipeline-parallel communication is used only inside a replica to execute the sharded model. The Engine Core often runs a persistent busy loop and can be sensitive to CPU scheduling and process contention. The API server also performs tokenization, JSON handling, template rendering, and network I/O. Therefore, apparent GPU idleness, TTFT jitter, or streaming stalls may be caused by CPU-side problems. Potential hidden bottlenecks include CPU quotas, NUMA placement, thread pools, tokenizer parallelism, network-card interrupts, and ZMQ fan-out. Increasing API-server process count can reduce an entry-layer bottleneck for high-QPS, short-request workloads. It is less helpful when a single request’s model execution is itself slow. In that case, the relevant options may be tensor parallelism, pipeline parallelism, quantization, CUDA graph, or a more suitable model.

Each API-server and Engine-Core process consumes CPU, memory, connections, and IPC resources. Too many API processes may cause tokenization contention and ZMQ fan-out overhead. Excessive tensor parallelism may place collective communication on the critical path. Pipeline parallelism may introduce pipeline bubbles. A deployment should be verified using process-level CPU usage, resident set size, IPC queue behavior, and GPU utilization. vLLM’s interface layer, scheduler, KV-cache manager, attention backend, and model implementation may evolve at different rates. Before applying a configuration, verify:

- Whether the system is using the V1 path or a compatibility path
- Which attention backend is active
- Whether a parameter truly participates in the current execution graph

Blindly copying a configuration from an older version may produce a setup that launches but does not behave as expected. The difficult part of KV caching is not merely total size. It is the irregular lifecycle. When a request arrives, only the prompt is known. Generation may end after one token, continue to the maximum length, be cancelled, or be preempted. If every request reserves contiguous memory for its maximum possible length, several types of waste appear. Unused capacity is reserved for future tokens that may never be generated. The final partially used allocation wastes space inside the reserved region. Free memory exists, but not as one sufficiently large contiguous region. A request cannot enter because no sufficiently large contiguous segment exists, even when total free memory is adequate.

PagedAttention divides KV cache into fixed-size blocks containing a fixed number of token slots. The allocation unit changes from a contiguous region for the request’s possible maximum length to the blocks required by the sequence’s current length. It solves an address-space and physical-memory-binding problem. It does not make the mathematical attention pattern sparse.

Let:

- $\(B\)$ be the block size in tokens
- $\(T\)$ be the current sequence length

The number of logical blocks required is:

$$
N_{\text{block}}(T) =
\left\lceil
\frac{T}{B}
\right\rceil.
$$

All blocks except the final block are full. Therefore, the internal waste for one sequence is strictly less than B token slots.

If one token’s KV state occupies $\(m_t\)$ bytes, then one block occupies:

$$
M_b = Bm_t.
$$

If the KV-memory pool has capacity \(M_{\text{pool}}\), the number of blocks that fit is approximately:

$$
\left\lfloor
\frac{M_{\text{pool}}}{M_b}
\right\rfloor.
$$

Admission then becomes how many free blocks are available. It no longer depends on finding one large contiguous region.

Fixed blocks provide several direct benefits:

- Requests can grow incrementally.
- Completed requests can immediately return blocks.
- Requests of different lengths can share one memory pool.
- External fragmentation is largely removed.
- The scheduler can admit requests according to available blocks.
- Shared-cache mechanisms can operate on block IDs and reference counts.

The same basic block abstraction can support parallel sampling, beam search, prefix caching and cross-instance KV transfer without copying an entire contiguous cache.

Block size balances memory efficiency and kernel efficiency. Smaller blocks have less tail waste, finer-grained allocation and better chance of matching shorter prefixes. But they also have longer block tables, more metadata, more block-ID lookups and more address segments for kernels. Larger blocks have better contiguous access, better vectorization and lower metadata overhead, but more waste in the final block, lower utilization for short sequences, and a shared prefix ofte nmust cover a whole block before it can be reused. The appropriate value depends on model KV-head count, data type, typical request lengths, attention backend, and head dimesion. It should not be tuned in isolation.

Each request sees a contiguous sequence of logical blocks. The GPU KV-cache pool contains fixed-ID physical blocks. A block table maps logical block to physical block. The scheduler and KV-cache manager can assign any available physical block to a new logical block. Logical token order remains contiguous even when physical addresses are not. This is similar to virtual-memory page tables.

For token position $\(t\)$:

$$
j =
\left\lfloor
\frac{t}{B}
\right\rfloor,
\qquad
o =
t \bmod B.
$$

Here:

- $\(j\)$ is the logical block number.
- $\(o\)$ is the offset inside that block.

The physical block is:

$$
p =
table[j].
$$

The token’s $\(K\)$ and $\(V\)$ values are written to offset $\(o\)$ inside physical block $\(p\)$.

The actual slot mapping expands this logical location into addresses usable by the kernel across:

- Layers
- KV heads
- Head dimensions

The model sees tokens in logical order. Address rearrangement is handled jointly by the cache manager and attention backend.

Block allocation is incremental. During prefill, it will estimate how many new token positions will be added, and allocate a new block only if the current tail block is full or insufficient. During decode, most iterations add one position. Usually only the tail block’s filled count changes. The block table changes only when a block boundary is crossed. When a request finishes, is cancelled or is preempted, blocks whose reference count reaches zero return to the free queue. 

The logical interpretation of a sequence is determined by token order and causal masking. The physical memory location is determined by the block allocator. These are separate coincerns. This is why logical and physical separation matters. 

The block pool usually precreates block metadata rather than creating large numbers of Python objects every iteration. Each block may track immutable block ID, reference count, reusable hash, and free-queue links. The free queue supports efficient removal and reinsertion. A cached block may remain in the free queue while still containing valid reusable data. If that block is hit by prefix caching, the system can “touch” it by removing it from the free queue, incrementing its reference count, and reusing it without rewriting the GPU KV data. Only when a truly free block is required does the allocator evict from the head of the queue. This makes the fast allocation path mostly metadata manipulation.

A block appearing in the free queue does not mean its contents have been erased. A cache system may retain its hash and its previous KV contents until the block is actually reassigned. This is delayed eviction. It prevents reusable KV data from being deleted as soon as a request releases it. However, incorrect handling of hashes, tenant isolation and reference counts can cause incorrect cache hits or cross-tenant leakage. For multi-tenant systems, cache key, salt, and access scope must be designed together.

Paging does not remove every source of memory overhead. Remaining costs include tail-block waste, different block requirements among attention layers, CUDA graph pools, workspace memory, metadata, alignment, attention-backend buffers, NCCL buffers, model weights, and actications. Capacity calculations should record these separately. It is incorrect to treat all memory covered by a setting such as gpu_memory_utilization as available KV-cache space.

A unified Block Pool allows cache manager, scheduler, and attention kernel to use the same IDs. This avoids repeatedly translating among Python objects, allocator handles, and GPU addresses. It also makes memory level observable. The system can directly monitor number of free blocks, number of cached blocks, number of active references, and number of new blocks allocated during the current iteration. These values can feed admission control and monitoring.

PagedAttention is not only an allocator. The attention kernel must use the block table to gather physically scattered keys and values while preserving standard causal-attention normalization.

For query $\(q_i\)$, suppose the logical KV history is divided into $\(J\)$ blocks. The block-level scores are:

$$
s_j =
\frac{q_iK_j^\top}{\sqrt d}.
$$

The output can be written as:

$$
o_i =
softmax
\left(
[s_1,\ldots,s_J]
\right)
[V_1;\ldots;V_J].
$$

The mathematics still attends to every permitted historical token. Only the physical source addresses of $\(K_j\)$ and $\(V_j\)$ come from the block table.

The kernel should not independently apply softmax to each block. It must preserve normalization across all blocks. A numerically stable implementation keeps, for each partition:

- A local maximum mj
- A local exponential sum ℓj

These are rescaled and merged using the global maximum. This prevents block-wise normalization from changing the result. The kernel also avoids materializing the entire attention-score matrix.

A single-query decode kernel often maps a combination such as one sequence, one attention head, and one partition to a thread block. The query is small and reused across many KV tokens, so it is suitable for registers and shared memory. K reads can be divided among thread groups to enable vectorized loading and coalesced access. V has a different layout and reduction direction, so threads may read contiguous elements to support accumulation. The block table introduces one level of address indirection, but accesses inside each block remain regular.

PagedAttention describes a KV-addressing abstraction. It does not imply one fixed CUDA implementation forever. Depending on GPU, data type, head size, model structure, and enabled features, vLLM may choose different attention backends, such as FlashAttention-style backends, decode-specific kernels, quantized-KV kernels, and backends compatible with hybrid KV managers. When analyzing performance, first confirm the actual backend in use. The name “PagedAttention” alone does not prove that a particular historical kernel is active.

Very small blocks may cause more block-ID reads, more softmax partition merges, and more metadata work. Very large blocks may cause more waste on short sequences and low thread utilization when the last block is sparsely filled. Other factors affecting effective block-byte width include head size, number of KV heads, and tensor-parallel sharding. Tuning should inspect memory throughput, occupancy, kernel-launch count, actual batch shape, and kernel time, not only cache utilization.

The block table and sequence metadata must be updated consistently. A typical iteration proceeds as follows:

1. Scheduler chooses tokens.
2. KV manager ensures the required slots exist.
3. Model Runner builds input positions, slot mappings, and attention metadata.
4. Kernel writes new KV and reads historical KV.
5. Sampling completes.
6. Scheduler advances the sequence length.

Any ordering mistake can cause a new token to overwrite an old slot, incorrect prefix reads, and a block to be reclaimed while still referenced. These bugs may be harder to detect than an out-of-memory error because tensor shapes may remain valid.

When allocator, metadata builder, and kernel all use the same paging protocol:

- A sequence can grow without relocating its existing KV.
- Block IDs from several requests can be assembled into one batch.
- Requests can enter or leave at iteration boundaries.
- Existing caches do not need to be rearranged.

Batch changes require metadata updates rather than moving every sequence’s contiguous KV cache. This is the foundation of iteration-level continuous batching.

Paging improves allocation efficiency, memory reuse, fragmentation, and admission flexibility. It does not remove the need to read long-context KV history during decode. For a long context, one-step decode still reads an amount of KV that grows approximately with context length. It can still be limited by HBM bandwidth and interconnect bandwidth. Reducing the KV-read scale requires other techniques, such as sliding-window attention, local or global sparse patterns, KV eviction, context parallelism, and model-level state compression. These can be combined with PagedAttention, but they solve a different problem.

To judge whether a paging optimization works, inspect at least number of valid KV tokens, allocated blocks, tail-block waste, cache-hit rate, prermption, and attention-kernel time. A lower memory-use figure by itself cannot determine whether block management improved, batch size became smaller, KV data type changed, and requests happened to be shorter. Static batching chooses a request set before execution and holds the batch until all sequences complete. For variable-length generation, shorter sequences remain idle while waiting for the longest sequence. Continuous batching moves the scheduling boundary to one engine iteration. At the end of every iteration, the scheduler rechecks waiting requests, running requests, finished requests, and cancelled requests. Finished requests leave immediately. New requests may enter as long as they have token budget and required KV blocks. The batch is therefore an active set reconstructed every iteration.

The scheduler’s input is not a simple queue entry. Each request may contain number of computed tokens, number of generated tokens, target token count, priority or arrival time, sampling branches, encoder or multimodal requirements, and associated KV blocks. The scheduler must decide how many tokens each request computes this iteration, how many new blocks are needed, and which requests must be delayed or preempted. The Model Runner then packs the selected tokens into a flat executable batch.

In steady decode, a normal request may contribute one token per iteration. If there are R running sequences, the decode workload is roughly R tokens. A single prefill request may contribute hundreds or thousands of tokens. Therefore, limiting by “number of requests” incorrectly treats prefill and decode as equally expensive. vLLM V1 emphasizes the number of scheduled tokens. The key idea of continuous batching is not only that requests can dynamically enter and leave. It is that the scheduler redoes admission every iteration using both token resources and KV resources.

A queue policy must balance FCFS fairness, priority, and resource feasibility. Strict shortest-request-first may improve average latency but can starve long prompts. Strict FCFS is easier to explain, but a very long prefill can block later short requests unless it can be chunked. Production systems must also handle client disconnects. If an upstream client has cancelled but the engine does not release the request, the GPU may continue computing useless tokens. Cancellation propagation, timeout handling, and block reclamation are part of scheduler correctness.

Iteration-level scheduling can:

- Keep the GPU busy more consistently
- Return completed requests’ KV blocks quickly
- Admit new requests without waiting for a static batch to finish
- Improve utilization for heterogeneous conversational workloads

Its advantage is smaller when offline prompts are similar in length, or when a large static batch already has low scheduling overhead. Continuous batching does not guarantee fairness. It also does not guarantee bounded latency. If arrival rate remains above service rate, the waiting queue grows without bound. A serving system still needs admission control, rate limiting, and queue limits. These should generally exist before GPU-internal preemption becomes the only overload mechanism.

vLLM V1 represents both prefill and decode as tokens that still need computation. Let the scheduler’s per-iteration token budget be $\(C\)$. If request $\(i\)$ receives $\(c_i\)$ scheduled tokens, then:

$$
\sum_i c_i \leq C.
$$

The budget is also constrained by:

- `max_num_seqs`
- Available KV blocks
- Maximum supported model length
- Encoder budget
- Feature-specific limits

Decode requests commonly receive their required tokens first. The remaining budget is then used for prefill. If a prompt’s remaining tokens exceed the available budget, only one chunk is processed. This prevents a long prefill request from occupying the entire iteration.

The same resource model can represent chunked prefill, prefix caching, speculative decoding, encoder tokens, draft tokens, and recomputed tokens. The scheduler ultimately asks how many tokens must be executed this iteration, and how many new KV slots will those tokens require. This is simpler than maintaining separate and incompatible batching rules for every feature.

A larger budget C may provide larger GEMMs, better prefill throughput, higher GPU utilization, and faster completion of long prompts. Its costs may include longer iteration time, higher ITL, and longer wait until the next decode iteration. A smaller budget may provide shorter iterations and better streaming responsiveness. Its costs may include batches too small for efficient GPU use and worse TTFT for long prompts. Therefore, max_num_batched_tokens is fundamentally a trade-off between per-iteration compute efficiency and scheduler response granularity. It is not a workload-independent optimum.

There are several important control parameters. Typical benefit when increasing max_num_batched_tokens include prefill throughput and GPU utilization. Main costs include longer iterations and possibly worse ITL. Observe iteration time and TTFT and ITL when tuning it. Typical benefit when increasing max_num_seqs include decode concurrency and decode throughput. Main costs include KV pressure, scheduling metadata, and preemption. Obser cache usage and prermption count when tuning it. Typical benefit when increasing gpu_memory_utilization is more allocatable KV blocks. Main cost is less headroom for other buffers. Observe graph pool, workspace and OOM behavior when tuning it. For block size, typical benefit when increased include better large-block access and better metadata efficiency. Main costs are more tail waste and coarser prefix reuse. Observe valid-token ratio and kernel time. These controls are not independent. For example, increasing sequence count without enlarging the KV pool may increase preemption. Increasing token budget beyond CPU metadata-building capacity may leave the GPU idle. Quantized weights do not make the KV token budget unbounded because decode still reads more KV as context grows.

The budget/latency trade-off should be tested separately on workloads such as short prompt, long output; long prompt, short output; and mixed traffic. Plot goodput against token budget. A value that is best for one fixed sequence length may perform poorly on a mixed production workload.

Chunked prefill divides the uncomputed part of a long prompt into multiple pieces. Suppose:

- P prompt tokens remain
- r tokens of budget remain after decode scheduling

Then the current iteration computes min(P,r) prompt tokens. The new KV values are stored, and the request continues in a later iteration. This prevents one long prompt from becoming an indivisible task. It also allows decode traffic to keep advancing between prefill chunks.

Chunks cannot be processed arbitrarily. A later chunk depends on the causal KV state produced by all previous chunks. Therefore, chunks must advance in order. Already computed prefixes do not need to be recomputed. After each chunk, the following must remain consistent: Computed-token count, block table, and tail-block filled count. If prefix caching already covers H tokens, prefill resumes from the first uncached complete-block boundary. If the hit covers only part of a block, the implementation’s full-block reuse rule must still be respected.

When a running request needs more blocks but the free queue cannot satisfy the allocation, the scheduler must release resources. A common V1 policy is recomputation:

1. Pause selected requests.
2. Release their KV blocks.
3. When rescheduled, combine original prompt and already generated tokens.
4. Re-run prefill to rebuild their KV cache.

This trades additional FLOPs for memory. It may be easier to control than repeatedly moving large KV tensors between CPU and GPU, but it can create TTFT or ITL tail spikes.

Preemption should not be treated as the normal throughput strategy. A small, occasional amount may come from traffic variation. Frequent preemption suggests a mismatch among:

- Active-token population
- Block size
- Concurrency cap
- KV-pool configuration

It can create hidden waste: Many tokens are computed; Their KV state is discarded; They are later recomputed; And user-visible progress remains small.

When choosing a preemption remedy, the response should depend on the underlying cause. If KV cache stays near full, possible actions include increasing KV capacity, reducing max_num_seqs, using a smaller KV data type, shortening allowed sequence length, and adding a data-parallel replica. If long prefill causes decode jitter, possible actions include lowering the token budget and separating workload classes. If rare giant requests cause sudden jumps, possible action is bucket requests by prompt length at admission. Blindly increasing GPU-memory utilization may reduce memory available for CUDA Graphs, attention workspace, and NCCL buffers, and can turn prermption into OOM.

Reasonable chunking can control iteration duration while preserving correctness of long prompts. It also creates a natural interface between prefill producers of KV state and decode consumers of KV state. The SLOs for first token and streamed tokens can then be tuned separately. For example, if a 16K-token prefill monopolizes one iteration while many short decode requests are active, ITL for those short requests may spike. Splitting the prefill into chunks lets decode progress between chunks. The cost is that the long prompt takes more iterations to finish, so its TTFT may rise. The token budget should be selected using P99 measurements for both workload classes, not only total tokens per second.

Behavior from older vLLM versions should not be copied directly into V1. Preemption mode can depend on version, backend, and enabled features set. Diagnosis should use actual logs such as preemption mode, number of recomputed tokens, and KV-transfer volume. If a request merely appears slower but recomputation is not measured, one may incorrectly blame model kernels for preemption overhead.

Automatic Prefix Caching reuses KV cache when multiple requests have the same token prefix under the same conditions. The matching prefix must correspond to the same model, positional semantics, and relevant extra conditions. Prefix caching avoids repeating prefill work for the shared prefix. It reduces repeated prefill computation and corresponding TTFT. It does not reduce decode’s cost of reading the reused historical KV.

Token text alone is not a safe cache key. The same block of tokens can appear after different earlier prefixes. In that case, its hidden state and KV values differ. Therefore, a safe chained hash should include parent block hash, current block token IDs, all extra values affecting KV, and optional salt. This can be expressed as 

$$
h_j =
H
\left(
h_{j-1},
\text{tokens}_j,
\text{extras}_j,
\text{salt}
\right).
$$

Possible `extras` include LoRA adapter ID, multimodal input hash, model namespace, and cache namespace. The parent hash commits the full context from the sequence start to the current block.
 The salt separates tenants or trust domains.

During the prefix-cache hit process, on a lookup:

1. The cache map finds candidate block IDs by hash.
2. The system confirms the blocks have not been overwritten.
3. Their reference counts are incremented.
4. The corresponding block IDs are inserted into the request’s block table.
5. Only the uncached suffix receives token budget.

The scheduler must align cache hits with computation accounting. A hit should reduce number of prompt tokens requiring prefill and required new KV allocations.

Reuse granularity is a full block chain. The reuse unit is a complete block chain from the beginning of the sequence. It is not an arbitrary matching substring. Only fully populated blocks are stable cache entries. A partially filled tail block usually continues to receive writes, so inserting it into the cache map too early would make later tokens change its contents.

The Block Pool maintains several relationships.

- Preallocated block metadata. A fixed metadata list for physical blocks.

- LRU-ordered free-block queue. Blocks with no active reference are arranged for possible reuse or eviction.

- Cache map. Maps hash to block IDs.

- Active request mapping. Maps request ID to block table. When a block is reused, it is removed from the free queue, and its reference count increases. When a request releases it, the reference count falls. If it reaches zero, the block returns to the queue tail. Its hash may remain temporarily. When a new allocation truly needs the block, the system takes it from the queue head and removes the stale cache-map entry. This is delayed eviction.

A hash system must balance speed, collision behavior, canonical serialization and isolation. A fast noncryptographic hash has low CPU overhead, but a collision may result in incorrect KV reuse rather than a normal cache miss. A cryptographic hash costs more but may be more appropriate for multi-tenant environments and untrusted input. Canonical serialization prevents semantically identical values from producing inconsistent hashes due to language differences, type encodings, and byte-order differences. Optimizing only CPU hash cost while ignoring collision and isolation risk hides correctness risk inside a performance optimization.

A cache_salt can ensure that only requests with the same salt share a block chain. This can reduce a timing side channel in which a user infers whether another user’s prefix exists by measuring response time. However, salt does not replace authentication, authorization, GPU-memory cleanup policy, model-instance isolation, LoRA isolation, multimodal-data isolation, and logging controls. Production systems should explicitly define which requests are allowed to share cache state. The goal should not automatically be the highest global cache-hit rate.

Prefix caching is especially useful when requests repeatedly share system prompts, few-shot templates, long-document prefixes, and common instructions. Its benefit decreases when prefixes diverge early because of timestamps, random IDs, different template whitespace and different tokenization. Before optimization, measure reusable-token ratio, full-block hit rate, and number of saved computed tokens, not only the number of requests that recorded a hit.

The scheduler should query already computed blocks before allocating work. Cached blocks are attached to the new request’s block table and their reference counts increase. Only the uncached suffix receives token budget. When free blocks are scarce, cached-but-unreferenced blocks may be evicted. The best retention policy may differ between high-cache-hit workloads and low-cache-hit workloads. A high cache-hit rate does not mean active references may be ignored. A block still referenced by a running request cannot be evicted.

vLLM can be understood as a token-and-memory operating system for LLM inference. Its main loop is:

1. Receive requests.
2. Track their complete lifecycle.
3. Represent pending work as tokens.
4. Allocate KV state in fixed-size physical blocks.
5. Build a block table for every sequence.
6. Reconstruct a runnable batch every iteration.
7. Execute prefill and decode according to one token budget.
8. Stream results.
9. Reclaim or cache blocks.
10. Repeat.

Its main ideas are:

- Continuous batching: rebuild the active batch every engine iteration.
- Unified token budgeting: schedule prefill, decode, chunked prefill, and related work using one resource model.
- PagedAttention: separate logical token order from physical KV-cache placement.
- Chunked prefill: prevent long prompts from monopolizing an iteration.
- Preemption: recover capacity when KV blocks are insufficient.
- Prefix caching: reuse complete block chains for repeated prefixes.
- Process separation: scale request handling, scheduling, and GPU execution independently.
- SLO-aware tuning: optimize TTFT, ITL, tail latency, and goodput rather than unconstrained peak throughput.


The central principle is, vLLM does not treat serving as repeatedly running a static batch. It continuously schedules token work and KV-cache blocks under changing latency, memory, and concurrency constraints.

The earlier sections explained how vLLM manages requests, token budgets, KV-cache blocks, scheduling, chunked prefill, preemption, and prefix caching. The remaining pieces describe how vLLM handles multiple candidates that share one prompt, speculative decoding, multi-GPU and multi-replica serving, disaggregated prefill and decode, and production tuning, benchmarking, and observability. A single prompt may generate several candidate sequences. Examples include parallel sampling, beam search, and best-of-n generation. Before the candidates diverge, they have exactly the same token history and therefore exactly the same KV cache. A naive implementation would copy the entire prompt KV cache once for every candidate. If there are many candidates, memory use would grow approximately linearly with the candidate count. PagedAttention avoids this by allowing several logical block tables to point to the same physical KV blocks. Each physical block maintains a reference count indicating how many sequences currently use it.

Suppose several candidate sequences share a common prefix. Their logical block tables can all reference the same physical blocks. As long as those blocks are read-only, no copying is required. The system only needs to create a private copy when one candidate attempts to modify a shared block. This is copy-on-write, or COW.

Let physical block $\(p\)$ have reference count $\(r_p\)$. When a branch wants to write into $\(p\)$:

- $\(r_p > 1\)$. The block is shared. The system:

1. Allocates a new physical block $\(p'\)$.
2. Copies the valid slots from $\(p\)$ into $\(p'\)$.
3. Changes that branch’s block-table entry from $\(p\)$ to $\(p'\)$.
4. Decrements the old block’s reference count:

$$
r_p \leftarrow r_p - 1.
$$

5. Writes the new token into $\(p'\)$.

- $\(r_p = 1\)$. The branch is the only owner. It can write into the block in place. The copy is limited to one KV block, not the entire sequence prefix.

Copy-on-write transforms “sharing until divergence” into block-table updates, reference-count updates, and at most one block copy at the divergence point. The memory cost of a sampling tree is therefore closer to one copy of the shared prefix blocks plus private blocks after branches diverge. It is not one complete prompt cache per candidate. This resembles operating-system fork, where several processes initially share physical pages and copy a page only when one process modifies it. In vLLM, however, the copy unit is a KV block.

Beam search repeatedly expands candidate sequences, keeps the best candidates, and prunes less promising candidates. Copy-on-write helps at each step. When a candidate is pruned, its private blocks can be released immediately. Shared blocks only have their reference counts decremented. When a new candidate is created from an existing branch, it inherits the parent’s block table. Reference counts on inherited blocks increase. Only the final shared block may need copying when the child writes a different token. Therefore, beam candidates can share most of their history even while the beam structure changes over time.

Copy-on-write is most beneficial when the prompt is long, many candidates are generated, candidate outputs share a long prefix before diverging, and beam search retains related branches for several steps. Its benefits are smaller when candidates diverge almost immediately, the block size is large relative to the shared region, every output is very short, and the number of candidates is small. Copy-on-write only reduces memory duplication. It does not eliminate the model computation required to decode a different next token for each candidate. Parallel sampling still requires model execution for every active candidate.

Reference counts are not merely a memory optimization. They define whether a block may be modified in place, copied, released, or reassigned. Several errors are dangerous. 

- Releasing too early. If a reference count reaches zero while another branch still uses the block, that block may be reallocated and overwritten.

- Leaking a reference. If a finished branch does not decrement its references, blocks remain permanently unavailable.

- Incorrect tail metadata. If the tail block’s filled count or COW order is wrong, one branch may overwrite another branch’s data.

- Asynchronous execution hazards. A block cannot be reclaimed merely because the Python request object says the branch has ended. The system must wait until the GPU has finished all reads involving that block.

Autoregressive decoding usually requires one serial target-model step for each generated token. Speculative decoding tries to reduce the number of target-model serial steps. The idea is:

1. A proposal mechanism suggests several candidate tokens.

2. The target model verifies those candidates in one parallel forward pass.

3. The system accepts the longest valid prefix.

4. At the first rejected position, generation resumes according to the corrected target distribution.

The acceleration comes from accepting several tokens during one target-model verification step. It does not come from skipping target-model verification. Speculative decoding can remain lossless if the acceptance rule is correct, rejection sampling is correctly implemented, or continuation after rejection follows the target distribution. Therefore, a correct speculative decoder can preserve the output distribution of ordinary target-model sampling. The speculative model proposes. The target model remains the authority.

Suppose:

- The proposer suggests $\(k\)$ tokens per round.
- An average of $\(a\)$ proposal tokens are accepted.
- Drafting takes time $\(T_d\)$.
- Target verification takes time $\(T_v\)$.

A rough efficiency indicator is:

$$
\eta \approx \frac{a+1}{T_d+T_v}.
$$

The extra $\(1\)$ represents the token produced by the target model after:

- A rejection boundary, or
- Acceptance of the entire proposal

depending on the exact protocol. This is only a rough indicator because implementations differ.

Increasing k, the number of proposed tokens, may increase the number of tokens accepted per verification step. However, a larger k also increases draft cost, verification-batch size, wasted work when acceptance is low, and temporary token and KV requirements. Therefore, a larger proposal length is not automatically better. The correct value depends on acceptance probability, draft latency, verification efficiency, memory overhead, and request concurrency. Acceptance is often lower when the target model is difficult to predict, sampling temperature is high, sampling constraints are complex, or when the proposer differs substantially from the target model. Low acceptance can turn proposal generation into pure overhead. A system should not enable speculative decoding merely because the feature exists.

Several speculative strategies are possible.

- Draft model. A smaller auxiliary model proposes tokens. Extra states include additional model weights, draft KV cache, or draft scheduling state. The draft model must be substantially faster than the target while remaining highly aligned with it. Drafting cost may cancel the benefit from accepted tokens. The main benefit condition is the target architecture supports the method and acceptance is high. The main risk is that compatibility and deployment complexity.

- EAGLE, MTP, or related auxiliary modules. These methods use auxiliary heads, additional prediction modules, and architecture-specific multi-token prediction. Extra state is a specialized head or auxiliary model component. Main

- N-gram or suffix matching. This method proposes tokens from repeated text patterns without loading a full draft model. The extra state is token-matching structures. The main benefit condition is that the workload contains repeated content, such as source code, long documents, and repeated templates. The main risk is that it cannot predict genuinely new content, so benefits are limited when repetition is low.

When choosing a speculative method, the following should be considered:

- Acceptance rate
- Draft latency
- Additional GPU memory
- Additional KV state
- Target verification efficiency
- Architecture compatibility
- Sampling support
- Workload repetition

A more accurate proposer may still be slower overall if its own cost is high. A very cheap proposer may still be useless if acceptance is too low.

The vLLM scheduler must budget for both proposal work and verification work. It must also ensure that:

- Accepted tokens are committed to the target sequence’s KV cache.
- Rejected proposals do not contaminate valid target KV blocks.
- Multi-token acceptance across a block boundary updates the tail-block filled count correctly.
- Temporary proposal state is reclaimed safely.

When speculative decoding is combined with prefix caching, LoRA, structured output, tensor parallelism, and sampling constraints, the compatibility matrix becomes more complex than ordinary greedy decoding.

Three important optimizations in vLLM target different bottlenecks. Speculative decoding reduces serial target-model decode depth. PagedAttention improves KV address management and capacity utilization. Prefix caching avoids repeated prefill work. They can be enabled together, but their benefits do not multiply automatically. For example, if the target model is already saturated by a large decode batch, verification may add more work per iteration without increasing overall throughput. Speculative decoding is often more useful for reducing single-request latency at low or moderate concurrency.

A complete evaluation should report:

- Average accepted length
- Acceptance-rate distribution
- Number of target-model steps
- Draft time
- Verification time
- ITL percentiles
- Total goodput
- Additional memory use

This distinguishes a real reduction in serial target steps from a lucky cache or short-example result. Do not assume speculative decoding is always faster. Low acceptance, high concurrency, unsupported sampling behavior, or limited memory can make proposal generation pure overhead. A production A/B test should preserve:

- Input distribution
- Output-length distribution
- Stop conditions
- logprobs
- Random-seed semantics
- Sampling configuration

When one GPU cannot hold the model, the model must first be distributed using model parallelism. When one replica already satisfies latency requirements but lacks aggregate throughput, more independent data-parallel replicas can be added. The three primary parallelism dimensions are tensor parallelism, pipeline parallelism, and data parallelism. Tensor parallelism divides each layer’s weights and matrix operations across several GPUs. Attention and MLP boundaries usually require collective communication. Therefore, collective latency enters the critical path of each generated token. Tensor parallelism is useful when a model cannot fit on one GPU, when multiple GPUs have high-speed interconnects, or when the reduction in per-GPU model state justifies the communication. Excessive tensor parallelism can increase single-request latency because every layer invokes distributed communication. Pipeline parallelism divides model layers into stages. Activations move from one stage to the next through point-to-point communication. It can be useful when a model must span several nodes, when tensor parallel groups should remain within high-bandwidth domains, or when the model is too deep or large for one tensor-parallel group. Its major online-serving drawback is pipeline bubbles, especially with small or irregular batches. Pipeline stages should be balanced using more than layer count. Balance should account for compute per layer, KV-cache occupancy, special modules, communication, or memory use. Data parallelism creates complete executable replicas. Each replica handles different requests independently. Its primary purpose in serving is to increase aggregate request throughput. Unlike training data parallelism, there is no per-step gradient synchronization; Replicas do not need AllReduce for model updates; and the main problem is request routing and load balancing. Data parallelism is therefore outside the critical path of one request, while tensor and pipeline parallelism are inside it. Let:

- $\(G\)$ be the total GPU count.
- $\(D\)$ be the data-parallel degree.
- $\(TP\)$ be the tensor-parallel degree.
- $\(PP\)$ be the pipeline-parallel degree.

A common dense deployment satisfies:

$$
G = D \times TP \times PP.
$$

A useful planning procedure is:

1. Choose the smallest $\(TP \times PP\)$ that can hold:

   - Model weights
   - KV cache
   - Required workspace

2. Use the remaining GPUs to increase $\(D\)$.

This is not an absolute rule. Other model types may require:

- Context parallelism
- Expert parallelism
- Data-parallel attention

But it prevents a common mistake: Increasing tensor parallelism merely to occupy more GPUs, even though the model already fits, can place unnecessary collectives in every token’s latency path.

Tensor-parallel groups should ideally remain inside a high-bandwidth domain such as NVLink and NVSwitch. Cross-node scaling may be more suitable for pipeline parallelism and data parallelism. For mixture-of-experts models, attention and routed experts may use different parallel mappings. For example:

- Attention may be data-parallel.
- Experts may use expert or tensor parallelism.
- Token dispatch and combination may require All-to-All communication.

In such systems, rank placement and network topology directly affect tail latency. Serving expansion should distinguish:

- Enlarging one request’s execution domain. Tensor and pipeline parallelism do this. They allow one request to use several GPUs. Their communication lies on the request’s critical path.

- Adding independent request execution domains. Data parallelism does this. It creates more replicas that can process different requests concurrently. A router assigns requests to replicas. When one GPU can already hold the model, low-latency serving often prefers smaller model-parallel degree or larger data-parallel degree. When the model cannot fit on one GPU, use only the model-parallel degree necessary to make it executable.

An internal data-parallel load balancer may inspect running-request count, waiting-request count, or rank health. However, a short queue does not necessarily mean more free KV capacity, better prefix-cache locality, or less expensive requests. A simple external round-robin policy may distribute identical prefixes across several replicas and reduce cache locality. More advanced production routing may consider request length, model or LoRA identity, prefix affinity, KV-cache water level, or replica health. But increasingly complex routing requires more state synchronization, failure handling, and operational complexity.

A useful production design treats each data-parallel replica as a clear capacity unit, failure domain, and scaling unit. This allows replicas to be scaled independently, drained independently, upgraded gradually, and dedicated to long or short workloads. Admission control can then limit active tokens per rank. API-server processes may also scale independently to handle more HTTP connections, tokenization work, or streaming sessions. The multiprocess V1 architecture helps separate these scaling dimensions.

Parameters such as max_num_seqs commonly apply per data-parallel rank, not to the deployment as a whole. Treating a per-rank limit as a global limit can make total capacity grow unexpectedly with D. For example, if each of D ranks allows S active sequences, deployment-wide capacity may approach D×S. That affects KV demand, CPU load, API connections, and queueing. Capacity documentation should explicitly distinguish per rank, per replica, and deployment-wide limits.

In a mixed deployment, prefill and decode use the same GPUs. This simplifies scheduling and avoids cross-instance KV transfer. However, the two phases compete for different resources:

- Prefill favors large GEMMs and compute throughput
- Decode favors predictable iteration time, high concurrency, and memory bandwidth

Disaggregated prefill places these phases in different vLLM instances. A typical workflow is:

1. A request is sent to a prefill instance.
2. The prefill instance processes the prompt.
3. It generates prompt KV cache and potentially the first token
4. KV block metadata and KV data are transferred to a decode instance.
5. The decode instance maps transferred blocks into its local block table.
6. Decode continues from the already computed token position.

The proxy or orchestration layer must preserve request ID, sampling parameters, streaming connection, cancellation state, and retry state.

The essential handoff is not merely forwarding text. It is transferring multi-layer key/value blocks. After prefill, a connector exposes the KV data through a channel discoverable by the decode side. The decode instance uses a lookup key to locate the corresponding blocks, acquires the KV data, maps it into its local block table, and resumes generation after the already computed positions. The transport abstraction may involve components such as connector, lookup buffer, and pipe. The physical implementation may use GPU RDMA, host memory, shared cache, and external KV systems.

Let:

- $\(T_p\)$ be the prefill computation time.
- $\(T_x\)$ be the KV-transfer time.
- $\(T_d\)$ be the first-token preparation time on the decode side.
- $\(T_{\text{queue},p}\)$ be the prefill-side queueing time.
- $\(T_{\text{queue},d}\)$ be the decode-side queueing time.

Then first-token latency is approximately:

$$
T_{\text{TTFT}}
\approx
T_{\text{queue},p}
+
T_p
+
T_x
+
T_{\text{queue},d}
+
T_d.
$$

Disaggregation is useful only when the benefits from:

- Resource isolation
- Better queueing behavior
- Better phase-specific tuning

are greater than:

- KV-transfer cost
- Two-queue overhead
- Additional failure-handling overhead

Long prompts increase both prefill computation and KV-transfer volume. Therefore, high-speed interconnects and batched transfer matter. Prefix-cache locality also becomes important. If transferred KV first lands in CPU memory or remote storage, transfer latency may eliminate the benefit of separating the phases.

The primary goal is not automatically higher total throughput. It is to control different resource curves independently prefill pool for TTFT and compute-heavy prompt processing and decode pool for ITL and steady streaming. The prefill pool can be tuned for large GEMMs, long prompts, and large token budgets. The decode pool can be tuned for high sequence concurrency, stable iteration duration, and low ITL. The pools can also scale independently as traffic composition changes.

A production disaggregated system must handle transfer not completed, missing KV blocks, decode-instance failure, client cancellation, timeout, duplicate request IDs, partial transfer, retries, instance restart, and version mismatch. A fallback may recompute prefill on the decode side, but this changes SLO behavior, cost, and load distribution. The system must define whether repeated requests are retried, recomputed, failed, and routed elsewhere. Disaggregation may help when:

- Occasional long prefill requests disrupt decode ITL.
- Prefill and decode require different GPU types.
- They need different parallel configurations.
- The prefill side benefits from concentrated prefix caching or KV offload.
- The decode side needs stable high-concurrency operation.

It can turn one mixed resource curve into two more explicit capacity pools. Disaggregation may not improve performance when:

- KV transfer is slow.
- Both pools are lightly loaded.
- The mixed scheduler already satisfies the SLO.
- Transfer failures or orchestration overhead dominate.
- Prefix locality is lost.
- The additional queueing exceeds the isolation benefit.

It should not be presented as guaranteed throughput improvement because it adds one additional KV movement. Disaggregated prefill may remain experimental depending on:

- vLLM version
- Connector implementation
- Backend
- Feature combination

Before production deployment, validate:

- KV checksum or length
- Timeout cleanup
- Duplicate request IDs
- Partial transfer handling
- Instance restart behavior
- Cross-version compatibility
- Tenant isolation
- Failure semantics

A successful normal-path benchmark is not sufficient.

Let:

- $\(\lambda_p\)$ be incoming prompt tokens per second.
- $\(\lambda_o\)$ be incoming output tokens per second.

The prefill and decode pools should be sized according to the effective token rate each can sustain while satisfying its own SLO. They should not simply be assigned GPUs in proportion to request count. Different workloads may produce very different ratios:

- Reasoning workloads may have

$$
\lambda_o \gg \lambda_p.
$$

- RAG workloads with long documents may have

$$
\lambda_p \gg \lambda_o.
$$

The optimal pool ratio can therefore change by product or workload.

Production tuning should begin with a stable baseline. Change variables in layers.

1. Layer 1: fix model semantics. Keep these constant: Model revision, tokenizer, data type, quantization, maximum model length, sampling behavior, and structured-output behavior.

2. Layer 2: fix workload. Keep these constant: Input-length distribution, output-length distribution, arrival process, concurrency, warm/cold prefix mix, and multimodal ratio.

3. Layer 3: tune the serving system. Then adjust scheduler, KV-pool size, attention backend, CUDA Graph, parallel degrees, and replica count. Otherwise, one speed change may simultaneously come from shorter outputs, higher cache-hit rate, different model behavior, different engine configuration and cannot be attributed correctly.

A useful observability model separates API layer, scheduler, model runner, GPU kernel, KV manager and network and distributed runtime. API layer observes request queue, tokenization time, streaming backpressure; Scheduler observes waiting requests, running requests, scheduled tokens, and preemption; GPU execution observes iteration time, prefill-token count, decode-token count, kernel time, SM utilization and memory utilization; KV manager observes cache usage, allocated blocks, free blocks, cached blocks, prefix-cache hits, and KV transfers; Distributed layer observes collective time, ZMQ queues, data-parallel imbalance, and network retransmission. Only by connecting internal token/block metrics to user-visible SLOs can one explain why a service is slow. “GPU is busy” is not enough.

Symptom-based diagnosis:

1. High TTFT tail. First inspect waiting time, number of prefill tokens, and API CPU load. Possible actoins include chunk long prompts, add prefill capacity, add data-parallel replicas, and apply entry-layer rate limiting. Do not assume that increasing decode concurrency will help.

2. ITL jitter. First inspect engine iteration time, preemption, and prefill interference. Possible actions include lower the token budget, separate workload pools, and increase KV headroom. Average TPOT may still look normal while P99 ITL is poor.

3. Low GPU utilization. First inspect CPU bottlenecks, tokens per batch, kernel-launch overhead, and network behavior. Possible actions include form larger useful batches, use CUDA Graph, and fix entry-layer bottlenecks. High memory usage does not imply high compute utilization.

4. High throughput but low goodput. First inspect P99 latency, timeouts, cancellations, and recomputation count. Possible actions include reducing admission and separating requests by length. Peak tokens per second do not equal serving capacity.

5. Frequent out-of-memory errors. First inspect separate peaks for KV cache, CUDA Graph pools, workspace, and other buffers. Possible actions include leaving more headroom, reducing concurrency, and changing KV data type. Do not simply increase gpu_memory_utilization.

CUDA Graph can reduce CPU launch overhead for repeated execution shapes. Its costs include graph memory pools, captured-shape management, and fallback handling for dynamic shapes. torch.compile and fused kernels may reduce framework overhead. Their costs may include first-run compilation, cache complexity, and more difficult debugging. These optimizations must be tested using the actual request-shape distribution.

Weight quantization may reduce model-weight memory and weight bandwidth. KV quantization directly affects bytes per cached token and active-token capacity. However, every combination must be tested independently for accuracy, backend support, sequentization cost and kernel efficiency. Features such as LoRA, multimodal encoders, and structured output may change which requests can be batched together. A baseline measured without these features may not remain valid after they are enabled.

The most effective tuning often comes from removing structural mismatch. Examples include:

- Separating short requests from extremely long requests
- Keeping tensor parallelism inside high-speed interconnect domains
- Using prefix caching for stable templates
- Adding data-parallel replicas when decode is overloaded
- Avoiding one mixed instance for fundamentally different workloads

This is usually more effective than endlessly increasing unrelated parameters on one mixed deployment. 

A disciplined benchmark can proceed in four stages.

1. Stage 1: single-request baselines. Measure combinations of short and long prefill, and short and long decode. This identifies model and kernel lower bounds.

2. Stage 2: fixed-concurrency sweep. Find batch-efficiency gains and KV-capacity limits.

3. Stage 3: arrival-rate sweep. Increase arrival rate until the SLO first fails. This identifies the saturation point.

4. Stage 4: realistic production behavior. Add real prefix distribution, sampling, cancellations, failures, and timeouts. At every stage, record request throughput, token throughput, TTFT percentiles, ITL percentiles, cache usage, prermption, and error rate.

A warm benchmark is not equivalent to cold-start capacity. Warm execution may benefit from model already loaded, CUDA Graph already captured, compilation already complete, prefix cache prewarmed, and file cache populated. Real traffic may also have long-tailed sequence lengths, cancellations, timeouts, and different cache-hit behavior. Capacity commitments should therefore include stable-period measurements for both cold path and warm path. Do not use synthetic equal-length requests as the only production-capacity estimate.

Correctness is part of production evaluation. Serving optimization must preserve semantics. Validate behavior for:

- Greedy generation
- Sampling
- Stop tokens
- logprobs
- Random seeds
- Prefix caching enabled and disabled
- Speculative decoding enabled and disabled
- Single-GPU and multi-GPU execution
- Preemption and recovery
- Connector fallback

A speed improvement that changes the intended output distribution is not acceptable. The final requirement is to preserve model semantics within the service-level objective. Throughput, latency, memory, and feature compatibility must all be evaluated around this condition.

A usable production recipe should specify resource topology, per-rank parameters, input- and output-length boundaries, request-arrival assumptions, expected TTFT and ITL, alert thresholds, overload behavior, rollback conditions, and version pinning. It should also distinguish per-rank capacity, per-replica capacity, and deployment-wide capacity. This makes the configuration a verifiable engineering contract rather than a disconnected collection of startup flags.

vLLM evolves quickly. The following may change across versions: 

- Attention backend
- Scheduler behavior
- Feature combinations
- Default parameter values
- Preemption policy
- Connector behavior

An upgrade should be treated as a combined performance change and correctness change. A safe rollout process should:

1. Replay shadow traffic.
2. Compare user-visible output semantics.
3. Compare cloud metrics and latency distributions.
4. Compare memory peaks.
5. Increase replica traffic gradually.

It is not enough to confirm that the new process starts successfully.

#### 5.2.3.3 TGI

TGI (Text Generation Inference) is a specialized, high-performance inference framework developed by HuggingFace (HF). It is specifically designed as the primary deployment mode for the online inference of models hosted on the HuggingFace platform. The architecture is split into request handling and parallel model execution:

- The Web Server (Request Handling). The system features a Web Server that acts as the entry point. It receives multiple, simultaneous user requests, denoted by the /generate endpoints.To ensure hardware is used efficiently, TGI does not process these requests one by one. Instead, the incoming /generate requests are collected in a Buffer. They are then passed to a Batcher, which groups the requests together to be processed simultaneously, maximizing throughput.

- Distributed Execution (Model Shards). Once the requests are batched, they are transmitted via gRPC (a high-performance Remote Procedure Call framework) to the actual model processing units.  The LLM is divided into multiple Model Shards (represented by the green boxes featuring Python and Rust logos). This means the massive neural network is split across different processing units to handle the heavy computation. To work together, these separated Model Shards must constantly communicate. They do this using NCCL (or rccl, etc.), which are standard communication libraries designed to synchronize data rapidly across multiple GPUs.

- Hardware Compatibility. TGI is versatile and supports a wide variety of hardware accelerators for these Model Shards, including NVIDIA GPUs, AMD GPUs, Inferentia2, and Gaudi2.

TGI is built for seamless deployment. All dependencies are installed within a Docker container. This means developers do not have to worry about complex local environment setups or conflicting libraries; the entire framework can be spun up reliably anywhere Docker is supported. To handle massive LLMs that cannot fit on a single GPU, TGI natively uses Tensor Parallelism. This technique splits the model's massive mathematical computations (tensors) across multiple devices. This is what enables the fast, distributed execution of huge models. Because TGI is developed directly by HuggingFace, it offers the absolute best compatibility for models published on the HF platform. If a model is hosted on HuggingFace, TGI is the most optimized and stable deployment mode for running it online.

#### 5.2.3.4 Ray Serve

Ray Serve is a powerful and flexible deployment tool designed specifically for machine learning models. Its most significant architectural advantage is that "Serve is framework-agnostic". This means it is not locked into a specific ecosystem like TensorFlow or PyTorch. As a result, developers can use this single toolkit to serve all aspects of their deep learning models, regardless of how those models were originally built. It supports Dynamic Batching. his feature is crucial when the cost of using the model is very high (e.g., massive Large Language Models). To ensure that the expensive hardware (like GPUs) is utilized to its maximum potential, Ray Serve can adopt this strategy. Instead of processing incoming user requests one by one, it dynamically groups them together into batches on the fly, dramatically increasing throughput. Ray Serve benefits directly from the underlying elastic nature of the broader Ray ecosystem. Users can utilize the Ray Dashboard to gain full visibility and obtain the real-time status of both the overall Ray cluster and the specific Ray Serve applications running on it. To handle sudden spikes or drops in user traffic, Ray Serve features intelligent auto-scaling. It does this by constantly "observing the queue size" of incoming requests. Based on this real-time data, it automatically makes scaling decisions to either add new replicas (to handle traffic spikes) or remove unneeded replicas (to save resources when traffic drops).

### 5.2.4 Repetitiveness

During the inference phase of Large Language Models (LLMs) like GPT-4, a common and frustrating issue is the generation of repetitive text. This repetitiveness not only degrades the overall quality of the output but also significantly lowers the user experience. There are four primary categories that cause a model to loop into repetitive patterns.

1. Language models learn by analyzing patterns in training data to generate a probability distribution for the next possible word. However, this statistical approach has inherent flaws:

- High-Frequency Vocabulary Over-weighting: Models tend to assign disproportionately high weights to extremely common words. Because their baseline probability is so high, they are prone to appearing repeatedly during generation.
- Local Dependency Problems: Models tend to generate words based heavily on the immediate local context. If the current local context already contains repetitive information that is assigned a high weight, the model is highly likely to continue that repetitive trend.
- Lack of Long-Range Planning: Because generation is done token-by-token without a holistic structural plan for the entire text, the model can easily fall into generating similar, looping content when focusing only on local text chunks.

2. The Impact of Decoding Strategies. How we ask the model to pick the next word drastically affects repetitiveness:

- Greedy Search: This strategy always picks the single word with the absolute highest probability at every step. This forces the model to constantly choose the "safest" and most common words, inevitably leading to repetitive loops.
- Beam Search: While Beam Search tracks multiple candidate sequences simultaneously to find the best overall path, if it lacks a diversity penalty mechanism, the multiple paths tend to converge, leading to the repeated generation of similar sentences.
- Lack of Entropy Regularization: Without entropy regularization, the model's output probability distribution can become too sharp (concentrated on just a few high-probability words), leading directly to repetition.

3. Training Data Issues. A model is only as good as its data.

- Presence of Repetitive Samples: If the training data contains massive amounts of repetitive sentences or fragments, the model will learn that repetition is a valid and correct language structure.
- Corpus Imbalance: If certain topics or specific language structures represent too high a proportion of the dataset, the model will be heavily biased toward generating those specific patterns.
- Low-Quality Data Impact: Training data containing noise or artificial/spam repetition will actively mislead the model during generation.

4. Model's Local Optima. During generation, the model can get trapped in a mathematical local optimum.

- High-Probability Path Trap: Once a specific generation path forms a familiar pattern, the model might endlessly repeat it because it cannot mathematically "see" a better, global path to switch to.
- Lack of Diversity Incentives: If the generation process does not actively reward diversity, the model will naturally default to safe, repetitive choices.

To break these repetitive loops, engineers use several dynamic tuning and algorithmic penalties during the decoding phase.

1. Adjusting the Temperature Parameter. The Temperature parameter ($T$) controls the smoothness of the model's generation probability distribution. High Temperature smooths out the probability distribution, artificially increasing the chances of picking less common words, thereby boosting diversity. (e.g., At $T=1.2$, the model generates more uncertain but interesting content). Caution: Setting the temperature too high will result in incoherent text. Low Temperature ($T < 1$) sharpens the distribution, making the text highly deterministic but extremely prone to repetition. Low temperature is suitable for task-based generation like Q&A, not open-ended generation. For strict tasks (like translation), set $T = 0.7$. For creative tasks (like story generation), try $T = 1.2$.

2. Employing Sampling Strategies. Instead of always picking the top word (Greedy Search), we sample from a pool of likely words. Top-$k$ Sampling and Nucleus Sampling / Top-$p$ Sampling work. In Top-$p$ Sampling, the common range is $p = 0.8 \sim 0.95$. It is recommended to prioritize Nucleus Sampling when generating long paragraphs.

3. Algorithmic Penalties.

- Repetition Penalty. Dynamically lowers the probability of words or phrases that have already been generated. This dynamically controls already-generated high-frequency words (like repeated n-grams), avoiding text verbosity. Define a penalty coefficient $\alpha$ (where $\alpha > 1$). The adjusted probability is calculated as:

$$P_{\text{adjusted}}(w) = \frac{P(w)}{\alpha}$$

- n-gram Blocking. Record all generated n-gram sequences to prevent the model from generating them again. Usually, $n$ is set to $3$ or $4$. Before generating a new word, the system checks if it will create an n-gram that already exists in the output. If a repetition is detected, the system skips that word and chooses the one with the next highest probability.

- Introducing Diversity in Beam Search. Add diversity penalty mechanisms to Beam Search to encourage the generation of different sequences. Add a diversity reward to each beam path to suppress similarity between paths within the same beam. The formula is:

$$P_{\text{adjusted}}(w) = P(w) - \lambda \cdot \text{similarity}(w)$$

(Where $\lambda$ is the weight controlling diversity).

In practical engineering:

- Tune Decoding Parameters: Select appropriate Temperature, Top-$k$, and Top-$p$ parameters based on the specific task. (Example: For summary generation, set $T=0.8, k=50, p=0.9$).

- Use Advanced Decoding Strategies: Prioritize the use of repetition penalties and n-gram blocking.

- Optimize Training Data: Ensure linguistic diversity and strictly reduce artificial repetition through data cleaning (removing highly repetitive sentences and balancing corpus topics).

- Model Fine-Tuning: Fine-tune the model for specific tasks to enhance its generation logic and semantic fluency.

- Introduce Post-Processing: After generation is complete, apply text post-processing algorithms to physically detect and remove repetitive content before showing it to the user.

