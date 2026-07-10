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
