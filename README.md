### **1. LLM Basics**

**1.1 Tokenization**
* **1.1.1 Word-based tokenization**
* **1.1.2 Character-based tokenization**
* **1.1.3 Subword-based tokenization**
    * **1.1.3.1 BPE (Byte-Pair Encoding)**
    * **1.1.3.2 BBPE (Byte-level BPE)**
    * **1.1.3.3 WordPiece**
    * **1.1.3.4 Unigram**
    * **1.1.3.5 SentencePiece**

**1.2 Embedding**
* **1.2.1 History**
    * **1.2.1.1 One-hot Encoding**
    * **1.2.1.2 Co-occurrence Matrix**
    * **1.2.1.3 Distributed Word Representation**
* **1.2.2 Static Embeddings**
    * **1.2.2.1 Word2Vec**
    * **1.2.2.2 GloVe**
    * **1.2.2.3 FastText**
* **1.2.3 Contextual Embeddings**

**1.3 Positional Encoding**
* **1.3.1 Learnable Positional Embedding**
* **1.3.2 Sinusoidal Positional Encoding**
* **1.3.3 Bucketed Relative Position Bias**
* **1.3.4 ALiBi**
* **1.3.5 RoPE**
* **1.3.6 Length Extrapolation**
    * **1.3.6.1 NTK (Neural Tangent Kernel)**
    * **1.3.6.2 YaRN**
    * **1.3.6.3 Dual-Chunk Attention**
    * **1.3.6.4 Other Methods**
 
**1.4 Attention**
* **1.4.1 Multi-Head Attention (MHA)**
* **1.4.2 Multi-Query Attention (MQA)**
* **1.4.3 Grouped-Query Attention (GQA)**
* **1.4.4 Multi-head Latent Attention (MLA)**
* **1.4.5 Linear Attention**
* **1.4.6 Sparse Attention**
* **1.4.7 Mask**
    * **1.4.7.1 Padding Mask**
    * **1.4.7.2 Casual Mask**
    * **1.4.7.3 MLM (Masked Language Model) Mask**
 
**1.5 FFN (Feed-Forward Network)**
* **1.5.1 ReLU**
* **1.5.2 Tanh**
* **1.5.3 Sigmoid**
* **1.5.4 Leaky ReLU**
* **1.5.5 PReLU**
* **1.5.6 ELU**
* **1.5.7 GELU**
* **1.5.8 Swish**
* **1.5.9 GLU**
* **1.5.10 GeGLU**
* **1.5.11 SwiGLU**

**1.6 Add & Normalization**
* **1.6.1 Residual Connection**
    * **1.6.1.1 Post-Norm**
    * **1.6.1.2 Pre-Norm**
    * **1.6.1.3 Sandwich-Norm**
    * **1.6.1.4 Deep Norm**
* **1.6.2 Normalization**
    * **1.6.2.1 Batch Normalization**
    * **1.6.2.2 Layer Normalization**
    * **1.6.2.3 Instance Normalization**
    * **1.6.2.4 Group Normalization**
    * **1.6.2.5 RMS Norm**
    * **1.6.2.6 pRMS Norm**
    
**1.7 Transformer**
* **1.7.1 Weight Sharing**
* **1.7.2 Parameters & FLOPs**
* **1.7.3 Encoder-Only, Decoder-Only, Encoder-Decoder**
* **1.7.4 Decoding Strategy**
    * **1.7.4.1 Greedy Search**
    * **1.7.4.2 Beam Search**
    * **1.7.4.3 Top-K Sampling**
    * **1.7.4.4 Top-P Sampling**
    * **1.7.4.5 Random Sampling**
    * **1.7.4.6 Best-of-N**
    * **1.7.4.7 Majority Vote & Self-Consistency**
    * **1.7.4.8 Temperature**


### **2. Modern Models**
**2.1 Encoder-Only Transformer**
* **2.1.1 MLM & NSP Model**
    * **2.1.1.1 BERT**
    * **2.1.1.2 RoBERTa**
    * **2.1.1.3 ALBERT**
    * **2.1.1.4 SpanBERT**
    * **2.1.1.5 DeBERTa**
    * **2.1.1.6 XLNet**
* **2.1.2 Contrastive Learning Model**
    * **2.1.2.1 Sentence-BERT**
      
**2.2 Encoder-Decoder**
* **2.2.1 BART**
* **2.2.2 T5**
 
**2.3 Non-Causal Decoder-Only**
* **2.3.1 UniLM**
* **2.3.2 GLM**
 
**2.4 Causal Decoder-Only**
* **2.4.1 GPT**
    * **2.4.1.1 GPT-1**
    * **2.4.1.2 GPT-2**
    * **2.4.1.3 GPT-3**
    * **2.4.1.4 GPT-3.5**
    * **2.4.1.5 GPT-4**
    * **2.4.1.6 OpenAI o1**
    * **2.4.1.7 GPT-OSS**
* **2.4.2 LLaMA**
    * **2.4.2.1 LLaMA-1**
    * **2.4.2.2 LLaMA-2**
    * **2.4.2.3 LLaMA-3**
    * **2.4.2.4 Alpaca**
    * **2.4.2.5 Code LLaMA**
* **2.4.3 DeepSeek**
    * **2.4.3.1 DeepSeek-V1**
    * **2.4.3.2 DeepSeek-V2**
    * **2.4.3.3 DeepSeek-V2.5**
    * **2.4.3.4 DeepSeek-V3**
    * **2.4.3.5 DeepSeek-V3.2-EXP**
    * **2.4.3.6 DeepSeek-R1**
* **2.4.4 Qwen**
    * **2.4.4.1 Qwen1**
    * **2.4.4.2 Qwen1.5**
    * **2.4.4.3 Qwen2**
    * **2.4.4.4 Qwen2.5**
    * **2.4.4.5 Qwen3**
    * **2.4.4.6 Qwen3 Next**
    * **2.4.4.7 GTE-Qwen3**
    * **2.4.4.8 Qwen3 Embedding & Reranker**
* **2.4.5 Gemini**
  
### **3. Pretraining**
**3.1 Data**
* **3.1.1 Data Collection**
* **3.1.3 Data Preprocessing**
    * **3.1.3.1 Data Quality Filtering**
    * **3.1.3.2 Sensitive Content Filtering**
    * **3.1.3.3 Data Deduplication**
    * **3.1.3.4 tokenization**
* **3.1.2 Data Augmentation**
* **3.1.3 Data Scheduling**

**3.2 Training Tasks**
* **3.2.1 Goals**
* **3.2.2 Long Context**

**3.3 Optimizer**
* **3.3.1 Naive SGD**
* **3.3.2 Momentum SGD**
* **3.3.3 NAG**
* **3.3.4 AdaGrad**
* **3.3.5 RMSProp**
* **3.3.6 Adam**
* **3.3.7 AdamW**

### **4. Post-Training**
