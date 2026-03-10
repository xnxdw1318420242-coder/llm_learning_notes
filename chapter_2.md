# 2. Modern Models
## 2.1 Encoder-Only

In the context of Large Language Models—specifically BERT (Bidirectional Encoder Representations from Transformers)—MLM and NSP are the two primary pre-training objectives used to teach the model language and context.

MLM is a "fill-in-the-blank" task. It allows the model to learn a bidirectional representation of a sentence, meaning it looks at both the left and right context of a word simultaneously. To implement Masked Language Modeling (MLM), we treat the problem as a multi-class classification task where the model predicts the identity of "hidden" tokens from the entire vocabulary.

### 2.1.1 MLM & NSP Model

#### 2.1.1.1 BERT
#### 2.1.1.2 RoBERTa
#### 2.1.1.3 ALBERT
#### 2.1.1.4 SpanBERT
#### 2.1.1.5 DeBERTa V1/2
#### 2.1.1.6 DeBERTa V3
#### 2.1.1.7 XLNet

### 2.1.2 Contrastive Learning Model
#### 2.1.2.1 Sentence-BERT

## 2.2 Encoder-Decoder
### 2.2.1 BART
### 2.2.2 T5

## 2.3 Non-Causal Decoder-Only
### 2.3.1 UniLM

## 2.4 Causal Decoder-Only
### 2.4.1 GPT
### 2.4.2 LLaMA
### 2.4.3 DeepSeek
### 2.4.4 Qwen
### 2.4.5 Gemini
