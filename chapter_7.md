# 7. LLM Application

## 7.1 Prompt Engineering

When working with Large Language Models (LLMs), the quality of the model's output is directly dictated by the quality of the instructions it receives. At its most basic level, a Prompt is the initial text input that a model receives. It acts as the steering wheel for the AI. As the text states, we give the AI a set of Prompt inputs to guide the model to generate a response to execute a task. A prompt is incredibly flexible. It can be a simple question, a lengthy description, a set of keywords, or any other form of text. The ultimate goal is to guide the model to produce a response that closely matches the user's specific requirements (e.g., asking ChatGPT to answer a question, generate an essay, or write code).

Prompt Engineering, also referred to as "in-context prompting," elevates prompt writing from a casual action to an empirical science. It is the methodology of communicating with an LLM to guide its behavior toward a desired outcome, without needing to update the model weights. Instead of fine-tuning the model's internal brain, you are changing how you talk to it. Prompt Engineering is highly empirical. Prompt Engineering's effects can vary greatly between different models, therefore requiring massive amounts of experimentation and heuristic methods. Even when an algorithm engineer is preparing to do Supervised Fine-Tuning (SFT), they must first prepare a reasonable prompt. There are two highly effective techniques for structuring prompts scientifically.

- Task Breakdown and Verifiable Output. When dealing with complex requests, models often hallucinate or lose focus.  This involves breaking down a complex task into simpler sub-tasks. Once broken down, you must force the model to explicitly answer each part. Verifiable output means having the model output an answer for every single sub-task... thereby guiding the model to think about every sub-problem to reduce hallucinations.

- Structured Input and Output. For highly complex, multi-step logic, natural language paragraphs become messy and confusing for the model. Often, tasks involve branching logic. When workflows have these if-else branches and jumps, it is best to use programming languages or Mermaid to express this workflow, which can increase the model's understanding of the process. To make the prompt perfectly clear, the text suggests using structural markers like triple quotes ("""), XML tags (<step>), or section headers to separate different parts of the prompt structure. Similarly, you should instruct the model to use structural markers when generating its output.

### 7.1.1 Zero-shot Prompting

Zero-shot prompting means asking an AI model to perform a task without giving it any examples of how the task should be done. You provide only the instruction and the input to process. A zero-shot prompt works best when the instruction is clear and specifies the expected output.

### 7.1.2 Few-shot Prompting

Few-shot prompting means giving a language model a small number of high-quality examples before asking it to complete a new task. Each example usually contains an example input and the expected output. The model observes these demonstrations and uses them to infer the task, the desired response format, and the standard for a correct answer. Few-shot prompting is often described as a form of in-context learning. The model does not update its parameters or undergo additional training. Instead, it temporarily learns how to perform the task from the examples and instructions included in the current prompt. The examples can serve two related purposes:

- Demonstration: Show the model how to perform the task.

- Reference: Provide rules, standards, or information that the model should use when answering.

For example, when asking a model to evaluate the quality of a user review, the prompt could include:

- Several reviews with human-generated scores, allowing the model to imitate the scoring pattern

- A written evaluation rubric covering relevance, informativeness, verifiability, readability, and bias

The first approach teaches through examples. The second provides criteria that the model should reference. These ideas also form part of the basic mechanism behind retrieval-augmented generation, or RAG, where relevant information is placed in the prompt so that the model can use it when answering. Large language models are often already capable of completing many tasks with zero-shot prompting. However, for more complex or ambiguous tasks, few-shot prompting can produce better results because the demonstrations clarify:

- what the task really means
- what output format is expected
- what level of detail is appropriate
- how labels or categories should be interpreted
- what style or reasoning pattern should be followed

Compared with zero-shot prompting, few-shot prompting often improves consistency and accuracy.

Few-shot prompting consumes more tokens because the examples must be included in the prompt.

This creates two practical problems:

- It increases input cost and processing time.
- It leaves less context space for long user inputs or long model outputs.

Therefore, examples should be high quality, relevant, and as concise as possible.

The position of demonstrations inside the prompt can affect model performance. For example, in a particular experimental setup, the suggested order is:

- Put few-shot examples at the beginning of the system prompt when the system prompt can be changed.

- If the system prompt is fixed, place the examples near the beginning of the user message.

- Avoid putting them at the very end of the user message when possible, because this placement may lead to lower accuracy and less stable predictions.

For especially important instructions, repeating them can sometimes increase the chance that the model follows them. For example:

Return only valid JSON.
Do not include explanations.
Again, return only valid JSON.

However, repetition should be used carefully. Too much repetition wastes tokens and can make the prompt noisy. 

### 7.1.3 Instruction Prompting

Instruction prompting means directly telling a language model what task to perform and how to perform it. Instead of teaching through examples, you describe the desired behavior using natural-language instructions. Instruction prompting and zero-shot prompting are closely related, but they describe slightly different things. Instruction prompting focuses on giving the model explicit directions. Zero-shot prompting means asking the model to perform a task without providing examples. Instruction prompting can also be combined with few-shot prompting.

A strong instruction prompt usually contains several parts.

1. The task. Clearly state what the model should do. Avoid vague instructions such as: Do something with this paragraph.

2. The input. Clearly separate the material that the model should process. Using labels such as Text, Article, Question, or Data makes the prompt easier to understand.

3. The expected output. Specify what the answer should look like.

4. Constraints. State important limits or rules. Constraints help control length, style, scope, and format.

5. Audience or tone. Tell the model who the response is for. The same subject may need very different explanations depending on the audience.

Language models have learned many patterns for following natural-language directions. A clear instruction narrows the set of possible responses and tells the model what kind of output is useful.

### 7.1.4 CoT

Chain-of-Thought prompting, usually abbreviated as CoT prompting, is a prompting method that encourages a language model to solve a problem through a sequence of intermediate reasoning steps rather than jumping directly from the input to the final answer. CoT prompting is especially useful for tasks that require multiple reasoning steps, such as arithmetic, symbolic reasoning, logic, and some common-sense problems.

With standard prompting, an example may contain only a question and its final answer. The model sees the input and the final output, but not the reasoning connecting them. With CoT prompting, the example also includes intermediate steps:

```python
Question:
Roger has 5 tennis balls. He buys 2 cans of tennis balls.
Each can contains 3 balls. How many balls does he have now?

Reasoning:
Two cans contain 2 × 3 = 6 tennis balls.
Roger started with 5 balls, so he now has 5 + 6 = 11 balls.

Answer:
11
```

When the model later receives a similar problem, it is more likely to organize the solution into several steps. The step-by-step structure helps prevent mistakes such as incorrectly combining all the numbers.

CoT prompting is usually implemented through few-shot learning. The prompt contains a small number of demonstrations. Each demonstration includes a problem, intermediate reasoning steps, and the final answer. The model observes the complete path from input to output and then applies a similar reasoning structure to a new problem. A general format looks like this:

```python
Example 1

Question:
[Example problem]

Reasoning:
[Intermediate steps]

Answer:
[Final answer]

Example 2

Question:
[Example problem]

Reasoning:
[Intermediate steps]

Answer:
[Final answer]

New question:
[Problem to solve]

Reasoning:
```

The important difference is that the demonstrations provide not only the correct outputs but also the process used to produce them. Both CoT prompting and standard few-shot prompting give the model examples, but the examples contain different information. In standard few-shot prompting, the model learns the relationship between a question and its expected output. In chain-of-thought prompting, the model learns a more complete path from the question to the answer. This is especially helpful when the final answer depends on several calculations or logical decisions. CoT prompting can improve results because it encourages the model to break a complex task into smaller and more manageable parts. It can help the model:

- identify the important facts in a problem
- determine the order of operations
- keep track of intermediate results
- organize multi-step reasoning
- reduce errors caused by answering too quickly

The method has shown particular value on tasks where ordinary prompting performs poorly but a structured solution path makes the problem easier. CoT becomes more effective with sufficiently capable language models. Smaller or weaker models may generate reasoning steps that look plausible without actually improving correctness.

A generated chain of reasoning is not necessarily true or logically valid.  A model may make an incorrect assumption, calculate an intermediate value incorrectly, produce a convincing explanation for a wrong answer, or reach the correct answer through faulty reasoning

Therefore, a detailed explanation should not automatically be treated as proof that the answer is correct. Important results should still be checked against facts, calculations, tools, or external evidence. CoT can also be encouraged without providing examples. This is sometimes called zero-shot Chain-of-Thought prompting. Few-shot CoT usually gives the model a clearer pattern to imitate.

Self-consistency with CoT is often written as CoT-SC. Instead of generating only one reasoning path, the model generates several different possible reasoning paths:

```python
Input
  ├── Reasoning path 1 → Answer A
  ├── Reasoning path 2 → Answer A
  └── Reasoning path 3 → Answer B
```

The system then selects the answer supported by the majority of the paths. In this example, Answer A would be selected because two of the three reasoning paths reached it. The intuition is that a single reasoning path may contain an accidental mistake, while multiple independent paths can make the final result more robust. However, self-consistency requires more computation because the model must generate several solutions instead of one.

Tree of Thought explores several possible reasoning branches:

```python
Problem
  ├── Approach A
  │     ├── Next step A1
  │     └── Next step A2
  ├── Approach B
  │     ├── Next step B1
  │     └── Next step B2
  └── Approach C
```

Each node represents a possible intermediate thought or subproblem. The system can evaluate the branches, continue promising ones, and stop exploring unsuitable ones. Therefore:

- CoT develops one reasoning sequence.
- Self-consistent CoT generates several complete reasoning sequences and uses a majority vote.
- ToT actively expands, evaluates, and prunes multiple branches during problem solving.

ToT is more suitable for problems that may require exploring alternative strategies, backtracking, or searching a large solution space.

## 7.2 RAG

Retrieval-Augmented Generation, or RAG, is a method that provides a large language model with information retrieved from an external data source. For a user question, RAG first uses information-retrieval techniques to find relevant content from an external database. It then places the retrieved information into the prompt and sends the complete prompt to the LLM. The LLM generates its answer by combining the user’s original question and the retrieved external information. RAG retrieves relevant knowledge first, then asks the LLM to answer using that knowledge.

An LLM mainly depends on the data used during pretraining. Because of this, it may not have access to the latest or dynamically changing information. RAG addresses this information gap by retrieving relevant knowledge from external databases at inference time. Without RAG, an LLM may be unable to answer questions about recent events because the required information is not included in its pretrained knowledge. With RAG, relevant recent documents can be retrieved and included in the prompt, allowing the LLM to generate a more informed answer. The basic process can be summarized as:

1. User query. 

2. Retrieve relevant information from an external database. 

3. Combine the query and retrieved information into a prompt.

4. Send the complete prompt to the LLM.

5. Generate a response.

The original question is sent in two directions:

- It is used to search the external data source.

- It is also provided to the LLM as part of the final prompt.

The retrieved documents provide additional context for generation.

Main stages of a RAG system include the following:

1. User query. The process begins when the user asks a question. The query represents the information need that the system must answer.

2. External data source. RAG connects the query to an external datastore containing documents or other knowledge. This datastore provides information beyond what is already stored in the model’s pretrained parameters.

3. Indexing. Before retrieval, documents are processed and indexed. Documents are divided into chunks and represented in a form that supports retrieval.

4. The system uses the user’s query to search the indexed data and identify relevant documents or chunks. The retrieved results should contain information related to the user’s question.

5. The retrieved documents are combined with the original question. This produces a more complete prompt containing both the task and the supporting context.

6. Generation. The combined prompt is sent to the LLM. The model then generates an answer using the retrieved information together with the original question.

Traditional search engines such as Google or Bing mainly retrieve information. They have retrieval ability, but they do not directly generate a complete answer in the same way as an LLM. A pretrained LLM stores large amounts of knowledge in its model parameters. This gives the model memory, but it is limited to the information learned during training. RAG combines retrieval with the LLM’s generation ability. From this perspective, RAG sits between traditional search and a memory-based language model. The retrieved information is loaded into the LLM’s working memory. In this setting, working memory refers to the model’s context window: the maximum amount of text that the model can receive during one generation process. The context window may contain the user’s question, the retrieved documents, prompt instructions, and other relevant context. The model uses all of this information together when generating the answer.

RAG relies on prompt construction. The system does not simply retrieve documents and return them directly. Instead, it places the relevant documents into a prompt together with the user’s question. Therefore, the generation stage is based on a prompt containing external knowledge. This allows the LLM to generate an answer based on more complete information.

The development of RAG is described in three stages.

1. Early stage. The emergence of RAG was closely connected to the rise of the Transformer architecture. The main goal was to introduce external knowledge into pretrained models in order to enhance language models. Early research focused on foundational improvements to pretraining methods.

2. Shift after ChatGPT. The appearance of ChatGPT became an important turning point. Large language models demonstrated strong in-context learning abilities. As a result, RAG research increasingly focused on supplying better information to LLMs during inference. This was especially useful for more complex and knowledge-intensive tasks.

3. Later development. As research continued, RAG optimization was no longer limited to the inference stage. Researchers also began combining RAG with LLM fine-tuning techniques.

Retrieval-Augmented Generation (RAG) is not a single, monolithic technique. Depending on how the retriever augments the generator, RAG paradigms can be classified into four distinct categories. 

1. Query-based RAG. Inspired by the concept of prompt augmentation, this is the most common and intuitive form of RAG. It seamlessly integrates the user's question with the retrieved information and directly inputs it into the generator's initial stage. After retrieval is complete, the acquired content is merged with the user's original query to form a composite input, which is then processed by the generator to produce a response. It provides modular flexibility, allowing for the rapid integration of pre-trained components for fast deployment. However, prompt design is critical in this framework to effectively utilize the retrieved data. In Text Generation, REALM uses a dual BERT framework to simplify knowledge retrieval and integration, combining a pre-trained model with a knowledge extractor. Self-RAG introduces a critique module to judge whether retrieval is even necessary. REPLUG treats the language model as a black box (via API calls) and effectively integrates relevant external documents into the query. In-Context RALM uses BM25 for document retrieval and trains a predictive reranker to reorder and integrate top-ranked documents. In Code Domain, they integrates context from text or code into the prompt to improve downstream task performance. In Knowledge Base Q&A, systems like Uni-Parser, RNG-KBQA, and ECBRF combine queries and retrieved info into prompts to significantly boost performance and accuracy. In AI for Science, Chat-Orthopedist helps users make shared decisions by integrating retrieved data into model prompts, increasing accuracy. In Image Generation, RetrieveGAN integrates retrieved data (like selected image patches and their bounding boxes) into the generator's input stage to improve image relevance and precision. IC-GAN links noise vectors with instance features to adjust specific conditions and details of the generated image. In 3D Generation, RetDream first uses CLIP to retrieve relevant 3D assets, then merges the retrieved content with user input at the input stage.

2. Latent Representation-based RAG. The retrieved objects are integrated into the generative model in the form of latent representations. This enhances the model's comprehension capability and elevates the quality of generated content.  It possesses high adaptability to multi-modal and diverse tasks, capable of fusing the hidden states of the retriever and the generator. However, it requires additional training to align the latent spaces. It allows developers to seamlessly integrate complex algorithms for retrieval information. In Text Domain, FiD and RETRO are the two classic structures here. FiD uses different encoders to separately process each retrieved paragraph along with its title and the query, and then merges the generated latent representations, which are then decoded by a single decoder to generate the final output. RETRO retrieves relevant info for each split sub-query and introduces a new module called chunked cross-attention, CCA (Chunked Cross-Attention), to integrate the retrieved content with the tokens of each query. Furthermore, some works integrate k-Nearest Neighbors (kNN) search into Transformer blocks, theoretically solving the context length limitation issue that Transformers face. In Code and Science, FiD has found broad application across multiple tasks related to code and AI for Science. In Image Domain, some works use cross-attention to fuse retrieval results via latent representation. Others build text-image affine combination modules like ACM (Affine Combination Module) to directly connect hidden features. In Knowledge Domain, derivative methods of FiD are used for downstream tasks. EaE enhances the generator's understanding via entity-specific parametrization. TOME pivots towards fine-grained encoding of mentions, prioritizing mention granularity over just entity representation. In 3D Generation, ReMoDiffuse introduces a semantic modulated attention mechanism. AMD merges the original diffusion process with reference diffusion processes. In Audio Domain, some works use LLMs, combining dense feature encoding in the attention module to guide audio subtitle generation. Re-AudioLDM extracts deep features from text and audio using different encoders, then integrates them into the attention mechanism of the LDM (Latent Diffusion Model). In Video Subtitle Generation, R-ConvED uses a convolutional encoder-decoder network to process retrieved video-sentence pairs via an attention mechanism, generating hidden states to produce subtitles. CARE introduces a concept detector to generate conceptual probabilities, integrating conceptual representations into a hybrid attention mechanism. EgoInstructor uses gated cross-attention to combine text and video features.

3. In Logit-based RAG, the generative model integrates retrieval information through logits during the decoding process. Typically, logits compute the step-by-step probabilities generated by simple summation or model calculation. Logit-based RAG utilizes historical data to infer the current state and fuses information at the logit layer, which is highly suitable for sequence generation tasks. It focuses on training the generator and allows for the development of new methods for utilizing probability distributions, providing support for future tasks. In Text Domain, kNN-LM and its variants combine the language model's probability with the retrieval distance probability of similar prefixes at every decoding step. TRIME and NPM are radical evolutionary versions of kNN-LM, outputting highly aligned tokens using a local database, significantly improving performance, especially in long-tail distribution scenarios. Beyond text, code and image tasks also utilize Logit-based RAG. In Code Domain, some work uses the concept of kNN to enhance the control of final output, achieving better performance. EDITSUM integrates prototype summaries at the logit layer, enhancing the quality of code summaries. In Image Subtitle Generation, MA directly applies the kNN-LM framework to solve image subtitle problems with good results.

4. Speculative RAG. Speculative RAG uses retrieval instead of pure generation, thereby saving resources and accelerating response speed. REST replaces the small model in speculative decoding with retrieval, achieving draft generation. GPTCache addresses the high latency of using LLM APIs by constructing a semantic cache to store LLM responses. COG breaks down the text generation process into a series of copy-paste operations, retrieving words or phrases from documents rather than generating them. Speculative RAG is currently mostly applicable to sequential data. It decouples the generator and retriever, allowing pre-trained models to be used directly as components. Under this paradigm, a wider range of strategies can be explored to effectively utilize retrieved content.

### 7.2.1 RAG Components

#### 7.2.1.1 Retrieval Module

In a Retrieval-Augmented Generation (RAG) system, the final output quality is heavily dependent on the external knowledge base. The Retrieval Module is responsible for navigating this knowledge base to find the most relevant information. The type of data a RAG system uses directly impacts its performance and complexity. Historically, text was the primary source, but this has expanded significantly. 

1. Unstructured Data. This is the most widely used retrieval source, primarily obtained from massive text corpora. In ODQA (Open-Domain Question Answering) tasks, the main retrieval source is typically Wikipedia datasets. Mainstream versions include HotpotQA and DPR. Beyond general encyclopedias, unstructured data includes cross-lingual text and domain-specific data (e.g., Medical and Legal domains).

2. Semi-Structured Data. This refers to data that mixes text and tabular information, such as PDFs. Handling this is a major challenge for traditional RAG systems for two reasons. Text splitting processes can inadvertently split tables, leading to data corruption during retrieval. Fusing tables into the data increases the complexity of semantic similarity search. Current Solutions (which still have drawbacks and require further exploration) are to use the coding ability of an LLM to execute Text-2-SQL queries on tables in a database (e.g., TableGPT), or to convert tables into a text format and then perform further analysis based on text methods.

3. Structured data, such as Knowledge Graphs (KGs), is usually verified and can provide highly accurate information. KnowledgeGPT generates knowledge base search queries and stores knowledge in a personalized database to enhance the RAG model's knowledge richness. G-Retriever addresses LLM limitations in understanding text-based graph questions by combining Graph Neural Networks (GNNs), LLMs, and RAG. It uses LLM soft prompting to improve graph understanding and uses PCST (Prize-Collecting Steiner Tree) to optimize targeted graph retrieval. Building, verifying, and maintaining structured databases requires significant extra effort compared to unstructured data.

4. To overcome the limitations of external auxiliary information, some research focuses on mining the LLM's own internal knowledge. SKR classifies questions as known or unknown to selectively apply retrieval augmentation. GenRead directly replaces the retriever with an LLM generator, finding that the context generated by an LLM aligns better with causal language modeling pre-training goals and generally contains more accurate answers. Selfmem iteratively creates an unlimited memory pool through a retrieval-augmented generator and uses a memory selector to pick an output that forms a dual problem with the original question, achieving self-enhancement of the generative model.

Beyond the data format, the granularity of the retrieval unit is crucial. Coarse-grained retrieval units can provide more relevant information for a question but may also contain redundant content, which will distract the retriever and the language model in downstream tasks. Fine-grained retrieval units increases the burden of retrieval, and cannot fully guarantee semantic completeness and meet the required knowledge demands.
Selecting the appropriate granularity during reasoning can enhance the performance of dense retrievers and downstream task effectiveness. Granularity ranges from fine to coarse: Token -> Phrase -> Sentence -> Proposition -> Chunks -> Document. DenseX introduced the concept of using a Proposition as the retrieval unit. A proposition is defined as an atomic expression in text, encapsulating a unique factual fragment, presented in a concise, self-contained natural language format. This aims to improve retrieval accuracy and relevance. In Knowledge Graphs, granularity includes entity -> Triplet -> sub-Graph.

During the indexing phase, documents are processed, segmented, and transformed into Embedding vectors, which are then stored in a vector database. The quality of index construction determines whether the correct context can be retrieved in the retrieval phase. Chunking is the most common method is splitting documents by a fixed number of Tokens. Larger chunks capture more context but introduce noise, requiring longer processing time and higher costs. Smaller chunks have less noise but may fail to convey necessary context. Because chunking can truncate sentences, optimization methods like recursive splitting and sliding windows have emerged. These methods achieve layered retrieval by merging globally relevant information across multiple retrieval processes. However, balancing semantic completeness and context length remains difficult. Methods like Small2Big propose using small sentences as retrieval units while providing the surrounding large-scope sentences as context input to the LLM.

Chunks can be enriched with metadata like page number, file name, author, category, and timestamp. This allows for filtering during retrieval, narrowing the scope.
Assigning different weights to document timestamps allows for time-aware RAG, ensuring knowledge freshness and avoiding outdated information.
Besides extracting metadata, it can be artificially constructed. For example,  adding paragraph summaries, or introducing hypothetical questions. This is known as reverse HyDE. The LLM generates questions the document can answer, and during retrieval, the similarity between the original question and the hypothetical questions is calculated to reduce the semantic gap between question and answer.

Building a hierarchical structure for documents accelerates the retrieval and processing of relevant data. 

- Hierarchical Index Structure: Files are arranged in parent-child relationships, linked to chunks. Each node stores a data summary, aiding in rapid traversal of data, and helping the RAG system determine which chunks need to be extracted. This mitigates hallucinations caused by chunk extraction issues.

- Knowledge Graph Indexing: Using KGs to build document hierarchy maintains consistency. "This can represent connections between different concepts and entities, significantly reducing the possibility of hallucinations" It also transforms information retrieval into instructions the LLM understands, improving accuracy and generating coherent responses. To capture logical relationships, KGP proposed an indexing method using KGs across multiple documents. It consists of nodes (document paragraphs or structures) and edges (semantic/lexical similarity between paragraphs or structural relationships), effectively solving retrieval and reasoning in multi-document environments.

A primary challenge of basic RAG is directly relying on the user's original query as the retrieval basis. Formulating precise questions is hard, and inappropriate queries lead to poor retrieval. Problems include complex questions, unclear language, and language complexity and ambiguity. For instance, an LLM might struggle to determine if "LLM" refers to a Large Language Model or a Master of Laws depending on the context. To solve this, RAG systems employ Query Transformation and Expansion.

Expanding a single query into multiple queries enriches content and provides more context to cover specific missing details. 

- Multi-Query uses prompt engineering to utilize the LLM to expand the query. These queries execute in parallel. The expansion is carefully designed, not random.

- Sub-Query means that sub-question planning generates necessary sub-questions. Combining their answers provides a complete response to the original question. Specifically, complex questions can be broken down into a series of simpler sub-questions through a least-to-most prompting method.

- Chain-of-Verification (CoVe): The expanded queries are verified by the LLM to reduce hallucinations. Verified expanded queries have higher reliability.

The core concept of Query Transformation is retrieving fragments based on the transformed query, rather than the user's original query.

- Query Rewrite. The original query may be unsuitable for LLM retrieval. Therefore, one can prompt the LLM to rewrite the query. Beyond general LLMs, specialized smaller language models like RRR (Rewrite-Retrieve-Read) can be used. A method implemented in Taobao, BEQUE, significantly improved long-tail query recall, thereby boosting GMV.

- Query Optimization. Another method uses prompt engineering to have the LLM generate a new query based on the original one for subsequent retrieval. HyDE constructs a hypothetical document a hypothetical answer to the original query, focusing on the embedding similarity from answer to answe rather than from question or query to answer. Using Step-back prompting abstractly generates higher-level conceptual questions. In RAG, both the step-back question and original question are used for retrieval, and their combined results form the basis for the LLM's final answer.

Depending on the nature of the query, it must be directed to the appropriate RAG pipeline. Query Routing is crucial for multi-functional RAG systems designed for diverse scenarios.

- Metadata Router/Filter. The first step is often to extract keywords or entities from the query. Then, filter based on keywords and metadata within the fragments to narrow the search scope.

- Semantic Router. Another method utilizes the semantic information of the query to direct it to the appropriate pipeline.

Systems can also employ a hybrid routing approach, combining semantic and metadata methods to enhance query routing effectiveness.

In RAG, retrieval is achieved by calculating the similarity (e.g., Cosine Similarity) between the question and document fragment embeddings. The semantic representation capability of the embedding model plays a crucial role. This includes sparse encoders (like BM25) and dense retrievers (like pre-trained language models based on the BERT architecture). Recent work has introduced superior embedding models like AngIE, Voyage, and BGE, which benefit from multi-task instruction fine-tuning. Regarding the question of which embedding model to use, there is no one-size-fits-all answer. Certain models are more suitable for specific use cases.

Sparse and dense embedding methods capture different relevance features and can mutually benefit each other. For example, sparse retrieval models can provide initial search results to train dense retrieval models. Conversely, Pre-trained Language Models (PLMs) can be used to learn word weights, enhancing sparse retrieval. Research shows that sparse retrieval models can enhance the zero-shot retrieval capability of dense retrieval models, and help dense retrievers handle queries containing rare entities, thereby improving robustness.

hen the context significantly deviates from the pre-training corpus—especially in highly specialized fields like medicine or law which are full of jargon—fine-tuning the embedding model on domain-specific datasets becomes crucial to mitigate this discrepancy. Besides supplementing domain knowledge, another goal of fine-tuning is to align the retriever and the generator.

- LSR (LM-Supervised Retriever):"Uses LLM results as supervisory signals for fine-tuning.

- PROMPTAGATOR: Utilizes the LLM as a few-shot query generator to create task-specific retrievers, solving challenges in supervised fine-tuning, especially in data-scarce domains.

- LLM-Embedder: Uses the LLM to generate reward signals for multiple downstream tasks. The retriever is fine-tuned using two supervisory signals: hard labels from datasets and soft rewards from the LLM.

- REPLUG: Uses the retriever and LLM to calculate the probability distribution of retrieved documents, then performs supervised training by calculating KL divergence.

Fine-tuning models can bring challenges, such as API integration limits or restricted local compute resources. Therefore, some methods opt to introduce external adapters to assist alignment. UPRISE trains a lightweight prompt retriever capable of automatically retrieving prompts suitable for zero-shot task inputs from a pre-built prompt pool. AAR (Augmentation-Adapted Retriever) introduces a universal adapter aimed at adapting to multiple downstream tasks. PRCA adds a pluggable reward-driven contextual adapter to improve specific task performance. BGM fixes the retriever and LLM, training a bridging Seq2Seq model between them. This bridge model aims to transform the retrieved information into a format the LLM can process effectively, enabling dynamic paragraph selection per query. PKG introduces an innovative method integrating knowledge into a white-box model via instruction fine-tuning. In this method, the retriever module is directly replaced to generate relevant documents based on the query.

#### 7.2.1.2 Generation Module

In a Retrieval-Augmented Generation (RAG) system, simply taking all retrieved information and directly feeding it into a Large Language Model (LLM) is highly discouraged. The Generation Module is responsible for bridging the gap between raw retrieved data and the final generated answer. The core issue with raw retrieved data is noise and length. Redundant information can interfere with the LLM's final generated result, and an overly long context can also lead to the problem of the LLM losing intermediate information. Similar to humans, LLMs suffer from the "lost in the middle" phenomenon—they tend to focus heavily on the beginning and end of a long text while ignoring the middle. Therefore, retrieved content must be processed.

Reranking is the process of reordering the retrieved document fragments to highlight the most relevant results, thereby effectively reducing the scale of the overall document pool. This method has a dual role in information retrieval, acting as both an enhancer and a filter, providing optimized input for more precise language model processing. It can be rule-based (using predefined metrics like Diversity, Relevance, and MRR) or model-based (using specialized encoder-decoder models like SpanBERT, dedicated rerankers like Cohere rerank or bge-reranker-large, or general LLMs like GPT).

A common misconception in RAG is that providing as many relevant documents as possible and splicing them into one massive prompt is beneficial. Too much context can introduce more noise, weakening the LLM's ability to perceive key information. Therefore, context must be selected and compressed to reduce irrelevant data. LLMLingua uses a Small Language Model (SLM) like GPT-2 Small or LLaMA-7B to detect and remove unimportant Tokens, transforming them into a format that is difficult for humans to understand but well-understood by LLMs. This balances language integrity and compression ratio without needing to train the main LLM. RECOMP uses contrastive learning to train an information compressor, calculating contrastive loss across training data points consisting of one positive sample and five negative samples.

Filter-Rerank Paradigm combines the strengths of SLMs and LLMs. The SLM acts as a filter, and the LLM acts as the reranker. Asking the LLM to rerank difficult samples identified by the SLM can significantly improve Information Extraction (IE) tasks. Self-Evaluation is a simple but effective method is to have the LLM evaluate the retrieved content before generating the final answer. For example, in Chatlaw, the LLM is prompted to critique the legal provisions it intends to cite to evaluate their relevance.

Beyond adjusting the prompt, adjusting the LLM itself based on specific scenarios and data characteristics yields superior results. This is a major advantage of locally deployed LLMs. 

- Domain-Specific Adaptation. When the LLM lacks data in a specific domain, fine-tuning can provide it with additional knowledge.  Another advantage is adapting the model to specific input/output formats and generating responses in a desired style. For complex structured data retrieval, frameworks like SANTA implement a three-stage training scheme that effectively encapsulates fine structural and semantic differences. The first stage focuses on the retriever, using contrastive learning to optimize user queries and document embeddings.

- Alignment via Reinforcement Learning. Aligning the LLM's output with human or retriever preferences through reinforcement learning is another powerful method. For example, manually annotating the final generated answers, and then providing feedback through reinforcement learning. You can also align the LLM with the retriever's preferences. If you cannot access a powerful proprietary model (like GPT-4) or lack massive datasets, a simple method is to co-train the fine-tuning of the LLM with the fine-tuning of the retriever to align preferences. For instance, RA-DIT uses KL divergence to align the scoring functions between the retriever and the generator.

### 7.2.2 Training

Beyond how the retriever augments the generator, RAG methods can be broadly divided based on whether they require active model training to function effectively. Training-Free Methods  operate purely at the inference stage. They directly utilize the retrieved knowledge by inserting the retrieved text into the prompt. This avoids additional training, and this method has high computational efficiency. The potential drawback is that the retriever and generator components have not been specifically optimized for downstream tasks, which may lead to the retrieved knowledge not being fully utilized. To fully exploit external knowledge, many Training-Based Methods propose fine-tuning the retriever, the generator, or both, guiding the LLM to effectively adapt to and integrate the retrieved information. Based on their training strategy, these methods are further divided into three sub-categories:

- Independent Training Methods. These methods independently train each component in the RAG pipeline separately.

- Sequential Training Methods. This strategy involves first training one module, then freezing the already trained components to guide the tuning process of the other parts.

- Joint Training Methods. The most integrated approach, which involves simultaneously training the retriever and the generator.

#### 7.2.2.1 Training-Free Methods

