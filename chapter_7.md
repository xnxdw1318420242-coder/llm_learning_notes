# 7. LLM Application

## 7.1 LLM Application

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
