# 1. Transformer Basics
## 1.1 Tokenization

**Tokenization** is the fundamental bridge between raw human text and the numerical processing power of a machine. It is the process of converting a string of characters into a sequence of discrete units—tokens—that the model can understand and manipulate.

Depending on the level of granularity, tokenization is primarily categorized into three types: word-based, subword-based, and character-based. Each of these levels offers a distinct set of advantages and trade-offs depending on the specific use case. In the context of modern LLMs, it’s worth noting that subword-based (specifically algorithms like BPE or WordPiece) is the current industry standard because it solves the "Out of Vocabulary" (OOV) problem that plagues word-based models.

### 1.1.1 Word-based tokenization
A word-level tokenizer partitions text into individual words, which serve as the most natural linguistic units.

Here is a breakdown of its advantages and disadvantages based on the technical details provided:

**Advantages**
- Semantic Clarity: Words serve as the fundamental units of meaning, allowing the model to capture lexical intent more effectively.

- Natural Implementation: Word-level tokenization is the most straightforward approach to text segmentation, as it simply breaks down a string into individual words based on spaces or punctuation.

**Disadvantages**
- Massive Vocabulary Size: Maintaining a unique entry for every word leads to an enormous vocabulary. 

- Out-of-Vocabulary (OOV) Issues: This is a major hurdle. Word-based models struggle with new words, slang, or proper nouns that weren't in the original dictionary, requiring complex fallback mechanisms to handle them.

- A word-level tokenizer partitions text into individual words, which serve as the most natural linguistic units.

### 1.1.2 Character-based tokenization
Character-level tokenization is the most fundamental method of text segmentation, where text is split into individual characters—such as letters or punctuation marks—to serve as tokens.

**Advantages**
- Minimal Vocabulary Size: Since the number of unique characters in a language is limited, the resulting vocabulary is much smaller compared to other methods.

- Elimination of Out-of-Vocabulary (OOV) Issues: Because every word is constructed from the character set, the model can theoretically process any word it encounters, resolving the OOV problem.

- Suitability for Open Vocabularies: Its small vocabulary size makes it well-suited for handling a wide and diverse range of input text.

**Disadvantages**
- Weak Semantic Representation: From an intuitive standpoint, individual characters hold little to no inherent meaning; meaning is typically found at the word level rather than the letter level.

- Increased Learning Difficulty: The model must work harder to understand word meanings by learning the complex combinations and patterns of multiple characters.

- Significant Increase in Sequence Length: Character-level tokenization results in a much higher volume of tokens per sentence. While a word-level tokenizer might treat an entire word as a single token, this method converts it into many tokens.

### 1.1.3 Subword-based tokenization
Subword tokenization serves as a strategic middle ground between word-level and character-level methods. It functions on the principle that frequently used words should remain intact, while rarer terms should be decomposed into meaningful sub-units such as roots, prefixes, and suffixes.

**Advantages**
- Balanced Vocabulary and Semantics: This method effectively reduces the overall size of the vocabulary while still preserving the essential semantic meaning of words.

- Robust Handling of New Terms: It naturally manages Out-of-Vocabulary (OOV) and compound words by synthesizing representations from existing sub-word units.

**Disadvantages**
- Semantic Fragmentation: Once a word is split, the resulting sub-tokens may lose a portion of their original semantic clarity.

- Computational Complexity: These algorithms are more sophisticated than basic splitting methods, requiring greater computational resources for both the initial training phase and subsequent inference.

#### 1.1.3.1 BPE (Byte-Pair Encoding)

Byte Pair Encoding (BPE) is a popular unsupervised subword-based tokenization algorithm that strikes a balance between word-level and character-level methods. It solves the Out-of-Vocabulary (OOV) problem by ensuring that frequently occurring words remain intact while rare words are broken down into smaller, meaningful units. It has been adopted by many Transformer models, such as GPT, GPT-2, RoBERTa, BART, and DeBARTa.

Here are the 5 core steps of the BPE process:

1. Initialize Vocabulary: The process begins by breaking the entire training corpus into individual characters to create the initial base vocabulary.

2. Represent Words as Sequences: Every word in the corpus is represented as a sequence of these base characters, often with a special marker (like \<\/w\>) to denote the end of a word.

3. Count Frequent Pairs: The algorithm scans the text to identify the most frequently occurring pair of adjacent tokens (e.g., finding that "t" and "h" appear together most often).

4. Merge the Pairs: The most frequent pair is merged into a single new token (e.g., "th"), which is then added to the vocabulary.

5. Iterate: Steps 3 and 4 are repeated for a pre-defined number of iterations or until the desired vocabulary size is reached, allowing the model to build up common subwords and full words.
