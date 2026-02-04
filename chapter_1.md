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

Code example:
```python
import re, collections

# 1. Initialize Vocabulary: Words are represented as character sequences 
# with an end-of-word </w> marker
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 
         'n e w e s t </w>': 6, 'w i d e s t </w>': 3}

def get_stats(vocab):
    """3. Count Frequent Pairs: Find the frequency of adjacent token pairs."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """4. Merge the Pairs: Replace the most frequent pair with a new merged token."""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# 5. Iterate: Repeat the process to build common subwords
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge #{i+1}: {best} -> {''.join(best)}")

print("\nFinal Vocabulary (Subwords):", vocab)
```

In frequency analysis and merging, for each iteration, the algorithm must scan the current vocabulary to find the most frequent adjacent pair. If there are $k$ desired merge operations, the complexity is roughly $O(k \cdot V)$, where $V$ is the average vocabulary size being scanned.

**Advantages**
- Robust OOV Handling: It naturally handles Out-of-Vocabulary (OOV) and new composite words by assembling them from existing subword representations.

- High versatility: BPE is an unsupervised learning algorithm capable of performing text segmentation without the need for manual human annotation.

**Disadvantages**
- Semantic Fragmentation: Once words are split into sub-units, some of the original semantic information may be partially lost or obscured.

- Computational Intensity: Training and performing inference with a subword tokenizer demands higher computational resources compared to simpler methods.

BPE tokenization performs global matching by iterating through the learned merge rules rather than iterating through character positions to find rule matches. Merge rules are strictly prioritized from highest to lowest; therefore, the tokenization process iterates through these rules sequentially to check for global matches within the text.

BPE tokenization code [example](https://github.com/huggingface/transformers/blob/05260a1fc1c8571a2b421ce72b680d5f1bc3e5a4/src/transformers/models/gpt2/tokenization_gpt2.py#L75).

#### 1.1.3.2 BBPE (Byte-level BPE)

Byte-level BPE (BBPE) is an evolution of the standard BPE algorithm used by models like GPT-2 and RoBERTa. While standard BPE often struggles with "Out-of-Vocabulary" (OOV) issues when encountering rare Unicode characters or emojis, BBPE solves this by operating on raw bytes rather than Unicode characters.

Code example:
```python
import collections

def bytes_to_unicode():
    """
    1. Base Vocabulary Initialization:
    Maps 256 bytes to unique Unicode strings to ensure no 'unknown' tokens.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# Initialize mapping and sample corpus
byte_encoder = bytes_to_unicode()
# 2. Byte Conversion: Convert raw text to our byte-string representation
corpus = "hello world"
tokens = [byte_encoder[b] for b in corpus.encode("utf-8")]
vocab = {" ".join(tokens): 1}

def get_stats(vocab):
    """3. Frequency Counting: Find most frequent adjacent byte pairs."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """4. Iterative Merging: Combine the best pair into a single new token."""
    v_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in v_in:
        # 5. Global Application: Apply the merge across the entire corpus
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = v_in[word]
    return v_out

# Perform 5 merges
for i in range(5):
    pairs = get_stats(vocab)
    if not pairs: break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge #{i+1}: {best}")

print("\nFinal BBPE Tokens:", list(vocab.keys())[0].split())
```

#### 1.1.3.3 WordPiece

WordPiece is a subword tokenization algorithm originally developed by Google for the Google Voice Search system and later popularized by BERT. It sits between character-level and word-level tokenization, designed to handle large vocabularies and out-of-vocabulary (OOV) words efficiently. 

WordPiece identifies the best subwords to merge by maximizing the likelihood of the training data. Here is the formal breakdown:

1. **Sentence Likelihood**: Assuming subwords are independent, the log-probability of a sentence $S$ is:
   $$\log P(S) = \sum_{i=1}^{n} \log P(t_i)$$

2. **The Change in Likelihood**: When merging subwords $x$ and $y$ into $z$, the impact on the total likelihood is:
   $$\log \left( \frac{P(t_z)}{P(t_x)P(t_y)} \right)$$

3. **Core Principle**: This value is equivalent to the **Mutual Information** between $x$ and $y$. 

WordPiece prioritize merges that result in the greatest increase in language model likelihood. It specifically targets pairs with the strongest statistical correlation in the corpus.

Here are the implementation steps of the WordPiece process:

1. Vocabulary Initialization: Start by collecting all individual characters present in the training corpus. This ensures that every word can at least be decomposed into its base characters, preventing "Unknown Token" errors.

2. Likelihood Model Construction: Build a language model using the current vocabulary tokens. The model assumes that the probability of a sentence is the product of the probabilities of its constituent subwords.

3. Candidate Selection (Mutual Information): Identify all potential pairs of adjacent subwords $(x, y)$ that could be merged into a new subword $z$. Calculate the **Mutual Information** for each pair using the formula:
     $$\text{Score} = \log \left( \frac{P(t_z)}{P(t_x)P(t_y)} \right)$$

4. Iterative Merging: Select the pair with the highest Mutual Information value and add the new merged subword $z$ to the vocabulary. Repeat this process until the desired vocabulary size (e.g., 30,000 tokens) is reached.

5. Greedy Tokenization: During the actual tokenization of new text, WordPiece uses a "Longest Match First" (MaxMatch) strategy. It scans the word from left to right and identifies the longest subword in its vocabulary that matches the beginning of the string, then repeats for the remaining part.

Code example:
```python
import collections
import math

def get_word_frequencies(corpus):
    """Initial step: Count word frequencies in the training data."""
    counts = collections.defaultdict(int)
    for sentence in corpus:
        for word in sentence.split():
            counts[word] += 1
    return counts

def segment_with_vocab(word, vocab):
    """Helper to split a word into current vocab units (Simplified MaxMatch)."""
    # This ensures we are evaluating pairs that actually exist in the current corpus state
    subwords = []
    start = 0
    while start < len(word):
        end = len(word)
        while start < end:
            substr = word[start:end]
            if start > 0: substr = "##" + substr
            if substr in vocab:
                subwords.append(substr)
                start = end
                break
            end -= 1
        if start < end: # Safety break for OOV characters
            start += 1
    return subwords

def find_best_mi_pair(word_counts, vocab):
    """
    Identifies the pair of subwords that maximizes Mutual Information.
    Score = P(xy) / (P(x) * P(y))
    """
    # 1. Segment the current words into subwords using the existing vocabulary
    # In a real implementation, this would use the MaxMatch/Greedy approach
    token_counts = collections.defaultdict(int)
    pair_counts = collections.defaultdict(int)
    
    for word, count in word_counts.items():
        # This is a simplified segmentation for demonstration
        # Real WordPiece would use the 'wordpiece_tokenize' logic here
        subwords = segment_with_vocab(word, vocab) 
        
        for i in range(len(subwords)):
            token_counts[subwords[i]] += count
            if i < len(subwords) - 1:
                pair = (subwords[i], subwords[i+1])
                pair_counts[pair] += count

    # 2. Calculate the Total Number of Tokens to derive probabilities
    total_tokens = sum(token_counts.values())
    
    best_score = -1
    best_pair = None

    # 3. Calculate MI Score for each pair
    for pair, count in pair_counts.items():
        x, y = pair
        # P(x, y) = count(xy) / total_tokens
        # P(x) = count(x) / total_tokens
        # P(y) = count(y) / total_tokens
        # Simplified Score: count(xy) / (count(x) * count(y))
        
        # We multiply by total_tokens to maintain the proper probability ratio
        score = (count * total_tokens) / (token_counts[x] * token_counts[y])
        
        if score > best_score:
            best_score = score
            best_pair = pair
            
    return best_pair

def train_wordpiece(corpus, target_vocab_size):
    """
    Implements WordPiece training logic:
    Selects merges that maximize language model likelihood (Mutual Information).
    """
    word_counts = get_word_frequencies(corpus)
    # Start vocabulary with all individual characters
    vocab = set()
    for word in word_counts:
        for char in word:
            vocab.add(char)
    
    # Add the ## prefix versions for subwords
    vocab.update({"##" + c for c in vocab})
    
    while len(vocab) < target_vocab_size:
        # 1. Calculate probabilities P(x) for all current tokens
        total_tokens = sum(word_counts.values()) # Simplified for example
        token_freqs = collections.defaultdict(int)
        
        # 2. Identify potential pairs and calculate scores
        pair_scores = {}
        # In practice, this requires segmenting the corpus with current vocab
        # Logic: Score = P(pair) / (P(first) * P(second))
        # This is the Mutual Information calculation

        best_pair = find_best_mi_pair(word_counts, vocab) 
        
        if not best_pair:
            break
            
        vocab.add("".join(best_pair))
        
    return vocab
```
