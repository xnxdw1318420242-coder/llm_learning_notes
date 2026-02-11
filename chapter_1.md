# 1. LLM Basics
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

**Advantages**
- Zero Out-of-Vocabulary (OOV) Rate: By falling back to bytes ($256$ base tokens) instead of characters, BBPE can represent any string of text, ensuring the model never encounters an "unknown" token.

- Compact Vocabulary: It allows for a dynamically controlled vocabulary size that balances common words with rare subword units, making it efficient for large models.

- Strong Generalization: Breaking words into subwords allows the model to understand the semantic relationships between different forms of a word (e.g., "smart," "smarter," "smartest").

**Disadvantages**
- Reduced Encoding Efficiency: It can result in longer sequence lengths because single characters often decompose into multiple bytes, which increases the overall computational cost.

- Semantic Information Loss: BBPE ignores word-level structures, meaning it can lose more semantic meaning compared to methods using higher-level units.
  
- Complex Post-processing: The system becomes more complex because extra steps are needed to merge bytes back into their original character or word forms.
  
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

**Advantages**
- Efficient OOV Handling: Like BPE, it effectively manages Out-of-Vocabulary words by breaking them down into known sub-units, ensuring the model can process almost any input.

- Reduced Vocabulary Redundancy: The likelihood-based approach helps avoid merging common but uninformative character pairs that might otherwise take up space in a frequency-based vocabulary.

**Disadvantages**
- Potential for [UNK] Tokens: While it handles subwords well, if the base character set is not comprehensive, WordPiece may still resort to [UNK] tokens for truly unseen characters, a problem Byte-level BPE (BBPE) avoids.

- Loss of Single-Word Meaning: Highly frequent but long words may still be fragmented into sub-units, which can sometimes obscure the direct semantic meaning of the original word.

- Segmentation Ambiguity: Because WordPiece relies primarily on learned statistical correlations to merge units, it cannot resolve all segmentation ambiguities, which can lead to inconsistent or ambiguous tokenization results.

#### 1.1.3.4 Unigram

Unigram Language Model Tokenization is a subword tokenization method that treats tokenization as a probabilistic optimization problem. Unlike BPE or WordPiece, which start with characters and merge them upward, Unigram starts with a very large initial vocabulary and iteratively removes tokens that contribute the least to the overall likelihood of the corpus. The Unigram algorithm is a core component of the SentencePiece library and is the default tokenizer for several state-of-the-art transformer architectures, including AlBERT, T5, Big Bird, and XLNet.

The core philosophy of Unigram is to find the most likely way to segment a sentence $S$ into a series of subwords $\vec{x} = (x_1, x_2, \dots, x_m)$. If we assume that each subword $x_i$ exists independently, the probability of a specific segmentation $\vec{x}$ is the product of the probabilities of all its constituent subwords:

$$P(\vec{x}) = \prod_{i=1}^{m} P(x_i)$$

For a given sentence $S$, the best segmentation $x^*$ is the one that maximizes this likelihood among all possible tokenization combinations $U(x)$:

$$x^* = \text{arg max}_{x \in U(x)} P(\vec{x})$$

In practice, a vocabulary can contain tens of thousands of tokens, making it impossible to list and compare every possible combination of subwords manually. To solve this efficiently, Unigram utilizes the Viterbi algorithm to find the optimal path ($x^*$) through all potential segmentations.

To determine the probability $P(x_i)$ for each subword, Unigram uses the Expectation-Maximization (EM) algorithm. During the Maximization (M) step, the objective is to maximize the following likelihood function across the entire corpus $D$:

$$L = \sum_{s=1}^{|D|} \log(P(X^{(s)})) = \sum_{s=1}^{|D|} \log \left( \sum_{x \in U(X^{(s)})} P(x) \right)$$

This formula calculates the total probability by summing the probabilities of every possible segmentation for every sentence in the entire training corpus. By maximizing this value, the model learns subword probabilities that best represent the actual patterns in the language data.

Here are the implementation steps of the Unigram process:

1. Vocabulary Initialization: Generate an extensive initial vocabulary by collecting all individual characters plus the most frequent substrings found in the training corpus. The goal is to start with a set of subwords that is significantly larger than your desired final vocabulary size.

2. Estimate Probabilities using EM Algorithm: Apply the Expectation-Maximization (EM) algorithm to estimate the occurrence probability $P(x_i)$ for every subword in the current vocabulary. The algorithm maximizes the log-likelihood of the entire training corpus by considering all possible segmentations for every sentence.

3. Calculate the "Loss" for Each Token: For every subword in the current vocabulary, calculate how much the total corpus likelihood would decrease if that specific token were removed. 

4. Prune the Vocabulary: Sort the tokens by their loss value. Remove a fixed percentage (e.g., 10–20%) of tokens that have the lowest impact on the overall likelihood. Always keep individual characters in the vocabulary to ensure the model can always tokenize any string (avoiding OOV issues).

5. Repeat Until Target Size is Reached: Repeat steps 2 through 4 until the vocabulary reaches your predefined size (e.g., 32,000 tokens).

 Training code example:
 ```python
import math
import collections

def train_unigram(corpus, target_vocab_size):
    """
    Implements the Unigram training process:
    1. Initialize a large seed vocabulary.
    2. Use EM to estimate probabilities.
    3. Prune tokens that contribute the least to corpus likelihood.
    """
    # Step 1: Initialize Seed Vocabulary
    # In practice, this would be all frequent substrings.
    # Here we simplify it to common words split into substrings.
    word_freqs = collections.Counter(" ".join(corpus).split())
    vocab = initialize_seed_vocab(word_freqs)

    while len(vocab) > target_vocab_size:
        # Step 2: Expectation-Maximization (EM) Step
        # Estimate P(x) for each subword to maximize corpus likelihood L
        token_probs = estimate_probabilities(word_freqs, vocab)

        # Step 3: Loss Calculation
        # How much does the total log-likelihood L drop if token x is removed?
        # L = Σ log(Σ P(x))
        token_losses = calculate_token_losses(word_freqs, vocab, token_probs)

        # Step 4: Pruning
        # Sort by loss and remove the bottom 10-20% of tokens
        # Note: Never prune individual characters to avoid OOV!
        sorted_tokens = sorted(token_losses.items(), key=lambda x: x[1], reverse=True)
        keep_count = max(target_vocab_size, int(len(vocab) * 0.8))
        vocab = {t for t, loss in sorted_tokens[:keep_count]}
        
    return vocab

def estimate_probabilities(word_freqs, vocab):
    """
    M-step of the EM algorithm: Maximize the likelihood function L.
    This calculates the normalized frequency of tokens in the optimal segmentations.
    """
    counts = collections.defaultdict(float)
    total_count = 0
    
    for word, freq in word_freqs.items():
        # Find optimal segmentation using current probabilities (Viterbi)
        # and increment counts of subwords used in those segmentations.
        best_segmentation = viterbi_segment(word, vocab)
        for token in best_segmentation:
            counts[token] += freq
            total_count += freq
            
    # Return P(x_i) = count(x_i) / total_count
    return {token: count / total_count for token, count in counts.items()}

def calculate_token_losses(word_freqs, vocab, token_probs):
    """
    Calculates the impact of removing each token on the global likelihood L.
    L = Σ log(P(S))
    """
    token_losses = {}
    current_total_log_likelihood = compute_corpus_likelihood(word_freqs, vocab, token_probs)
    
    for token in vocab:
        if len(token) == 1: continue # Skip base characters
        
        # Temporarily remove token and see how much the likelihood drops
        temp_vocab = vocab - {token}
        new_likelihood = compute_corpus_likelihood(word_freqs, temp_vocab, token_probs)
        token_losses[token] = current_total_log_likelihood - new_likelihood
        
    return token_losses

def compute_corpus_likelihood(word_freqs, vocab, token_probs):
    """
    Calculates the total log-likelihood of the corpus.
    L = Σ log(P(word)) * frequency
    """
    total_log_likelihood = 0.0
    
    for word, freq in word_freqs.items():
        # For each word, we calculate the sum of probabilities of all possible segmentations.
        # We use a variation of the Forward Algorithm (Dynamic Programming).
        word_prob = compute_word_likelihood(word, vocab, token_probs)
        
        # Add to total using log to handle the corpus-level product
        if word_prob > 0:
            total_log_likelihood += freq * math.log(word_prob)
        else:
            # Handle cases where word cannot be formed (should not happen if chars are kept)
            total_log_likelihood += freq * -1e10 
            
    return total_log_likelihood

def compute_word_likelihood(word, vocab, token_probs):
    """
    Calculates the sum of probabilities of ALL possible ways to segment a word.
    Implementation of: Σ P(x) for x in U(X)
    """
    n = len(word)
    # dp[i] stores the sum of probabilities of all segments ending at index i
    dp = [0.0] * (n + 1)
    dp[0] = 1.0 # Base case: empty string has probability 1
    
    for end_idx in range(1, n + 1):
        for start_idx in range(end_idx):
            subword = word[start_idx:end_idx]
            
            # If subword is in current vocabulary, add its contribution
            if subword in vocab and subword in token_probs:
                # Probability of path to start_idx * Probability of this subword
                dp[end_idx] += dp[start_idx] * token_probs[subword]
                
    return dp[n]
```

Inference code example:
```python
import math

def unigram_viterbi_tokenize(text, vocab_probs):
    """
    Implements Viterbi decoding to find the most likely segmentation.
    x* = arg max P(x) where P(x) is the product of subword probabilities.
    """
    n = len(text)
    
    # best_probabilities[i] stores the maximum log-probability to reach position i
    # Initialize with negative infinity
    best_probs = [-float("inf")] * (n + 1)
    best_probs[0] = 0.0
    
    # best_segment_starts[i] stores the starting index of the best subword ending at i
    best_segment_starts = [0] * (n + 1)

    # 1. Forward Pass: Dynamic Programming to find the max likelihood path
    for end_idx in range(1, n + 1):
        for start_idx in range(end_idx):
            subword = text[start_idx:end_idx]
            
            if subword in vocab_probs:
                # Use log-probabilities to avoid numerical underflow (summing logs = multiplying probs)
                # Formula: log(P(x1...xi)) = log(P(x1...xj)) + log(P(xj...xi))
                log_prob = math.log(vocab_probs[subword])
                current_prob = best_probs[start_idx] + log_prob
                
                if current_prob > best_probs[end_idx]:
                    best_probs[end_idx] = current_prob
                    best_segment_starts[end_idx] = start_idx

    # 2. Backward Pass: Reconstruct the best segmentation path
    if best_probs[n] == -float("inf"):
        return ["[UNK]"]

    tokens = []
    curr = n
    while curr > 0:
        start = best_segment_starts[curr]
        tokens.append(text[start:curr])
        curr = start
    
    # Reverse because we backtracked from the end
    return tokens[::-1]

# --- Example Usage ---
# Probabilities would typically be estimated via the EM algorithm
sample_vocab_probs = {
    "h": 0.1, "e": 0.1, "l": 0.1, "o": 0.1,
    "he": 0.2, "llo": 0.3, "hello": 0.5
}

# For "hello", Unigram will choose ['hello'] because it has the highest individual probability.
print(unigram_viterbi_tokenize("hello", sample_vocab_probs))
```

**Advantages**
- Simple and Efficient: The implementation is relatively straightforward and computationally efficient, making it well-suited for processing large-scale datasets.

- Highly Customizable: By preprocessing training samples and statistical word frequencies, it can be tailored with custom rules to meet the specific tokenization needs of different domains and tasks.

  
**Disadvantages**
- Lack of Contextual Information: The algorithm only considers the probability of each word in isolation. This lack of context can result in ambiguous or blurry segmentation results.

- Out-of-Vocabulary (OOV) Issues: It has a limited ability to handle "unseen" words that did not appear in the training set, potentially leading to incorrect segmentation of OOV terms.

- Ambiguity Problems: Because certain words can have different meanings, the Unigram algorithm may fail to accurately segment them without contextual cues.


#### 1.1.3.5 SentencePiece

SentencePiece is an open-source subword tokenization library developed by Google that treats the input text as a raw stream of characters, including spaces. It is unique because it performs tokenization and detokenization without requiring language-specific pre-tokenizers (like splitting by whitespace), making it truly language-independent.
SentencePiece is designed as a language-independent subword tokenizer. It treats the input as a raw stream of characters, relying on these four internal modules:

1. Normalizer
* Standardization: Converts raw text into a consistent format using **Unicode normalization** (typically NFKC) to handle character variations.
* Space Handling: Replaces whitespaces with a visible meta-symbol (usually `_`), allowing spaces to be treated as standard characters within the vocabulary.
  
2. Trainer
* Vocabulary Building: Learns the subword units from a training corpus using either **BPE** or **Unigram** logic.
* Probabilistic Estimation: In Unigram mode, it utilizes the **EM (Expectation-Maximization) algorithm** to find subword probabilities that maximize the total log-likelihood of the corpus.

3. Encoder (Tokenizer)
* Segmentation: Transforms the normalized text into a sequence of subword tokens or numerical IDs.
* Optimal Pathing: For Unigram, it employs the **Viterbi algorithm** to find the most likely segmentation by maximizing the product of the probabilities of all constituent subwords.

4. Decoder (Detokenizer)
* Reconstruction: Converts subword sequences back into the original raw text string.
* Lossless Mapping: Because spaces were preserved as meta-symbols, the decoder simply joins the tokens and restores standard whitespaces for a perfect reconstruction of the original text.

**Advantages**
- Dynamic Vocabulary: By merging units, SentencePiece can dynamically control vocabulary size, allowing it to adapt to different tasks and data scales efficiently.

- Superior Segmentation: It segments words into subwords with high precision, which provides better semantic representation and overall tokenization performance.

- Reduced OOV Issues: By breaking unknown words into known subword units, it significantly reduces out-of-vocabulary (OOV) problems and improves the model's ability to generalize.

**Disadvantages**
- Computational Cost: The Unigram training process involves iterative Expectation-Maximization (EM) steps, which are more computationally expensive than the simple frequency-counting used in BPE.

- Ambiguity in Segmentation: Certain words may have multiple valid segmentations depending on the learned vocabulary, which the algorithm may not always resolve accurately across different contexts.

## 1.2 Embedding

In the context of Large Language Models (LLMs), an **embedding** is a numerical representation of a word, subword, or sentence. It converts the discrete tokens produced by algorithms like Unigram or BBPE into a continuous vector of numbers that a machine can actually "understand." The process includes the following steps:
1. Build a Vocabulary: A collection of all unique words (or subwords) is created, where every word is assigned a unique numerical index.
2. Initialize the Embedding Matrix: A matrix is created with dimensions of (Vocabulary Size × Embedding Dimension). Each row in this matrix represents the "embedding vector" for a specific word.
3. Token to Index: Each word in the input text is converted into its corresponding unique index from the vocabulary.
4. Lookup Embedding Vectors: The model uses these indices to locate and "look up" the specific vector row from the embedding matrix.

An embedding matrix is a fundamental lookup table used in neural networks to translate discrete token IDs into dense, continuous vectors that a model can process mathematically.  It is a large matrix of weights with dimensions $V \times D$, where $V$ is the Vocabulary Size (the number of unique tokens, like 50,000) and $D$ is the Embedding Dimension (the length of the vector, like 768 or 1024). Every value in this matrix is a trainable parameter. During the model's training phase, these numbers are adjusted so that tokens with similar meanings end up with similar vector values.

In the context of building a language model, there are two primary ways to initialize an Embedding Matrix to translate token indices into dense vectors.
1. Random Initialization: In this approach, the matrix is initialized with small, random numbers (often following a specific distribution like Xavier or Heuristic initialization). The model learns the "meaning" of these vectors from scratch during training. As the model processes text, it updates these weights via backpropagation until words with similar meanings cluster together in the vector space.
2. Pre-trained Initialization: This approach involves using an embedding matrix that has already been trained on a massive external corpus (like Word2Vec, GloVe, or FastText). Instead of random numbers, the rows are populated with vectors that already represent semantic relationships learned from other data. This gives the model a head start. It already "knows" basic semantic associations, which is particularly helpful when the current training dataset is small.

### 1.2.1 History 

#### 1.2.1.1 One-hot Encoding

One-Hot Encoding is the most basic way to represent words as numerical vectors. Each word in the vocabulary is represented by a vector of the same length as the total vocabulary size. In this vector, only one element is set to 1 (at the unique index assigned to that word), while all other elements are set to 0.

One-hot Encoding has two disadvantages. First, as the vocabulary grows, the vectors become extremely long, leading to a massive, memory-intensive sparse matrix. Most of the vector consists of zeros, which provides very little useful information for the model to learn complex patterns compared to dense embeddings. Second, One-hot vectors treat every word as equidistant. They cannot capture relationships; for example, "cat" and "dog" are mathematically as different as "cat" and "refrigerator".

#### 1.2.1.2 Co-occurrence Matrix
A co-occurrence matrix is a statistical tool used to capture the relationships between words based on how frequently they appear near each other within a specified "window" of text. The matrix counts how many times two words appear together within a fixed distance (e.g., 2 or 5 words) across an entire corpus. It is a square matrix of size $V \times V$ (where $V$ is vocabulary size), where the value at $(\text{word } A, \text{word } B)$ is the count of their shared appearances. Words that appear in similar contexts will have similar row vectors, reflecting the distributional hypothesis that "similar words appear in similar neighborhoods."

The matrix size increases quadratically with the vocabulary ($V^2$), leading to massive storage and computational requirements for large datasets. Most word pairs never appear together, resulting in a matrix filled mostly with zeros, which is inefficient for model learning.

#### 1.2.1.3 Distributed Word Representation
Distributed word representation (embeddings) addresses the limitations of sparse methods like one-hot encoding or co-occurrence matrices by mapping words into a dense, low-dimensional space. Unlike one-hot encoding where all words are equidistant, distributed representations place words with similar meanings closer together in the vector space.

### 1.2.2 Static Embeddings

Static vectors are defined as representations that, once training is complete, no longer change. Regardless of the future scenario or context the word appears in, its corresponding vector remains the same. This approach includes well-known methods such as Word2Vec, GloVe, and FastText.

#### 1.2.2.1 Word2Vec

Word2Vec is a groundbreaking framework developed by Google in 2013 that uses a shallow, two-layer neural network to learn static word embeddings. It is based on the idea that "a word is characterized by the company it keeps." The model processes a large corpus of text and maps each word to a dense vector (typically 100–300 dimensions). The result of Word2Vec is a static lookup table. Once trained, the vector for "apple" is fixed, regardless of whether you are talking about the fruit or the tech company.

<p align="center">
  <img width="329" height="182" alt="eee82a74-7b47-4513-a8df-ca6b4f7fc1da" src="https://github.com/user-attachments/assets/a0c9c73d-b19f-4543-ab03-aa1812dd14d0" />
</p>

In Word2Vec, the CBOW and Skip-gram architectures are two different strategies for training a model to understand word meanings based on context. Skip-gram generally performs better with large datasets and is superior at representing rare words. CBOW is faster to train and often performs well with smaller datasets or when focusing on frequent words.

Here are the specific steps of CBOW:

1. Input and Encoding: The model identifies the context words within a fixed "window" (e.g., two words before and two words after the target). These words are converted into One-Hot Vectors.
- Example sentence: "The quick brown [fox] jumps over."
- Inputs: "quick", "brown", "jumps", "over".

2. Vector Lookup: Each one-hot vector is multiplied by an Input Embedding Matrix. This step essentially "plucks" the corresponding dense word vector for each context word from the vocabulary table.

3. Averaging (The "Bag" Step): The vectors of all the context words are summed or averaged together to create a single representative vector. This combined vector represents the overall "vibe" or meaning of the surrounding context.

4. Output Prediction: The averaged vector is then passed through an Output Embedding Matrix (and usually a Softmax layer). This produces a probability distribution across the entire vocabulary.

5. Loss Calculation: The model compares its prediction to the actual target word (e.g., "fox"). If the probability for "fox" is low, the model uses Backpropagation to adjust the weights in both embedding matrices. Over time, the vectors for words that appear in similar contexts (like "fox" and "wolf") begin to move closer together in the vector space.

Once the CBOW model is trained, we generally stop using the "prediction" part of the network and keep the learned weights. Discard the output layer and keep the Input Embedding Matrix. Each word in the vocabulary now corresponds to a specific row in that matrix. 

<p align="center">
  <img width="214" height="309" alt="02fa6ce2-416f-4257-8708-52dafe71bc91" src="https://github.com/user-attachments/assets/7270dcb1-e20b-4c14-891a-c0a74f70e15a" />
</p>

Skip-gram is the "inverse" of CBOW. Instead of using the context to guess a word, it takes a single target word and tries to predict the context words that likely surround it.

Here are the steps the model follows during training:

1. The model selects a center word (target) from the text and identifies its neighbors within a set window size.
- Example sentence: "The quick brown [fox] jumps over."
- Input: "brown"
- Targets to predict: "quick", "fox"

2. Vector Lookup: The input word ("brown") is converted into a One-Hot Vector. This vector is multiplied by the Input Embedding Matrix to retrieve the specific dense vector for that word.

3. Parallel Predictions: Unlike CBOW, which averages vectors, Skip-gram uses the single input vector to make multiple separate predictions. It tries to calculate the probability of each word in the vocabulary appearing in the nearby slots (e.g., position -1, position +1, etc.).

4. Softmax and Probability: The input vector is multiplied by the Output Embedding Matrix. A Softmax function is applied to turn the results into probabilities. The goal is for the probabilities of "quick" and "fox" to be high, while the probability of unrelated words like "keyboard" remains low.

5. Loss Calculation: The model calculates the error (loss) between its predictions and the actual context words. It then uses Backpropagation to update the weights in both the Input and Output matrices.

<p align="center">
<img width="320" height="465" alt="d82b8625-0387-48b6-9ac3-57deabdf2104" src="https://github.com/user-attachments/assets/51cfe6cd-ed3b-4663-af7a-a8a5462d3393" />
</p>

In standard Word2Vec, the final layer uses a Softmax function that must calculate a probability for every single word in the vocabulary (which could be 50,000+ words). This is computationally "expensive" and slow. Hierarchical Softmax and Negative Sampling are two optimization techniques designed to solve this bottleneck.

#### 1.2.2.2 GloVe

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm developed by Stanford for generating dense word embeddings. Unlike Word2Vec, which uses a "predictive" approach (local context windows), GloVe is a "count-based" model that leverages global co-occurrence statistics from the entire text corpus.

The core insight of GloVe is that the ratio of word-word co-occurrence probabilities encodes semantic meaning. For example, if you look at the words "ice" and "steam":

- The word "solid" co-occurs frequently with "ice" but rarely with "steam."

- The word "gas" co-occurs frequently with "steam" but rarely with "ice."

- The word "water" co-occurs frequently with both. By looking at the ratio of these probabilities, the model can precisely distinguish the relationship between words.

Training Steps
1. Build a Vocabulary: Tokenize the corpus and create a unique set of words.
   
2. Construct a Co-occurrence Matrix ($X$): Scan the corpus with a fixed window size. For every pair of words $(i, j)$, count how many times they appear near each other. $X_{ij}$ represents the global count of word $j$ appearing in the context of word $i$.
   
3. Calculate Co-occurrence Probabilities: Compute $P_{ij} = \frac{X_{ij}}{X_i}$, where $X_i$ is the total count of any word appearing in the context of word $i$.
   
4. Define the Objective Function: The model aims to learn vectors $w_i$ and $w_j$ such that their dot product equals the logarithm of their co-occurrence:
   
$$w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log(X_{ij})$$

5. Apply Weighted Least Squares: To prevent common words (like "the") from dominating the loss, a weighting function $f(X_{ij})$ is used. It assigns lower weights to rare word pairs and caps the weight for extremely frequent ones.
   
$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log(X_{ij}))^2$$

   There are three critical conditions that the weighting function must meet to work effectively:
   - Non-decreasing: Common word pairs (those that appear together often) should have a higher weight than rare pairs. Therefore, $f(x)$ must be a non-decreasing function.
   - Saturation: The weight should not increase indefinitely. Once the co-occurrence reaches a certain level, the weight should cap out (not increase further) to prevent extremely frequent words from dominating the loss.
   - Handling Zero Counts: If two words never appear together ($X_{ij} = 0$), they should not contribute to the loss function. This means $f(0) = 0$.

   The authors of the GloVe paper chose a specific piecewise function to satisfy these conditions:
   
$$f(x) = \begin{cases} (x/x_{max})^\alpha & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}$$
   
   - $\alpha$ (Alpha): Set to 0.75 for all experiments.
   
   - $x_{max}$: Fixed at 100.
   
   - The Curve: As shown in the graph, the weight grows quickly at first but "levels off" at 1.0 once the co-occurrence count ($X_{ij}$) hits 100.
    
6. Optimization: Use Stochastic Gradient Descent (SGD) to minimize the difference between the dot product of the vectors and the actual log-counts.

**Example: King vs. Queen**
In a large dataset, "King" and "Queen" will both frequently co-occur with words like "throne," "crown," and "rule." 
- **GloVe** records these counts globally.
- The model adjusts their vectors so the dot product of `vector(King)` and `vector(throne)` is high.
- Because they share many "probe words," `vector(King)` and `vector(Queen)` end up close together in the vector space.

> **Result:** King - Man + Woman ≈ Queen

GloVe uses the AdaGrad gradient descent algorithm. It performs stochastic sampling on all non-zero elements within the global co-occurrence matrix $X$. During training, its iteration count varies by vector size: 50 iterations for vectors under 300 dimensions, and 100 iterations for larger vectors. The model achieves optimal performance with a vector dimension of 300 and a context window size ranging approximately from 6 to 10.

The training process produces two separate sets of vectors, $w$ and $\tilde{w}$. To improve robustness and reduce noise, the final word representation used is the sum of the two vectors ($w + \tilde{w}$).


#### 1.2.2.3 FastText

FastText is an open-source library for word embeddings and text classification developed by Facebook's AI Research (FAIR) lab. It is essentially an evolution of the Word2Vec model that treats words as more than just individual tokens. In text classification tasks, FastText often achieves accuracy levels comparable to deep neural networks, yet it is many orders of magnitude faster in terms of training time. 

Unlike Word2Vec or GloVe, which assign a single vector to each unique word, FastText breaks words down into character n-grams. For example, with $n=3$, the word "apple" would be broken into: <ap, app, ppl, ple, le>. The final vector for "apple" is the sum of the vectors of these subword n-grams. By utilizing subword n-grams, the model significantly improves its understanding of phrases and varied expressions. This approach generates superior vectors for rare words and uniquely enables the construction of embeddings for out-of-vocabulary (OOV) terms by leveraging their internal character patterns.

FastText consists of three primary layers:
- Input Layer: Words and their features are represented as vectors.
- Hidden Layer: This layer calculates the average of multiple input vectors.
- Output Layer: A specific target—usually a document label—is predicted.

FastText introduces several unique elements designed for text classification:
- Input Features: Instead of just using the context words surrounding a target, FastText uses a combination of individual words and their subword n-gram features to represent a document.
- Vector Processing: Inputs are already processed as embeddings.
- The Objective: FastText predicts the class label (category) of the entire document.
- Character-level n-grams: By including subword features as extra inputs, the model captures internal word structure. This is a major reason why the model is so effective at classification.
- Hierarchical Softmax: At the output stage, FastText uses a hierarchical structure to process the multi-class classification. This significantly reduces the total training time compared to standard softmax.

The model effectively treats a document as a "Bag of Words". It first Collects vectors for all words and n-grams in the document, superimpose and average these vectors to form a combined document-level vector, and feed this final document vector into the linear softmax classifier to determine the category.

### 1.2.3 Contextual Embeddings

While traditional models like Word2Vec and GloVe assign a single static vector to each word, **Contextual Embeddings** represent a major leap forward by allowing a word's vector to change based on the words surrounding it. 

In static models, the word "bank" has the same vector whether you are talking about a "river bank" or a "bank account." Contextual embeddings solve this "polysemy" problem by looking at the entire sentence before assigning a vector. This allows the model to capture deep semantic meaning and syntax based on specific usage.

ELMo was one of the first models to successfully implement this idea. BERT revolutionized the field by moving away from LSTMs and adopting the Transformer architecture. We will introduce the technical details, training objectives, and implementation of ELMo and BERT in depth in the upcoming chapters.

## 1.3 Positional Encoding

Unlike previous sequential models (RNNs or LSTMs), the Transformer architecture processes an entire sequence in parallel rather than word by word. While this makes training much faster, it introduces a major flaw: Permutation Invariance.

Without additional information, the Transformer "sees" a sentence as a "bag of words." For example, the model would treat these two sentences as identical because they contain the same tokens:

- "The dog bit the man."

- "The man bit the dog."

Since self-attention calculates relationships between tokens regardless of their distance, we must explicitly inject information about the order of the words to preserve grammatical and semantic meaning.

**Positional Encoding (PE)** is a technique where a unique vector is added to each input word embedding to represent its specific position in the sequence. This vector can be viewed as a signal that is either added to or concatenated with the token's word vector (embedding) or feature vector. This process ensures that the subsequent self-attention mechanism is aware of the positional information for every token in the sequence. There are several primary methods for positional encoding:

* Absolute Positional Encoding
    * Learnable Positional Embedding
    * Sinusoidal Positional Encoding
    
* Relative Positional Encoding
    * RoPE (Rotary Position Embedding)
    * ALiBi (Attention with Linear Biases)
 
### 1.3.1 Learnable Positional Embedding

Unlike Sinusoidal encoding, which uses fixed mathematical functions, Learnable Positional Embeddings treat the position of a token as a parameter to be optimized during the training process. Instead of calculating a static value, the model initializes a dedicated embedding matrix where each row represents a specific position index (e.g., Position 0, Position 1, etc.). This method is the standard approach for several foundational Transformer architectures including BERT and GPT.

Training this method may require more data to effectively cover a longer range of sequences. Also, the maximum sequence length is fixed; if the model encounters a sequence longer than the maximum length used during training, it cannot effectively "extrapolate" or generalize to those new position indices.

Implementation details:
1. Initialization: A matrix $P \in \mathbb{R}^{L_{max} \times d_{model}}$ is initialized, where $L_{max}$ represents a pre-defined maximum sequence length.
2. Vector Assignment: For the token at a specific $position$ in a sequence, its positional encoding is the corresponding row vector $P[position]$ from this matrix.
3. Training Process: During training, this matrix $P$ is updated via backpropagation, similar to how word embeddings are updated, allowing the model to gradually learn how to encode different positions.

Code example:
```python
import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        """
        Implementation of Absolute Learnable Positional Encoding.
        
        Args:
            d_model: The dimensionality of the word vectors/model (d_model).
            max_len: The pre-defined maximum sequence length (L_max).
        """
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor with positional signals added to word vectors.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_encodings = self.position_embeddings(positions)
        
        return x + pos_encodings

# Example Usage:
# model = LearnablePositionalEncoding(d_model=768, max_len=512)
# input_embeddings = torch.randn(1, 10, 768) 
# output = model(input_embeddings)
```
**Advantages**
- Simplicity: It treats positional information exactly like word embeddings, making it straightforward to implement using standard embedding layers in frameworks like PyTorch.

**Disadvantages**
- Poor Generalization: If the model encounters a sequence during inference that is longer than the maximum length seen during training, it cannot effectively "extrapolate" or generalize to those new position indices.

- Memory Constraints: A separate vector must be stored for every position up to $L_{max}$, which can become memory-intensive if the context window is extremely large.

- Data Hunger: Training these embeddings may require a larger volume of data to ensure the model effectively learns representations for every possible position index in the sequence range.

### 1.3.3 Sinusoidal Positional Encoding

Since Transformer models process all tokens in a sequence simultaneously, they lack an inherent understanding of word order. Sinusoidal Positional Encoding solves this by injecting a unique mathematical "signature" into each token based on its position.

For a token at position $position$ (starting from 0) and a model with dimensionality $d_{model}$, the encoding for each dimension $i$ is calculated using sine and cosine functions:
- Even Dimensions ($2i$): Uses the sine function.

$$PE_{(position, 2i)} = \sin\left(\frac{position}{10000^{\frac{2i}{d_{model}}}}\right)$$

- Odd Dimensions ($2i+1$): Uses the cosine function.

$$PE_{(position, 2i+1)} = \cos\left(\frac{position}{10000^{\frac{2i}{d_{model}}}}\right)$$

The denominator $10000^{\frac{2i}{d_{model}}}$ controls the wavelength/frequency across different dimensions. As the dimension index $i$ increases, the wavelength becomes longer, allowing the model to capture a wide range of positional relationships. When a sequence length exceeds the maximum length seen during training, the model can still interpret position information by extending the sine and cosine periods. As the $position$ index increases, the changes in the encoding vector are smooth and continuous, without sudden jumps. All values are generated using a fixed formula, meaning no additional learnable weights are added to the model.

One of the most powerful properties of this method is that it allows the model to learn relative positions through linear transformations. By using these geometric properties, the Transformer doesn't just know "where" a word is; it can mathematically calculate how far apart words are from one another, regardless of where they appear in the sentence. For any two positions separated by a fixed distance $\Delta$, the encoding at $pos + \Delta$ can be expressed as a rotation of the encoding at $pos$. Mathematically, this is represented by a rotation matrix:

$$
\begin{aligned}
\begin{bmatrix} 
PE_{(pos + \Delta, 2i)} \\ 
PE_{(pos + \Delta, 2i+1)} 
\end{bmatrix} 
&= \begin{bmatrix} 
\sin((pos + \Delta) \cdot \theta_i) \\ 
\cos((pos + \Delta) \cdot \theta_i) 
\end{bmatrix} \\
&= \begin{bmatrix} 
\sin(pos \cdot \theta_i)\cos(\Delta \cdot \theta_i) + \cos(pos \cdot \theta_i)\sin(\Delta \cdot \theta_i) \\ 
\cos(pos \cdot \theta_i)\cos(\Delta \cdot \theta_i) - \sin(pos \cdot \theta_i)\sin(\Delta \cdot \theta_i) 
\end{bmatrix} \\
&= \begin{bmatrix} 
\cos(\Delta \cdot \theta_i) & \sin(\Delta \cdot \theta_i) \\ 
-\sin(\Delta \cdot \theta_i) & \cos(\Delta \cdot \theta_i) 
\end{bmatrix} 
\begin{bmatrix} 
\sin(pos \cdot \theta_i) \\ 
\cos(pos \cdot \theta_i) 
\end{bmatrix} \\
&= \begin{bmatrix} 
\cos(\Delta \cdot \theta_i) & \sin(\Delta \cdot \theta_i) \\ 
-\sin(\Delta \cdot \theta_i) & \cos(\Delta \cdot \theta_i) 
\end{bmatrix} 
\begin{bmatrix} 
PE_{(pos, 2i)} \\ 
PE_{(pos, 2i+1)} 
\end{bmatrix}
\end{aligned}
$$

- Rotation Angle: The rotation amount is $\Delta \cdot \theta_i$.
- $\theta_i$: This is the geometric frequency for a specific dimension $i$, defined as $\theta_i = \frac{1}{10000^{2i/d_{model}}}$.
- Position Independence: This rotation relationship depends only on the distance $\Delta$ and the dimension frequency $\theta_i$; it is completely independent of the specific absolute position $pos$.

<p align="center">
  <img width="290" height="220" alt="4205ebd7-61d1-4eea-9909-48f896bf64c3" src="https://github.com/user-attachments/assets/7159ded4-c0bc-4195-ad00-8edc537a7b1d" />
</p>

Code example:
```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Fixed Sinusoidal Positional Encoding module.
        
        Args:
            d_model: The dimension of the word embeddings.
            max_len: The maximum sequence length supported.
        """
        super().__init__()
        
        # 1. Create a matrix of shape (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create a column vector for positions (0, 1, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Calculate the frequency/division term
        # Use log space for numerical stability: 1 / 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. Apply sine to even indices (0, 2, ...) and cosine to odd indices (1, 3, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. Add a batch dimension (1, max_len, d_model) and register as a buffer
        # A buffer is part of the model state but not a trainable parameter.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Embeddings with positional information added.
        """
        # Add the positional encoding to the input embeddings
        # We slice self.pe to match the specific sequence length of x
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
```

**Advantages**
- It allows the model to distinguish the position of words in a sequence, which is critical for semantic understanding.
  
- Because these encodings are unique to each position, they help the model maintain global consistency across the entire sequence.
  
- It enables the self-attention mechanism to weigh information based on distance, improving the handling of long-distance dependencies.

**Disadvantages**
- Because it is not learnable, it may not adapt optimally to specific tasks or datasets.

- It may struggle to fully represent the features of extremely long sequences.

- In Tranformer, adding positional encodings linearly to word embeddings might cause some semantic information to be obscured or "overwritten."

### 1.3.4 Bucketed Relative Position Bias

While Sinusoidal Positional Encoding provides a fixed "coordinate" for every word, it treats position as an absolute value (e.g., "this word is at index 5"). Relative Positional Encoding (RPE) shifts the focus from where a word is to how far apart two words are.

Instead of adding a vector to the word embedding at the input layer, RPE modifies the Attention Mechanism itself. When calculating the attention score between a "Query" ($Q$) and a "Key" ($K$), the model injects a bias term or a transformation based on the distance $k = i - j$.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + \text{Relative Bias}}{\sqrt{d_k}}\right)V$$

In the context of Multi-Head Attention, each individual attention head is assigned its own unique set of bias terms.

In the T5 (Text-to-Text Transfer Transformer) model, Google introduced a simplified version of relative positional encoding called Bucketed Relative Position Bias. Instead of calculating a unique bias for every possible distance, it groups distances into "buckets." The fundamental idea is that precise distance matters more for nearby tokens than for distant ones. T5 uses a non-linear mapping to assign a distance $d = (i - j)$ to a specific bucket index:

- Small Distances: For tokens very close to each other (e.g., 0 to 7 tokens apart), T5 assigns a unique bucket for every single integer distance.

- Large Distances: As the distance increases, T5 uses logarithmic growth to group ranges of distances into the same bucket. For example, the difference between 100 and 110 tokens away is treated as functionally the same, so they share a single bias value.

During the self-attention calculation, the distance between query $i$ and key $j$ is converted into a bucket index. The corresponding bias is then added directly to the attention score before the softmax:

$$\text{Attention Score} = \frac{QK^T}{\sqrt{d_k}} + \text{Bias}_{\text{head, bucket}(i-j)}$$

Code example:
```python
import torch
import torch.nn as nn
import math

class T5RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, n_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        
        # Learnable bias table: (num_buckets, n_heads)
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Maps a matrix of relative distances to bucket indices.
        """
        relative_buckets = 0
        n = -relative_position # T5 convention

        # 1. Split buckets between forward (positive) and backward (negative) distances
        num_buckets //= 2
        relative_buckets += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        # 2. Linear mapping for small distances
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # 3. Logarithmic mapping for large distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / 
            math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        
        # Cap the value at the maximum bucket index
        val_if_large = torch.min(val_if_large, torch.full_as(val_if_large, num_buckets - 1))

        relative_buckets += torch.where(is_small, n.to(torch.long), val_if_large)
        return relative_buckets

    def forward(self, query_len, key_len):
        """
        Produces the bias matrix to be added to the attention scores.
        Returns shape: (1, n_heads, query_len, key_len)
        """
        # Create grid of positions: [query_len, 1] and [1, key_len]
        q_pos = torch.arange(query_len, dtype=torch.long)
        k_pos = torch.arange(key_len, dtype=torch.long)
        
        # Matrix of relative distances (i - j)
        rel_pos = k_pos[None, :] - q_pos[:, None] 
        
        # Get bucket indices for each pair
        buckets = self._relative_position_bucket(
            rel_pos, self.num_buckets, self.max_distance
        )

        # Lookup biases: (query_len, key_len, n_heads)
        bias = self.relative_attention_bias(buckets)
        
        # Permute to match attention shape: (1, n_heads, query_len, key_len)
        return bias.permute(2, 0, 1).unsqueeze(0)

# Example usage:
# model = T5RelativePositionBias(num_buckets=32, n_heads=8)
# bias_matrix = model(seq_len=128, seq_len=128)
# attn_scores = (Q @ K.T) + bias_matrix
```

### 1.3.5 ALiBi
ALiBi (Attention with Linear Biases) is a simpler, faster alternative to traditional positional encodings, introduced to solve the "length extrapolation" problem. ALiBi does not use any positional embeddings (absolute or relative). Instead, it adds a constant, non-learnable negative bias to the attention scores based on the distance between tokens. The attention score between a Query ($i$) and Key ($j$) is calculated as:

$$\text{Attention}_{i,j} = \text{softmax}(QK^T - m \cdot |i - j|)$$

- $|i - j|$: The linear distance between two tokens.
- $m$: A head-specific slope (a constant). Each attention head is assigned a different slope (e.g., $1, \frac{1}{2}, \frac{1}{4} \dots$), meaning some heads focus on local context while others have a "flatter" view of the whole sequence. For a model with $n$ attention heads, the slopes are typically set as powers of $2$. If the number of heads is a power of $2$, the set of slopes is defined by the following sequence:

$$m = \left( \frac{1}{2^1}, \frac{1}{2^2}, \frac{1}{2^3}, \dots, \frac{1}{2^n} \right)$$

<p align="center">
  <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/ae363985-94bb-432f-aebc-50d49459aae4" />
</p>

Code example:
```python
import torch

def get_alibi_slope(n_heads):
    """Returns a list of slopes for ALiBi."""
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start * ratio**i for i in range(n)]
    
    # Typically calculated as powers of 2 for stability
    return torch.tensor(get_slopes_power_of_2(n_heads))

def apply_alibi_bias(attn_scores, slopes):
    """
    attn_scores: (batch, n_heads, seq_len, seq_len)
    slopes: (n_heads, 1, 1)
    """
    seq_len = attn_scores.size(-1)
    # Create distance matrix: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # For causal models, we usually only care about the lower triangle
    context_position = torch.arange(seq_len).view(1, -1)
    memory_position = torch.arange(seq_len).view(-1, 1)
    relative_distance = torch.abs(memory_position - context_position)
    
    # Apply penalty: score - (slope * distance)
    bias = slopes * relative_distance * -1
    return attn_scores + bias.unsqueeze(0)
```

**Advantages**
- Superior Length Extrapolation: It allows models trained on short sequences (e.g., 1,024 tokens) to maintain stable performance and low perplexity on much longer sequences at inference time (e.g., 2x to 8x longer).
  
- Computational Efficiency: ALiBi requires zero learned parameters for positional encoding. 

**Disadvantages**
- Fixed Linear Decay: The penalty pattern is strictly linear and non-learned, which may be less flexible than learned positional embeddings that can adapt to specific datasets.
  
- Loss of True Global Attention: Because distant tokens are exponentially suppressed, the model may struggle with tasks where very long-range, non-local context is critical.

### 1.3.5 RoPE
Rotary Positional Embedding (RoPE) is a modern positional encoding technique used in state-of-the-art models like Llama, PaLM, and Mistral. It effectively bridges the gap between the fixed sinusoidal encodings you saw in your images and the flexibility of relative positional encoding.

RoPE encodes positional information by rotating the Query ($Q$) and Key ($K$) vectors in a high-dimensional space. It tries to find a transformation for Query ($q$) and Key ($k$) vectors such that their dot product depends only on the relative distance between positions $m$ and $n$. For a token at position $m$ and a feature pair $(x_1, x_2)$, the transformation looks like this:

$$
\begin{aligned}
\begin{bmatrix} 
q^{(m)}_{2i} \\ 
q^{(m)}_{2i+1} 
\end{bmatrix} 
&= \begin{bmatrix} 
\cos(m\theta_i) & -\sin(m\theta_i) \\ 
\sin(m\theta_i) & \cos(m\theta_i) 
\end{bmatrix} 
\begin{bmatrix} 
q_{2i} \\ 
q_{2i+1} 
\end{bmatrix}
\end{aligned}
$$

1. The Objective Function

We want to define a function $f(x, pos)$ that injects positional information into a vector $x$. The core requirement is that the inner product of two transformed vectors must be a function of their relative distance:

$$\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

2. The 2D Case

To solve this, we start in 2D space. Let the vector $x$ be represented as a complex number $z = x_1 + ix_2$. Applying a rotation by an angle proportional to the position $m$ can be written using Euler's formula:

$$f(x, m) = x \cdot e^{im\theta}$$

In matrix form, rotating a 2D vector $\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$ by angle $m\theta$ is:

$$
f(\mathbf{x}, m) = \begin{bmatrix} 
\cos(m\theta) & -\sin(m\theta) \\ 
\sin(m\theta) & \cos(m\theta) 
\end{bmatrix} 
\begin{bmatrix} 
x_1 \\ 
x_2 
\end{bmatrix}
$$

3. Proving the Relative Property

When we take the dot product (inner product) of two rotated vectors $q$ at position $m$ and $k$ at position $n$, we use the property that the dot product of 2D vectors is equivalent to the real part of $q \bar{k}$ in complex space:

$$
\begin{aligned}
\langle f(q, m), f(k, n) \rangle &= \text{Re}[ (q e^{im\theta}) \overline{(k e^{in\theta})} ] \\
&= \text{Re}[ q \bar{k} e^{i(m-n)\theta} ]
\end{aligned}
$$

The resulting expression depends only on the difference $(m - n)$. This proves that the attention mechanism will naturally perceive relative distances.

4. Generalization to D-Dimensions

Since the dot product is additive, we can generalize this to a $d$-dimensional space by splitting the vector into $d/2$ pairs of dimensions. Each pair $(x_{2i}, x_{2i+1})$ is rotated by its own frequency $\theta_i$:

$$\theta_i = 10000^{-2i/d}$$

The full transformation is a block-diagonal matrix:

$$
\text{RoPE}(\mathbf{x}, m) = \begin{bmatrix} 
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \dots \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \dots \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \dots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix} 
\begin{bmatrix} 
x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots 
\end{bmatrix}
$$

While RoPE is defined as an orthogonal matrix transformation $\mathcal{R}_m$, this matrix is highly sparse. Performing a standard direct matrix multiplication would be computationally wasteful. To save computational power, the operation is implemented using element-wise multiplication ($\otimes$) and vector addition rather than a full matrix dot product.

$$
\begin{aligned}
\text{Rotated Vector} &= \begin{pmatrix} 
q_0 \\ 
q_1 \\ 
q_2 \\ 
q_3 \\ 
\vdots \\ 
q_{d-2} \\ 
q_{d-1} 
\end{pmatrix} \otimes \begin{pmatrix} 
\cos m\theta_0 \\ 
\cos m\theta_0 \\ 
\cos m\theta_1 \\ 
\cos m\theta_1 \\ 
\vdots \\ 
\cos m\theta_{d/2-1} \\ 
\cos m\theta_{d/2-1} 
\end{pmatrix} \\ + \begin{pmatrix} 
-q_1 \\ 
q_0 \\ 
-q_3 \\ 
q_2 \\ 
\vdots \\ 
-q_{d-1} \\ 
q_{d-2} 
\end{pmatrix} \otimes \begin{pmatrix} 
\sin m\theta_0 \\ 
\sin m\theta_0 \\ 
\sin m\theta_1 \\ 
\sin m\theta_1 \\ 
\vdots \\ 
\sin m\theta_{d/2-1} \\ 
\sin m\theta_{d/2-1} 
\end{pmatrix}
\end{aligned}
$$

Code example:
```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequency constants (cos, sin) as complex numbers.
    Llama uses complex numbers to simplify the 'rotation' math.
    """
    # 1. Generate frequencies for each pair of dimensions
    # Using the formula: theta_i = 10000^(-2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. Map frequencies across the sequence length (positions)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (seq_len, dim/2)
    
    # 3. Convert to polar form (complex numbers: cos + i*sin)
    # This represents the rotation matrix in a compact form
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies the Rotary Positional Embedding to queries and keys.
    """
    # 1. Transform real vectors into complex pairs
    # (batch, seq_len, heads, head_dim) -> (batch, seq_len, heads, head_dim/2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 2. Reshape freqs_cis to broadcast across batch and heads
    # Llama alignment: (1, seq_len, 1, head_dim/2)
    freqs_cis = freqs_cis.view(1, xq_complex.shape[1], 1, xq_complex.shape[-1])

    # 3. Perform the rotation via complex multiplication
    # (cos + i*sin) * (q_real + i*q_imag) = Rotated Vector
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Suppose we have a 2-dimensional embedding for the word "Apple":
- Vector ($q$): $[1, 0]$
- Position ($m$): 1 (the first word in a sentence)
- Angle ($\theta$): $90^\circ$ (for simplicity)

To encode the position, we rotate $[1, 0]$ by $1 \times 90^\circ$.
- Rotated Vector ($q^{(1)}$): $[0, 1]$

If "Apple" was at Position 2, we would rotate it by $2 \times 90^\circ$ ($180^\circ$ total).
- Rotated Vector ($q^{(2)}$): $[-1, 0]$

RoPE combines the strengths of both absolute and relative positional encodings while introducing unique benefits for long-sequence modeling. It reflects relative position characteristics at the attention score level while maintaining the "extrapolation" property (the ability to handle longer sequences than seen during training) similar to sinusoidal encodings. As the distance between the Query and Key ($m - n$) increases, the rotation angle increases, which naturally causes the inner product (attention score) to decrease. This allows the model to naturally reduce focus on distant elements. It maintains effectiveness over longer sequences or across different segments. Computationally, it only requires a simple rotation transformation before vector multiplication, making it highly efficient.

Unlike many relative positional methods that require an explicit attention matrix to function, RoPE injects information directly into the Query and Key vectors. This makes it compatible with Linear Attention, enabling efficient long-sequence processing. It is the default positional encoding for many large-scale open-source models (such as Llama) and multimodal models. Compared to original sinusoidal encodings, RoPE demonstrates significantly better performance when aligning long texts and handling extensive context windows.

### 1.3.6 Length Extrapolation

Choosing an encoding depends on specific task requirements and model scales:

| Feature | Sinusoidal (Standard) | Learnable Embedding | ALiBi | RoPE (Rotary) |
| :--- | :--- | :--- | :--- | :--- |
| **Sequence Length** | Limited; struggles with very long sequences | Hard limit; cannot handle sequences longer than training | Excellent; naturally extrapolates to much longer sequences | Excellent; best for long-context and long-text alignment |
| **Complexity** | Low; fixed formula with no learned parameters | Medium; adds significant parameter overhead for long sequences | Low; simple linear penalty added to attention scores | High; requires complex-number math or 2D rotation logic |
| **Implementation** | Element-wise **addition** to word vectors | Element-wise **addition** or concatenation | **Bias** added directly to the Attention Matrix | **Multiplicative** rotation using a sparse matrix trick |
| **Stability** | Moderate; linear addition can "overwrite" semantic info | Moderate; depends on training data distribution | High; consistent penalty regardless of scale | High; orthogonal transformation preserves vector length |

Most models use addition to combine position and word vectors because it is simple and does not change the embedding dimension. Sine and cosine functions allow models to learn periodic patterns, enabling them to maintain regularity even in unseen positions. For modern large language models, RoPE is generally preferred for its ability to align long texts and handle extensive context windows effectively.

**Length extrapolation** refers to the ability of a model to perform effectively on sequences longer than those it encountered during training. Ideally, if a model is trained on a sequence length of 512 tokens, an "extrapolating" model should be able to process 1024 or 2048 tokens at inference time without a significant drop in performance or accuracy. We need length extrapolation primarily because training on extremely long sequences is computationally expensive and often impractical. In real-world applications, models often need to process extremely long documents, books, extensive Wikipedia entries, or codebases consisting of thousands of tokens. These inputs frequently exceed the original training length limits of the model. Models with strong length extrapolation capabilities can seamlessly adapt to various input lengths. This removes the need to perform costly model fine-tuning every time a longer text is encountered.

Length extrapolation is difficult because models struggle with position values they have never encountered during training. Taking sinusoidal encoding or Rotary Positional Embedding (RoPE) as examples, if a model is only trained up to a maximum position $p = T_{train}$, any inference involving a test position $T_{test}$ where $T_{test} > T_{train}$ will result in a corresponding rotation angle ($\alpha_{T_{test}} = T_{test} \cdot \theta_i$) that is significantly larger than any seen during the training phase. Because the model has never "seen" such large rotation angles or sinusoidal values, the interpolation range is effectively stretched beyond its learned bounds, which typically leads to a significant decline in inference performance.

As summarized in the table above, ALiBi and RoPE in general suit long sequence better. However, these two methods still have their limitation. While RoPE combines the benefits of absolute and relative encoding and maintains the "extrapolation" properties of sinusoidal methods, at extreme lengths, position information can "oscillate" or become blurry, making distant positions hard to distinguish. Therefore, RoPE is effective for long-text alignment but has specific limits. ALiBi is often considered stronger for extreme length extrapolation, because it does not suffer from "oscillation" and it captures relative relationships well even as sequences grow very long. But it is slightly more specialized and primarily used when long-sequence handling is the top priority.

#### 1.3.6.1 NTK (Neural Tangent Kernel)

In the context of length extrapolation, NTK acts as a tool to describe how a neural network's output changes relative to its input under the assumption of infinite width. It can be understood as a measure of similarity between inputs; specifically, it helps maintain a "smoothness" or similarity in the network's behavior even when the input sequence grows much longer than what was seen during training. "NTK-Aware" methods aim to adjust or "remap" position indices during inference so that the attention mechanism's behavior remains consistent with the distribution seen during training. Instead of using raw position indices that would result in massive, unfamiliar rotation angles, it applies non-linear scaling to ensure the "effective position" perceived by the model stays within its originally trained range.

We need NTK-aware approaches to solve the fundamental breakdown that occurs during long-sequence inference. Without it, a model trained on length $T_{train}$ encounters rotation angles at position $T_{test}$ that are far larger than anything it "saw" during training. This leads to a severe drop in performance because the model cannot interpolate these new, extreme values. 

By keeping the effective position within the trained boundaries, NTK-aware methods allow the model to generalize its understanding to extremely long sequences without requiring the massive computational cost of training on those lengths from scratch.

- Linear Scaling: This is the simplest form of remapping. It compresses the entire target sequence length ($T_{test}$) into the original training length ($T_{train}$) using a constant ratio. When the current position $p$ reaches the new maximum ($T_{test}$), the output $f(p)$ perfectly equals the original maximum ($T_{train}$). While easy to implement and ensures rotation angles never exceed training limits, it is often sub-optimal because it shifts the distribution of all intermediate positions uniformly. Also, linear interpolation causes a significant drop in numerical precision, particularly for the smaller integer components of a position.

$$f(p) = p \cdot \frac{T_{train}}{T_{test}}$$

- Power/Logarithmic Function Scaling: These methods use non-linear functions to prioritize certain parts of the sequence, often growing faster at the start and slower at the end (or vice versa). By using functions like $ln(p+1)$ or $p^{\alpha}$, the model can apply different "compression rates" to different parts of the sequence. The goal is to keep position differences locally distinct while compressing them more aggressively at a global scale. This helps maintain NTK consistency—the similarity in attention patterns—across a much larger range of positions compared to simple linear scaling.

- Multi-Head (Frequency-Specific) Scaling: This is a more sophisticated approach that applies different scaling factors to different attention heads or frequency "bands." In RoPE, different dimensions correspond to different rotation "frequencies." More complex methods apply unique scaling coefficients to each $i$-th attention head. This allows the model to be more flexible, perhaps keeping high-frequency heads (local detail) sharp while stretching low-frequency heads (global structure) to handle the longer context.

Not all data and tasks are sensitive to "NTK-Aware" methods. However, if your application truly requires an ultra-long context and has strong attention dependencies in the later stages of the sequence, NTK-Aware is almost a mandatory choice. Advanced implementations use different scaling coefficients for different attention heads. This is common in some large models (where different extrapolation strategies are set for different frequency $\theta_i$), allowing the model to balance local and global attention patterns effectively.

**Advantages**
- Model Integrity: It allows for window extension without changing the model's underlying parameters or architecture, meaning it does not negatively impact a model that has already been well-trained.
  
- Theoretical Control: The methods have clear theoretical support, making the interpolation process controllable and predictable.
  
- Deployment Efficiency: It is an ideal solution for a rapid "context window upgrade" before a model is deployed.

**Disadvantages**
- Structural Compatibility: These methods have specific requirements regarding the NTK properties of the model; consequently, they cannot be used with all types of model architectures.

- Risk of Boundary Effects: If the interpolation method is not handled correctly, it can result in "boundary effects," such as weakening the model's ability to handle extreme long-distance dependencies.

- Limited Fine-tuning Range: In practice, there is very little room for adjustment; it is viewed primarily as a "fine-tuning" tool rather than a major structural change.

#### 1.3.6.2 YaRN
YaRN (Yet Another RoPE Extension) is an advanced upgrade for Rotary Positional Embedding (RoPE) designed to solve the "performance crash" that standard RoPE experiences when dealing with ultra-long sequences. 

Standard RoPE is highly stable for short sequences, but as length increases, it suffers from several issues. Long sequences push trigonometric functions into new cycles, making originally clear position information blurry. Different positions may be confused due to periodic repetition, preventing the model from accurately distinguishing relationships between distant tokens. These issues cause massive fluctuations in attention scores, leading to a significant drop in generation quality. YaRN acts as a "patch" or upgrade to fix these specific problems and allow models to handle much longer contexts reliably.

- Dynamic Interval Stretching: YaRN dynamically adjusts the spacing between positional encodings based on the actual sequence length to prevent the model from entering a "new cycle" of trigonometric values that it hasn't seen before. 

- NTK-by-parts (Segmented Encoding): YaRN realizes that different dimensions of the embedding represent different wavelengths. It treats "high-frequency" and "low-frequency" components differently to balance local and global consistency. High-frequency dimensions are not interpolated; instead, they are only slightly extrapolated to maintain local stability. Low-frequency dimensions undergo standard interpolation to ensure global consistency across the entire long sequence. This "divided" approach prevents high-frequency dimensions from becoming too "crowded" while allowing low-frequency ones to transition smoothly.

- Global Normalization (Temperature Adjustment): When calculating attention scores, YaRN introduces a temperature parameter $t$ to modify the softmax distribution of the logits. The temperature $t$ regulates the flatness of the attention distribution, which helps control the stability and randomness of the generated content. Simultaneously, the Query ($q$) and Key ($k$) embeddings are scaled by a factor of $\sqrt{1/t}$. This allows the model to better balance local and global information regardless of whether the sequence is short or long.

$$\text{softmax}\left(\frac{q_m^T k_n}{t\sqrt{d_k}}\right)$$

YaRN provides a significant upgrade to the model's capabilities with virtually no extra computational cost for either inference or training. The RoPE embeddings used in YaRN are pre-generated, allowing them to be reused across different tasks without recalculation. The core logic—specifically Dynamic Interval Stretching and Segmented Encoding (NTK-by-parts)—is processed in linear time, making it exceptionally fast. For popular models like the LLaMA series (LLaMA1 and LLaMA2), YaRN provides a specific formula to find the ideal scaling factor ($s$) for context expansion, which allows the model to maintain the best possible scaling ratio across varying context lengths:

$$\sqrt{1/t} = 0.1 \ln(s) + 1$$

**Advantages**
- Minimal Implementation Cost: YaRN requires very small changes to the model.
  
- High Computational Efficiency: The position embeddings are generated in advance and can be reused. Additionally, the logic for dynamic stretching and segmented encoding operates in linear time, ensuring high performance.
  

**Disadvantages**
- Implementation Complexity: To use YaRN, developers must manually rewrite the specific parts of the code responsible for RoPE.
  
#### 1.3.6.3 Dual-Chunk Attention

Dual-Chunk Attention is a technique designed to handle long sequences efficiently by dividing the input into two distinct types of chunks: Local Chunks and Global Chunks. It aims to balance the need for fine-grained local detail with the necessity of broad global context, preventing the "blurriness" often seen in standard long-context methods. To implement Dual-Chunk Attention:

1. Segment the Input Sequence: Divide the entire sequence into fixed-size blocks or chunks. This allows the model to process information in manageable units rather than one massive, computationally expensive block.

2. Apply Local Attention: Within each chunk, compute standard self-attention. This ensures that tokens can interact closely with their immediate neighbors to capture high-frequency, precise information.

3. Establish Global Connectivity: Select specific tokens (often the first token of each chunk or a compressed summary) to act as "global anchors." These anchors attend to each other across the entire sequence length, allowing information to flow between distant chunks.

4. Integrate Positional Scaling (YaRN/NTK): Apply scaling strategies like YaRN or NTK-aware interpolation to the positional encodings within these chunks. Use different scaling factors for the local and global components to ensure that local detail remains sharp while global positioning stays stable.

5. Global Normalization: Adjust the attention scores using temperature scaling to ensure the softmax distribution remains numerically stable across the combined local and global attention results.

**Advantages**
- High Computational Efficiency: It offers higher calculation efficiency, leading to a significant reduction in overall Attention complexity.
  
- Ultra-Long Context Support: Theoretically, there is no limit to the sequence length it can support, allowing for extremely long context processing.

- Strong Flexibility: It is highly adaptable, as parameters like chunk size and structure can be dynamically adjusted based on the specific task requirements.

**Disadvantages**
- Major Structural Changes: Implementing this method requires significant modifications to the model architecture.

- Risk to Global Understanding: If the strategy for connecting different chunks is poorly designed, it can severely impair the model's ability to maintain a global understanding of the sequence.

- Hyperparameter Sensitivity: Chunk size, step length, or connection density.

#### 1.3.6.4 Other Methods

- Base-conversion: The core idea is to use a larger base to represent more information within the same number of dimensions. Models are naturally good at generalizing relationships rather than just memorizing exact digits. Even if a value exceeds the training range, the model can still understand relative magnitudes because it focuses on relative patterns rather than strict numerical limits. This method is more concise than adding more dimensions. It avoids the need for retraining or structural changes, solving range expansion issues without causing the model to collapse due to increased complexity.

- Linear Attention: This method aims to reduce the quadratic computational complexity of standard attention to linear complexity. While it allows for processing extremely long sequences efficiently, it often comes at the cost of reduced expressive power compared to traditional Softmax attention.

- Recurrent Chunking & Memory Tokens: This approach enables the model to maintain a "notebook" or summary of previous blocks of text. By passing compressed summaries (memory tokens) from one chunk to the next, the model can retain a persistent state of past information without needing the full context window.

- Transformer-XL & Compressive Transformer: These architectures utilize recurrence and caching mechanisms to preserve memory across different segments. By storing previous hidden states in a cache, the model can attend to information outside its immediate window, effectively creating a longer "functional" memory.

- Sliding Windows & Model Fusion (e.g., Longformer): This hybrid strategy combines local "fine-grained" reading with global perception. It uses a sliding window for immediate local context and specific global attention patterns to capture the broader structure of the document.

- Architectural Outsourcing (Retrieval-Augmented Generation): Rather than trying to "remember" everything internally, this method externalizes memory. The model solves long-document problems by querying external databases or documents and retrieving relevant information only when needed.

## 1.4 Attention

Since there is an abundance of resources available on the topic of Attention, we will not repeat the basics of standard mechanisms here. Instead, this entry will focus on documenting and exploring some discussions in the field.

- Why attention?

The Attention mechanism solves the information bottleneck and long-range dependency problems that plagued earlier sequence models like RNNs and LSTMs. Unlike fixed-length vector representations, Attention allows a model to focus on the most relevant parts of the input for each specific output token, essentially "paying attention" to the right context at the right time. By removing the need for sequential processing, Attention enables models to process entire sequences at once, which is the foundational breakthrough of the Transformer architecture. As tasks grew more complex—from simple translation to summarizing entire books—the ability to selectively retrieve information became mandatory to maintain performance without losing critical details.

- Self-Attention v.s. Cross-Attention

| Feature | Self-Attention | Cross-Attention |
| :--- | :--- | :--- |
| **Input Sequences** | One (e.g., just the input text) | Two (e.g., input text + output text) |
| **Q, K, V Source** | Same source: $Q, K, V$ all from one sequence | Hybrid: $Q$ from target; $K, V$ from source |
| **Primary Use Case** | Contextual understanding & internal relations | Alignment, translation & information fusion |
| **Typical Models** | GPT, BERT, Llama | T5, Whisper, Stable Diffusion |

- Why multi-head?

By splitting the attention mechanism into multiple "heads," the model can simultaneously focus on different types of relationships within the data. Each individual head can learn to identify and extract different linguistic or structural features. Instead of trying to find one "average" relationship, the model looks at the sequence through multiple "lenses" at once, providing a much richer understanding of the context. A single attention head can cause information to become too concentrated, making it difficult for the model to capture multi-dimensional dependencies in complex contexts. By introducing multiple heads, the model lowers the risk of encountering these bottlenecks when learning intricate relationships.

Multi-head also allows different attention heads to perform calculations simultaneously. Compared to traditional sequential models like RNNs, this significantly boosts both training and inference speeds by capturing multiple dependency relationships at once.

In practical use, a single head's weights can become overly focused on specific terms, leading to over-reliance. Combining multiple heads creates a more balanced and smooth attention distribution, which improves the model's overall comprehension across diverse contexts.

Multi-head Attention (MHA) will assign $d_{model}$ to $h$ heads, each with $d_{model} / h$ dimensions. This is because processing attention in several smaller chunks is significantly more efficient than doing it all at once. If the model used the full model dimension ($d_{model}$) for a single $Q, K, V$ dot product, the computational complexity would be prohibitively high. By dividing the dimension into smaller heads, the scale of the inner product operations is reduced, making the overall calculation faster and more manageable for the hardware.

- Why do we "scale" each head?

The reason we "scale" each head (specifically the dot product of $Q$ and $K$) is primarily to prevent gradient vanishing during training. When the dimensionality of the head ($d_k$) is large, the dot product of the Query ($Q$) and Key ($K$) vectors can result in very large numerical values. Large input values push the Softmax function into its extreme regions, where the curve becomes almost flat. In these flat regions, the gradient (derivative) of the softmax becomes nearly zero. This leads to the vanishing gradient problem, making it almost impossible for the model to update its weights effectively during backpropagation.

Mathematically, if we assume the components of $Q$ and $K$ are independent random variables with a mean of 0 and a variance of 1, their dot product will have a mean of 0 but a variance of $d_k$. By dividing the dot product by $\sqrt{d_k}$, we scale the variance back down to 1. This keeps the distribution of the attention scores stable, regardless of how large the embedding dimension is. This is why the standard Attention formula includes the scaling factor:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Weights Sharing in Transformers

The primary motivation of weights sharing is to significantly decrease the total number of parameters in the model without necessarily hurting performance. Because a Transformer is essentially a stack of identical blocks (Self-Attention + Feed-Forward Networks), these layers are structurally compatible for sharing. Experiments have shown that sharing weights across different layers of the Encoder or Decoder does not significantly degrade the model's overall capabilities.

- Why three separate matrices for Q, K, and V?

Each matrix serves a unique purpose—$Q$ is the "question" (searching), $K$ is the "label" (matching), and $V$ is the "content" (the information being retrieved). Independent linear projections allow the model to map the same input into different vector subspaces, enabling it to capture more complex and nuanced relationships than a single matrix could. 

- Different roles of Q, K, and V

$Q$ (Query - The Searcher) represents the current word's "question" to the rest of the sequence. It is used to determine which other words in the context are relevant to the current one. $K$ (Key - The Index) serves as a "feature description" for every word in the sequence. It acts as a label that the Query ($Q$) checks against to see how well two words match. $V$ (Value - The Content) is the actual "information carrier". Once $Q$ and $K$ establish a relationship strength (the attention weight), the model extracts the corresponding content from $V$ to build the new representation.

In Transformer architectures, attention mechanisms have evolved to balance performance and efficiency. **Multi-Head Attention (MHA)**, the standard, processes information through multiple independent "heads" to capture diverse features in parallel. **Multi-Query Attention (MQA)** streamlines this by allowing all heads to share the same Keys and Values, significantly reducing memory and speeding up inference. **Grouped-Query Attention (GQA)** offers a compromise, where heads are divided into groups, and each group shares its own set of Keys and Values. 

<p align="center">
  <img width="1031" height="265" alt="462d596d-c07a-4a21-9e6f-4420a0b6206b" src="https://github.com/user-attachments/assets/89e3ad46-b368-4842-9d35-05fd7ebf4fdd" />

</p>

## 1.4.1 Multi-Head Attention (MHA)

Multi-Head Attention (MHA) is the core architectural component of the Transformer. Instead of performing a single attention function over the entire hidden dimension, MHA splits the queries, keys, and values into multiple "heads," allowing the model to attend to information from different representation subspaces simultaneously.

Code example:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension per head
        
        # Linear layers to project input into Q, K, and V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model) # Final output projection

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projection & Split into heads
        # Shape change: (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # Score = (Q * K^T) / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # Block illegal tokens
        
        attn_weights = F.softmax(attn_scores, dim=-1) # Normalize to probabilities
        output = torch.matmul(attn_weights, v) # Weighted sum of Values

        # 3. Concatenate heads back together
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.w_o(output) # Final linear layer
```

**Advantages**
- Multi-faceted Learning: Captures different types of relationships (e.g., syntax vs. semantics) in parallel.
  
- High Computational Efficiency: Multi-head also allows different attention heads to perform calculations simultaneously.

**Disadvantages**
- High Memory Usage: The $O(N^2)$ complexity relative to sequence length makes it memory-intensive.

- Inference Bottleneck: During generation, loading $K$ and $V$ tensors for every head slows down throughput.

## 1.4.2 Multi-Query Attention (MQA)

Multi-Query Attention (MQA) is an optimization of the standard Multi-Head Attention (MHA) designed to significantly speed up inference and reduce memory overhead. While MHA gives each query head its own dedicated Key ($K$) and Value ($V$) head, MQA uses a single Key and Value head that is shared across all Query heads. MQA is adopted by PaLM, StarCoder, and Gemini.

In a standard Transformer, $Q, K, \text{ and } V$ all have the same number of heads. In MQA, the input is projected into multiple Query heads (just like MHA). However, the input is projected into only one Key head and one Value head. All Query heads perform their attention calculation against that same shared $K$ and $V$ pair.

$$\text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right)V$$

Code example:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # MQA: Multiple Query projections
        self.w_q = nn.Linear(d_model, d_model)
        
        # MQA: Only ONE Key and ONE Value projection for all heads
        self.w_k = nn.Linear(d_model, self.d_k)
        self.w_v = nn.Linear(d_model, self.d_k)
        
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Q: (batch, num_heads, seq, d_k)
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # K, V: (batch, 1, seq, d_k) - Notice the '1' head
        k = self.w_k(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)

        # Scaled Dot-Product: K and V are automatically broadcasted across num_heads
        # Q: (B, H, S, D) * K^T: (B, 1, D, S) -> Scores: (B, H, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Output: (B, H, S, D)
        output = torch.matmul(attn_weights, v)
        
        # Restore original shape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.w_o(output)
```

**Advantages**
- Increased Throughput: Allows for much larger batch sizes and faster decoding speeds.
  
- Drastic KV Cache Reduction: Dramatically reduces the memory needed to store the KV cache during generation.

- Reduced Bandwidth: Lowers the amount of data moved from memory to the GPU/TPU cores.

**Disadvantages**
- Minor Accuracy Loss: Sharing keys/values across heads can slightly reduce the model's expressive power.
  
- Training Stability: Can sometimes be more sensitive during the initial training phases compared to MHA.

## 1.4.3 Grouped-Query Attention (GQA)

Grouped-Query Attention (GQA) is a hybrid attention mechanism introduced to strike a balance between the high performance of Multi-Head Attention (MHA) and the high efficiency of Multi-Query Attention (MQA).

In GQA, query heads are divided into $G$ groups. Each group shares a single pair of Key ($K$) and Value ($V$) heads. The core attention operation remains the same, but the $K$ and $V$ matrices are "broadcasted" (repeated) to match the number of query heads in each group:

$$\text{Attention}(Q_i, K_g, V_g) = \text{softmax}\left(\frac{Q_i K_g^T}{\sqrt{d_k}}\right)V_g$$

In modern Large Language Models (LLMs) like Llama 3 or Mistral, the number of groups ($G$) is often set to 8 because it represents a "Goldilocks" sweet spot: it is just enough to save a massive amount of memory without significantly hurting the model's intelligence.

Code example:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_queries, num_kv_groups):
        super().__init__()
        self.num_queries = num_queries
        self.num_kv = num_kv_groups
        self.group_size = num_queries // num_kv_groups # Q heads per KV head
        self.d_k = d_model // num_queries

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, num_kv_groups * self.d_k)
        self.w_v = nn.Linear(d_model, num_kv_groups * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, s, _ = x.shape
        
        # 1. Project and Reshape
        q = self.w_q(x).view(b, s, self.num_queries, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, s, self.num_kv, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, s, self.num_kv, self.d_k).transpose(1, 2)

        # 2. Repeat KV heads to match Q head count
        # (b, num_kv, s, d_k) -> (b, num_queries, s, d_k)
        k = torch.repeat_interleave(k, repeats=self.group_size, dim=1)
        v = torch.repeat_interleave(v, repeats=self.group_size, dim=1)

        # 3. Standard Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 4. Final Projection
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.w_o(out)
```

**Advantages**
- Efficient KV Cache: Reduces memory bandwidth and storage (e.g., an 8x reduction in KV cache size for Llama 2 70B).
  
- High Performance: Retains nearly the same accuracy and modeling quality as MHA.

- Faster Inference: Significant speedups in autoregressive generation (TTFT and throughput).

**Disadvantages**
- Architectural Complexity: Slightly more complex to implement than MHA or MQA.

## 1.4.4 Multi-head Latent Attention (MLA)

## 1.4.5 Linear Attention

## 1.4.6 Sparse Attention
