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
