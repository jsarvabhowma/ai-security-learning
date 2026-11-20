# Unit 1: Agent Fundamentals

## 1. Transformers in LLM

### 1. Encoders

An encoder-based Transformer takes text (or other data) as input and outputs a dense representation (or embedding) of that text.

- **Example:** BERT from Google (Bidirectional Encoder Representations from Transformers). Reads an entire sentence at once to understand meaning, classify text, or find answers inside existing text.
- **Use Cases:** Text classification, semantic search, Named Entity Recognition
- **Typical Size:** Millions of parameters

### 2. Decoders

A decoder-based Transformer focuses **on generating new tokens to complete a sequence, one token at a time**.

- **Example**: Llama from Meta. Reads left-to-right to generate brand-new text word by word.
- **Use Cases**: Text generation, chatbots, code generation
- **Typical Size**: Billions (in the US sense, i.e., 10^9) of parameters

### 3. Seq2Seq (Encoder & Decoder)

A sequence-to-sequence Transformer _combines_ an encoder and a decoder. The encoder first processes the input sequence into a context representation, then the decoder generates an output sequence.

- **Example**: T5, BART
- **Use Cases**:  Translation, Summarization, Paraphrasing
- **Typical Size**: Millions of parameters

LLMs are typically decoder-based models with billions of parameters.

The underlying principle of an LLM is simple yet highly effective: **its objective is to predict the next token, given a sequence of previous tokens**. A "token" is the unit of information an LLM works with. You can think of a "token" as if it was a "word", but for efficiency reasons LLMs don't use whole words.

## 2. What is a Token & How It Works

- **Definition:** A token is the smallest unit of text an LLM reads and generates. It is not always a full word—it can be a **whole word**, a **part of a word (subword)**, a **single character**, or **punctuation/spaces**.
- **Rule of Thumb:** In English, **1 token ≈ 4 characters** or **¾ of a word** (100 tokens ≈ 75 words).
- **How Tokenization Works (Step-by-Step):**
  1. **Splitting (Tokenization):** Your text is broken into chunks using an algorithm (like _Byte-Pair Encoding / BPE_). Common words stay whole (`"apple"`), while rare or long words split into pieces (`"unbelievable"` → `"un"`, `"believ"`, `"able"`).
  2. **Mapping to IDs:** Every token in the model's vocabulary is assigned a unique number (Token ID), because neural networks only process numbers, not letters.
     - _Example:_ `"Hi there!"` → `["Hi", " there", "!"]` → `[15339, 1070, 0]`
  3. **Generation & Detokenization:** The model predicts the next Token ID one at a time, and the tokenizer converts those numbers back into readable text for you.

## 3. Special Tokens & EOS

- **What they are:** Hidden structural tokens added to prompts and outputs so the model knows **who is speaking** (system, user, assistant) and **where messages start/end**.
- **EOS (End of Sequence) Token:** The most important special token. It signals the model to **stop generating tokens** so it doesn't ramble forever.
- **Provider Differences:** Every model family uses its own specific EOS token format (chat template):
  - **GPT-4 (OpenAI):** `<|endoftext|>`
  - **Llama 3 (Meta):** `<|eot_id|>`
  - **DeepSeek-R1:** `<｜end▁of▁sentence｜>`
  - **SmolLM2 (Hugging Face):** `<|im_end|>`
  - **Gemma (Google):** `<end_of_turn>`
  - **Claude (Anthropic):** Not publicly documented (closed tokenizer; the API handles turn boundaries for you)

## 4. How an LLM actually picks the next word (The Decoding Loop/ Token predicition)

An LLM doesn't write a whole answer at once. It generates text one token at a time in a loop until it hits the **EOS** (stop) token. Each loop goes through 4 quick steps:

- **Read & understand the input:** First, it breaks the prompt into tokens and figures out both what each token means and where it sits in the sentence (since word order changes meaning).
- **Score every possible next token:** Next, it looks at its entire vocabulary (tens of thousands of tokens) and gives every single one a probability score for how likely it is to come next. For example, after _"The sky is..."_, `"blue"` gets a huge score, `"cloudy"` gets a decent score, and `"banana"` gets basically zero.
- **Pick one token (Decoding Strategy):** Now it has to choose from that ranked list. There are a few ways to do this:
  - **Greedy Decoding (The simplest way):** Just always pick whatever token got the #1 highest score. It's fast and predictable, but sometimes gets stuck on safe, repetitive phrases and misses a better sentence later on.
  - **Beam Search (Looking ahead):** Instead of blindly picking the #1 word at each step, it tracks several possible sentence paths at once and picks the whole sequence with the highest _overall_ score—even if one word along the way wasn't the #1 choice.
    - **`num_beams`:** How many parallel candidate paths to keep track of at once (setting this to `1` is just Greedy Decoding).
    - **`max_new_tokens`:** How many steps/tokens forward to generate before stopping.
    - **`length_penalty`:** Nudges the final score at the end without changing the search paths—**above `0`** favors **longer** answers, **below `0`** favors **shorter** ones.
    - **`num_return_sequences`:** How many of the top winning sentences to show at the end (must be $\le$ `num_beams`).
- **Add it and repeat:** It tacks the chosen token onto the end of the text, reads the whole thing again, and repeats the loop until the top choice is the **EOS** token.

**To see greedy decoding in action**, activate the course virtual environment, install the dependencies from `labs/requirements.txt`, then run `python labs/greedy_decode.py` from the `hugging-face-agents` directory. The script calls a Hugging Face Space that currently uses one beam with sampling disabled.

**To see beam search**, run `python labs/beam_search.py` from the same directory. It calls a different Hugging Face Space with four beams and prints the returned HTML and Markdown in the terminal.

## 5. Attention & Context Length (How an LLM focuses and remembers)

- **What is Attention?**
  When predicting the next token, not every word in a sentence matters equally. In _"The capital of France is..."_, words like **"capital"** and **"France"** carry all the real meaning, while _"The"_ and _"of"_ are mostly filler. **Attention** is the mechanism that lets the model spotlight and weigh the most relevant words—no matter how far back in the text they appeared.
  _(Named after the famous 2017 Google paper **"Attention Is All You Need"** that invented the Transformer.)_

- **What is Context Length (Context Window)?**
  This is the model's **maximum working memory / attention span**, measured in **tokens**. It is the total number of tokens the LLM can read and pay attention to at one time (covering your prompt, chat history, and its own generated reply).
  - Early models like GPT-2 could only handle ~1,024 tokens (a few paragraphs), while modern LLMs handle hundreds of thousands or millions of tokens (entire books or codebases).

## 6. Why Prompting Matters

- **What is a Prompt?**
  A prompt is simply the **input sequence of tokens** you feed into the LLM.
- **Why careful wording is so important:**
  Since an LLM's _only_ job is to predict the next token by using **Attention** to spot which words in your input matter most, the exact words you choose directly steer its probability scores.
  - **Vague prompts** leave many possible paths open, often leading to generic or off-target answers.
    - Example: `Write about dogs`
  - **Clear, specific prompts** give the model more guidance about the response you want.
    - Example: `List 3 low-shedding dog breeds for small apartments in a bulleted table`

## 7. How LLMs Are Trained (Pre-Training vs. Fine-Tuning)

Training an LLM happens in two main phases:

1. **Pre-Training (Learning Language & Patterns)**
   - **How it works:** The model reads massive datasets of raw text (books, websites, code) using **self-supervised / unsupervised learning**—meaning humans don't have to manually label the data.
   - **The training game:** It learns by playing "fill-in-the-blank" billions of times—either **predicting the next word** in a sequence (like GPT/Llama) or guessing **masked/hidden words** inside a sentence (like BERT).
   - **What you get:** A general "base model" that understands grammar, facts, and text patterns well enough to handle brand-new, unseen text—though it isn't yet trained to follow instructions or chat properly.

2. **Fine-Tuning (Learning Specific Tasks & Behaviors)**
   - **How it works:** The pre-trained model is trained further on smaller, curated datasets using **supervised learning** (clear input $\rightarrow$ desired output examples).
   - **What you get:** A specialized model tuned for real-world jobs—such as holding back-and-forth conversations, writing code, classifying text, or **using tools** (a key skill for AI Agents).

### Quick Cheat Sheet: The 4 Learning Types You'll See in LLM Docs

- **Self-Supervised Learning (Used in Pre-Training):** The data creates its own "answer key" without human labels (e.g., hiding the next word in a Wikipedia sentence and making the model guess it).

- **Unsupervised Learning:** The model looks for hidden patterns or groupings in raw, unlabeled data on its own (Self-Supervised learning is a modern cousin of this).
- **Supervised Learning (Used in Fine-Tuning / SFT):** A human or teacher dataset provides explicit **Input $\rightarrow$ Desired Output** pairs (e.g., _Question $\rightarrow$ Ideal Answer_ or _Prompt $\rightarrow$ Tool Call_).
- **Reinforcement Learning (RL / RLHF):** The model learns by **trial and error using rewards**—getting "thumbs up" points for helpful/accurate answers (or correct reasoning steps, like DeepSeek-R1) and penalties for bad ones.
