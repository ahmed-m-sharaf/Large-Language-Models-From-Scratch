# Large Language Models From Scratch
### A Deep Understanding of AI Large Language Model Mechanisms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/) 
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About the Repository
Welcome to the **Large Language Models From Scratch** repository! This project contains implementation notebooks, exercises, and code challenges corresponding to the course *"A deep understanding of AI large language model mechanisms"* (DULM).

The course offers an in-depth, hands-on journey from the absolute basics of tokenization and embedding math, building and training GPT-style transformers, conducting fine-tuning and chatbot alignment, up to cutting-edge AI Safety evaluations and Causal/Non-Causal Mechanistic Interpretability.

## Project Structure
The codebase is organized chronologically by parts. Code challenges and exercise notebooks are located inside the corresponding directories (e.g. `Part1_TokensEmbeddings/`). Each notebook has a corresponding identifier matching the syllabus below (e.g. `part1_text2num_text2numbers.ipynb`).

## Table of Contents
- [Introductions](#introductions)
  - [Introductions](#introductions)
- [Part 1: Tokenizations and embeddings](#part-1-tokenizations-and-embeddings)
  - [Words to tokens to numbers](#words-to-tokens-to-numbers)
  - [Embedding spaces](#embedding-spaces)
- [Part 2: Large language models](#part-2-large-language-models)
  - [Build a GPT](#build-a-gpt)
  - [Pretrain LLMs](#pretrain-llms)
  - [Fine-tune pretrained models](#fine-tune-pretrained-models)
  - [Instruction tuning](#instruction-tuning)
- [Part 3: Evaluating LLMs](#part-3-evaluating-llms)
  - [Quantitative evaluations](#quantitative-evaluations)
  - [Qualitative evaluations](#qualitative-evaluations)
- [Part 4: Overview of AI safety and mechinterp](#part-4-overview-of-ai-safety-and-mechinterp)
  - [AI safety](#ai-safety)
  - [Interpretability](#interpretability)
- [Part 5: Observation (non-causal) mech interp](#part-5-observation-non-causal-mech-interp)
  - [Investigating token embeddings (part 1)](#investigating-token-embeddings-part-1)
  - [Investigating neurons and dimensions](#investigating-neurons-and-dimensions)
  - [Investigating layers](#investigating-layers)
  - [Investigating token embeddings (part 2)](#investigating-token-embeddings-part-2)
  - [Identifying circuits and components](#identifying-circuits-and-components)
- [Part 6: Intervention (causal) mech interp](#part-6-intervention-causal-mech-interp)
  - [How to modify activations](#how-to-modify-activations)
  - [Editing hidden states](#editing-hidden-states)
  - [Interfering with attention](#interfering-with-attention)
  - [Modifying MLP](#modifying-mlp)
- [Part 7: Python tutorial](#part-7-python-tutorial)
  - [Python intro: Colab and notebooks](#python-intro-colab-and-notebooks)
  - [Python intro: Data types](#python-intro-data-types)
  - [Python intro: Indexing and slicing](#python-intro-indexing-and-slicing)
  - [Python intro: Functions](#python-intro-functions)
  - [Python intro: Flow control](#python-intro-flow-control)
  - [Python intro: Data visualization](#python-intro-data-visualization)
  - [Python intro: Strings and texts](#python-intro-strings-and-texts)
  - [Python intro: Pytorch](#python-intro-pytorch)
- [Part 8: Deep learning intro](#part-8-deep-learning-intro)
  - [Math of deep learning](#math-of-deep-learning)
  - [How models learn: gradient descent](#how-models-learn-gradient-descent)
  - [Essence of deep learning modeling](#essence-of-deep-learning-modeling)

## Course Syllabus & Takeaways
Click on any part to expand its sections, lecture listings, code files, and key learning points.

### Introductions

<details>
<summary><b>Show details for Introductions</b></summary>

#### Introductions

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 1 | [IMPORTANT] Prerequisites and how to succeed in this course | - | LLM architecture, training, and mechanisms are advanced topics. Have a positive attitude and embrace the challenge.<br>Taking notes by hand helps you learn more and remember better. Lecture notes are not available.<br>Having experience with coding, linear algebra, maching learning, and deep learning will help you excel in this course, although these topics are introduced in the course as they are necessary.<br>You can choose which lectures to watch and to skip, but keep in mind that knowledge and skills are cumulative. |
| 2 | Using the Udemy platform | - | - |
| 3 | Getting the course code, and the detailed overview | - | - |
| 4 | Do you need a Colab Pro subscription? | - | You can access GPUs for free on Google Colab, though compute time and RAM are limited.<br>Most of this course can be done on the CPU and limited GPU (free colab plan), but will be slower.<br>Paying to upgrade to Colab Pro will be convenient for many lectures but is not necessary.<br>You can upgrade to Pro and downgrade after the course. |
| 5 | About the "CodeChallenge" videos | - | You are the master of your education, and you should engage with this course in a way that best suits your skills and goals.<br>Use different difficulty levels for different videos.<br>Watch the code demos, even if you aren’t coding yourself. |

</details>

---

### Part 1: Tokenizations and embeddings

<details>
<summary><b>Show details for Part 1: Tokenizations and embeddings</b></summary>

#### Words to tokens to numbers

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 7 | Why text needs to be numbered | - | Text must be transformed into numbers before LLM.<br>A chunk of text is a “token” and can be a character, subword, or full word.<br>Embeddings are dense representations of tokens.<br>Tokenization and embeddings are learned from data, and there are many ways to create these schemes. |
| 8 | Parsing text to numbered tokens | [`part1_text2num_text2numbers`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_text2numbers.ipynb) | Text can be split into words via spaces, although this is not done in real tokenization.<br>Encoder and decoder functions are simple look-up tables.<br>Tokenization (encoding text using integers) is conceptually straightforward. |
| 9 | CodeChallenge: Create and visualize tokens (part 1) | [`part1_text2num_CCmakeATokenizer`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCmakeATokenizer_helper.ipynb) | Encoders and decoders are created using dictionary comprehension and functions.<br>The “context” of a token is its neighbors (before and possibly after); “context window” is the number of neighbors. |
| 10 | CodeChallenge: Create and visualize tokens (part 2) | [`part1_text2num_CCmakeATokenizer`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCmakeATokenizer_helper.ipynb) | “One-hot encoding” is a sparse tokenization, with one row per token and one column per vocab item. |
| 11 | Preparing text for tokenization | [`part1_text2num_preparingText4tokens`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_preparingText4tokens.ipynb) | Real text from the web is easy to import but a pain to clean…<br>Creating a tokenizing scheme is tricky and involves many choices with few clear optimal decisions.<br>Encoders and decoders are easy to create and use. |
| 12 | CodeChallenge: Tokenizing The Time Machine | [`part1_text2num_CCtimeMachineTokens`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCtimeMachineTokens_helper.ipynb) | Randomly generated tokens are usually nonsensical.<br>There is a mathematical relationship between word length and frequency (more on this later!), which has implications for LLM performance.<br>Tokenizers need special characters to deal with unknown tokens. |
| 13 | Tokenizing characters vs. subwords vs. words | - | Every tokenization scheme has advantages and limitations.<br>Current best tokenizers use a combination of characters, subwords, and words.<br>The vocab is learned based on statistical characteristics of human-written text. |
| 14 | Byte-pair encoding algorithm | [`part1_text2num_bytePairEncoding`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_bytePairEncoding.ipynb) | The BPE algorithm is based on replacing frequent token sequences with new tokens.<br>The basic BPE algorithm is simple and easy to implement.<br>Production-level tokenizers add several more steps to ensure accuracy, efficiency, and speed. |
| 15 | CodeChallenge: Byte-pair encoding to a desired vocab size | [`part1_text2num_CCbytePairEncodingLoop`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCbytePairEncodingLoop_helper.ipynb) | Even simple byte-pair encoding on a tiny datset creates tokens with preceeding spaces, just like professional tokenizers. |
| 16 | Exploring ChatGPT4's tokenizer | [`part1_text2num_GPT4tokenizer`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_GPT4tokenizer.ipynb) | OpenAI’s tokenizer is available, but is model-specific (e.g., GPT2 vs. GPT4).<br>Tokenizers use character- subword- and word-level tokens. Preceding spaces are part of tokens.<br>No text preprocessing is necessary! Just feed all the text into the tokenizer. |
| 17 | CodeChallenge: Token count by subword length (part 1) | [`part1_text2num_CCtokenEfficiency`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCtokenEfficiency_helper.ipynb) | Words and tokens differ in several ways, though they can overlap.<br>Words vary in their encoding efficiency, which is partly related to how often they appear in texts. |
| 18 | CodeChallenge: Token count by subword length (part 2) | [`part1_text2num_CCtokenEfficiency`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCtokenEfficiency_helper.ipynb) | Words and tokens differ in several ways, though they can overlap.<br>Words vary in their encoding efficiency, which is partly related to how often they appear in texts. |
| 19 | How many "r"s in strawberry? | [`part1_text2num_strawberry`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_strawberry.ipynb) | ChatGPT has difficulties with letter-based calculations because it represents words as tokens.<br>Asking ChatGPT to implement its calculations in python increases accuracy (same for math problems). |
| 20 | CodeChallenge: Create your algorithmic rapper name :) | [`part1_text2num_CCrapperName`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCrapperName_helper.ipynb) | Tokenization can be fun!<br>Tokenizers take lists, not ints, as input. |
| 21 | Tokenization in BERT | [`part1_text2num_BERT`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_BERT.ipynb) | Different tokenizers are optimized for different purposes.<br>BERT tokenizer has more words (vs. subwords) than GPT, and is therefore more human-interpretable.<br>BERT tokenizer by default adds special tokens before and after text. Don’t forget about this! |
| 22 | CodeChallenge: Character counts in BERT tokens | [`part1_text2num_CCbertChars`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCbertChars_helper.ipynb) | Professional tokenizers are easy to work with once you get used to them.<br>Tokenizers contain “special tokens” that you might want to filter out of analyses. |
| 23 | Translating between tokenizers | [`part1_text2num_tokenTranslation`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_tokenTranslation.ipynb) | Write your code for one specific tokenizer.<br>Choose the tokenizer (and LLM!) based on your goals. |
| 24 | CodeChallenge: More on token translation | [`part1_text2num_CCtranslatorFuns`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCtranslatorFuns_helper.ipynb) | The more familiar you are with tokenization, the easier it will be to understand embeddings and LLM mechanisms. |
| 25 | CodeChallenge: Tokenization compression ratios | [`part1_text2num_CCtokenCompression`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCtokenCompression_helper.ipynb) | Importing text data from the web is really easy.<br>Tokenization = compression? The primary goal of a tokenizer is to make text more efficient for LMs, but compression is a common byproduct due to redundancies in (some) written languages.<br>Token compression ratios are stable across different texts, with higher variability for text that includes code. |
| 26 | Tokenization in different languages | [`part1_text2num_languages`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_languages.ipynb) | Tokenization ≠ compression (but it often is).<br>Languages that have more complex written forms (e.g., morphemes in Chinese, richer morphology in Tamil) may require more tokens that characters.<br>Tokenization is less effective in languages they have less training data on. |
| 27 | CodeChallenge: Zipf's law in characters and tokens | [`part1_text2num_CCzipf`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_CCzipf_helper.ipynb) | Zipf’s law, a.k.a. power-law scaling, a.k.a. scale-free organization, a.k.a. fractal-like, a.k.a. self-similarity is a pervasive characteristic of biological and physical systems, and is taken as evidence of complex systems.<br>Modern computing tools and accessible digitized datasets allow you to explore nature in ways that were unthinkable until very recently. |
| 28 | Word variations in Claude tokenizer | [`part1_text2num_ClaudeVariations`](./Part1_TokensEmbeddings/text2numbers/part1_text2num_ClaudeVariations.ipynb) | Spaces are meaningful to humans, but are treated just like any other character to tokenizers.<br>Lots of subwords and words have preceding spaces.<br>Language models need a huge amount of data to learn all the ambiguities, errors, and varieties in human text. |

#### Embedding spaces

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 29 | Word2Vec vs. GloVe vs. GPT vs. BERT... oh my! | - | There are several word embeddings matrices that have different goals and applications, and are created in different ways. It is not trivial to compare them (more on this in the mech.interp section). The embeddings used in LLMs are not fixed, but instead are adjusted by the model based on context. |
| 30 | Exploring GloVe pretrained embeddings | `part1_embed_GloVe` | The GloVe embeddings can be used to study texts and relations between words.<br>The embeddings vectors are fixed once trained.<br>GloVe is not used in LLMs, partly because of the large and word-based vocabularies.<br>Cosine similarity has many applications in language modeling and computational linguistics. |
| 31 | CodeChallenge: Wikipedia vs. Twitter embeddings (part 1) | `part1_embed_CCwikiVsTwitter` | A lot of diversity across word embeddings matrices.<br>It is difficult or impossible to compare different embeddings matrices directly, but sets of relationships can be compared (see RSA in next section!).<br>Visualizing embeddings vectors often helps with interpretation. |
| 32 | CodeChallenge: Wikipedia vs. Twitter embeddings (part 2) | `part1_embed_CCwikiVsTwitter` | A lot of diversity across word embeddings matrices.<br>It is difficult or impossible to compare different embeddings matrices directly, but sets of relationships can be compared (see RSA in next section!).<br>Visualizing embeddings vectors often helps with interpretation. |
| 33 | Exploring GPT2 and BERT embeddings | `part1_embed_GPT2BERT` | All model parameters of publicly available LLMs are easily accessible.<br>GPT2 and BERT embeddings are not directly comparable, though model comparisons are possible (see next section).<br>LLM embeddings are not fixed! They are adjusted by attention and MLP layer as they pass through the LM. |
| 34 | CodeChallenge: Math with tokens and embeddings | `part1_embed_CCmathWithTokens` | Token indices are arbitrarily mapped to meaningful subwords (numbers).<br>“Token math” makes no sense (but can be fun :P ).<br>Embeddings vectors, however, can be manipulated mathematically — this is how LLM attention works!<br>Use LLMs for math theory/explanations, have it solve math problems by writing code to solve the problems. |
| 35 | Cosine similarity (and relation to correlation) | `part1_embed_cosineSimilarity` | Cosine similarity is one of the most commonly used measures of a relationship between two variables when working with embeddings and LLMs.<br>Cosine similarity is related to correlation: Both variance-normalize; Pearson additionally mean-centers.<br>Use the correlation to quantify a linear relationship in data that have different scales or mean offsets.<br>But when the variables are in the same scale and offsets are meaningful, use cosine similarity. This is often the case in LLM investigations.<br>When learning a new technical topic, try to code the math yourself; in applications, use established libraries if available and if it’s easier. |
| 36 | CodeChallenge: GPT2 cosine similarities | `part1_embed_CCcosineSimilaritiesGPT2` | Multitoken words present several challenges in LLM investigations (hint: give the model context and only analyze the final token).<br>Understanding the math of analyses makes you a better and more flexible data scientist. |
| 37 | CodeChallenge: Unembeddings (vectors to tokens) | `part1_embed_CCunembedding` | An “unembeddings matrix” is the conceptual inverse of the embeddings matrix (but not a literal inverse matrix).<br>The token with the largest unembeddings value is the next token in a generated sequence.<br>Generated text can quickly lose meaning, which was a major hurdle for language models to overcome.<br>The broad strokes mechanisms of next-token generation is simple conceptually and mathematically. |
| 38 | Position embeddings | `part1_embed_positionEmbeddings` | Language models use position embeddings to focus on tokens from various locations (“time points”) in the token sequence.<br>Predefined position embeddings are probably good enough for smaller models; modern architectures use learned embeddings.<br>Cosine similarity matrices can be tricky to interpret, but convey a lot of information. |
| 39 | CodeChallenge: Exploring position embeddings | `part1_embed_CCpositionExplorations` | The position embeddings matrix is complicated and impacts token processing in ways that are difficult to predict a priori.<br>Visual appearances (especially of apparent null effects) should be statistically evaluated using shuffled data before making strong interpretations.<br>The shuffling method here was too liberal; circular shifting and spectral phase scrambling are better. |
| 40 | Training embeddings from scratch | - | Embeddings matrices in LLMs are trained with the rest of the model. In the next several videos, however, we will train only and embeddings layer, to focus on the mechanisms of learning embeddings. |
| 41 | Create a data loader to train a model | `part1_embed_learnEmbeddings` | Preparing data to train models can become complicated (a common experience in many data fields…).<br>Simpler data organization methods are possible, but can be suboptimal for professional-grade model training. |
| 42 | Build a model to learn the embeddings | `part1_embed_learnEmbeddings` | Embeddings matrices are learned from text data using gradient descent and dimension-squeezing deep learning models.<br>The embedding dimension is preserved throughout the entire language model.<br>Language models generate text by concatenating one new token onto an existing token sequence. |
| 43 | Loss function to train the embeddings | `part1_embed_lossfunction` | Negative log likelihood is the main loss function used in language model training.<br>Log softmax increases sensitivity at small probabilities, and gives a stronger penalty for errors.<br>Most loss functions (and other mathematical bases of DL) are simple and well-defined. Difficulties arise from the explosive dimensionality of models. |
| 44 | Train and evaluate the model | `part1_embed_learnEmbeddings` | Language models are trained on a small number of epochs, because each epoch has a huge amount of data (batches).<br>GPU access will be increasingly important as the course progresses… |
| 45 | CodeChallenge: How the embeddings change | `part1_embed_learnEmbeddings` | The embeddings vectors expanded to fill up more of the embeddings space, reflecting encoding of high-dimensional information in a lower-dimensional space.<br>Interpretable semantic relationships (as measured through Sc) emerge even with a small amount of targeted training data, though the strength is weak. |
| 46 | CodeChallenge: How stable are embeddings? | `part1_embed_learnEmbeddings` | Embeddings vectors of the same token appear to be completely unrelated across repeated training runs.<br>Cosine similarity appears to be more consistent for related tokens.<br>Relative embeddings within a matrix are more interpetable than absolute vectors or across matrices.<br>The loss profile is relevant, but is just a tiny glimpse at what happens during training. |

</details>

---

### Part 2: Large language models

<details>
<summary><b>Show details for Part 2: Large language models</b></summary>

#### Build a GPT

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 48 | Why build when you can download? | - | Building a model from scratch is a fantastic way to learn how LLMs are created and trained.<br>Please don’t ever build an LLM from scratch again (except for more education) |
| 49 | Model 1: Embedding (input) and unembedding (output) | `part2_build_model1` | Saying that language models “use only the final token" for next-token prediction is not accurate. They  use all tokens; the final token contains the most information.<br>Next-token selection is probabilistic.<br>LLMs get complicated quickly; it’s good to learn about them one step at a time. |
| 50 | Understanding nn.Embedding and nn.Linear | `part2_build_embeddingVlinear` | nn.Embedding is a convenient wrapper for nn.Parameter. Use it to create embeddings matrices. |
| 51 | CodeChallenge: GELU vs. ReLU | `part2_build_CCreluVgelu` | GELU is the most common activation function in LLMs. It is more complicated than ReLU but smoother.<br>Many PyTorch functions are available as functions and classes; using one or the other is sometimes by necessity and sometimes a personal choice.<br>Measuring computation time is not trivial, because of optimized implementations, GPU fusing, overhead, etc. |
| 52 | Softmax (and temperature): math, numpy, and pytorch | `part2_build_softmax` | Softmax transforms LLM outputs from “raw” values (logits) into a probability distribution.<br>Temperature increases stochasticity, which helps language models produce more “creative” text. Typical temperatures are .5—1.5.<br>You’ll learn more about nuances of softmax theory and implementation in later videos. |
| 53 | Randomly sampling words with torch.multinomial | `part2_build_multinomial` | torch.multinomial directly maps token probability values onto selection probabilities.<br>Many PyTorch functions are sensitive to input data type — something to check if (when) you get errors!<br>Numpy and pytorch functions can sometimes produce identical results given proper input arguments, but often not by default. Be careful when translating between libraries. |
| 54 | Other token sampling methods: greedy, top-k, and top-p | - | There is no right or wrong token selection method.<br>Random sampling can increase response variability, which might be good for social chatting but bad for coding or legal documents.<br>The sampling method interacts with softmax temperature: Higher temperatures have stronger boosts of fewer tokens. |
| 55 | CodeChallenge: More softmax explorations | `part2_build_CCsoftmaxExtreme` | Softmax is mathematically simple, but some aspects of the transformation are apparent only for some numerical ranges.<br>The number of data values (e.g., vocab size) impacts the probability values, because they must all sum to 1. |
| 56 | What, why, when, and how to layernorm | `part2_build_layernorm` | Layernorm is simple and critical.<br>“Set it and forget it”: The learned parameters have little theoretical relevance other than preserving numerical stability throughout the model. |
| 57 | Model 2: Position embedding, layernorm, tied output, temperature | `part2_build_model2` | Position embeddings are added to the token embeddings, and help the model learn temporal patterns.<br>Embeddings are trained inside the model, not separately as with tokenization (c.f. previous section).<br>Tokens in a sequence are processed simultaneously, not in a for-loop (causality from attention mechanism). |
| 58 | Temporal causality via linear algebra (theory) | - | Time-causal attention can be implemented using a time vector in which the future is weighted zero while the past is weighted non-zero.<br>Causality can be implemented using matrices and softmax-probability, which is very computationally efficient on GPUs.<br>Causal attention is not necessary for LLMs, but improves next-token generation (good for chatbots). |
| 59 | Averaging the past while ignoring the future (code) | `part2_build_pastWithLinalg` | Matrix multiplication with masks are an efficient way to avoid for-loops.<br>PyTorch has optimized functions that fuse the attention algorithm, including the causal mask (next lecture!). |
| 60 | The "attention" algorithm (theory) | - | The “attention” mechanism is a clever way of pooling information across different embeddings vectors that allows for surrounding context to modify the current token transformation.<br>All analogies break down; some are useful (or at least entertaining).<br>The full LLM Transformer architecture is more complicated than just attention, but attention is a key aspect. |
| 61 | CodeChallenge: Code Attention manually and in Pytorch | `part2_build_CCattentionAlgo` | Q, K, and V matrices are trainable weights that are not changed during inference (applications).<br>Q, K, and V are the activations resulting from multiplication with token embeddings vectors.<br>“Highly optimized” functions can be hardware-specific.<br>Very powerful LLMs require specialized hardware, the sale of which is regulated by governments. |
| 62 | Model 3: One attention head | `part2_build_model3` | Attention adjusts (not replaces) the embeddings vectors as they pass through the model.<br>Different tokenizers (and also different pretrained models) use different terms and variable names. Understanding how LLMs work will help you identify features in a model. |
| 63 | The Transformer block (theory) | - | The Transformer block contains an attention sublayer and an MLP sublayer. Together, they calculate an adjustment to the token embedding to point towards an appropriate next-token embedding.<br>Expansion-nonlinearity-contraction is a typical MLP architecture for feature extraction and linear separability.<br>LLMs comprise dozens of Transformer blocks that learn different features and timescales of texts. |
| 64 | The Transformer block (code) | `part2_build_transformer` | You can now implement a GPT-style Transformer :)<br>Separating modules into callable classes helps keep code neat and organized.<br>Both Transformer sublayers comprise the operations copy → normalize → adjustment → add back to copy. |
| 65 | Model 4: Multiple Transformer blocks | `part2_build_model4` | Use nn.Sequential to create repeated instances of the same component of a model (with different weights matrices).<br>Specialization of Transformer blocks is not imposed by the architecture, but is thought to be an emergent property. We’ll discuss this more in the mechanistic interpretability sections. |
| 66 | Multihead attention: theory and implementation | `part2_build_multiheadAttention` | Multihead Attention (MHA) involves applying the attention equation to different slices of the QKV matrices.<br>MHA is thought to increase the richness and complexity of feature isolation and context-sensitivity, without increasing the number of trainable parameters.<br>Data from all heads are linearly combined via W0. |
| 67 | Working on the GPU | `part2_build_GPU` | GPUs are great at number-crunching, and can save a lot of time in LLM training and applications.<br>On the other hand, the CPU can be sufficient for a lot of explorations and investigations of LLMs.<br>Accessing and using a GPU might be expensive and requires additional code, so use it only when it's really beneficial. |
| 68 | Model 5: Complete GPT2 on the GPU | `part2_build_model5` | We just built a GPT2-small :)  although the weights are all random, so it’s not functional.<br>Commercial models, e.g., GPT4, have the same architecture and computations, but have more layers and parameters. But, quantitative increases can have qualitative impacts. |
| 69 | CodeChallenge: Time model5 on CPU and GPU | `part2_build_CCgpuVsCpu` | LLMs are basically worthless without a large number of dedicated high-end GPUs.<br>Regulating GPU access/sales is part of AI safety. |
| 70 | Inspecting OpenAI's GPT2 | `part2_build_OpenAIGPT2` | Publicly available pretrained models are really easy to use, explore, experiment with, and learn from. |
| 71 | Summarizing GPT using equations | - | There are many ways to understand a deep learning model; a combination of perspectives (diagrams, explanations, code, equations) is most beneficial.<br>Rewriting equations and checking matrix sizes can help you understand the flow and order of calculations.<br>Neither Q nor K directly contribute to the embedding vector adjustment; their combination determines which vectors in V guide next-token selection. |
| 72 | Visualizing nano-GPT | [Website](https://bbycroft.net/llm) | A neat website where you can visualize different GPT variants. |
| 73 | CodeChallenge: How many parameters? (part 1) | `part2_build_CCparameterCounts` | Biases are a tiny fraction of model parameters, which is why many deep learning models (especially models with layernorm) ignore them.<br>MLP layers are dense and can have 2-3x as many parameters as attention layers.<br>The number of parameters does not equate to importance in the model (cf layernorm).<br>Examining and dissecting models are useful skills. |
| 74 | CodeChallenge: How many parameters? (part 2) | `part2_build_CCparameterCounts` | Biases are a tiny fraction of model parameters, which is why many deep learning models (especially models with layernorm) ignore them.<br>MLP layers are dense and can have 2-3x as many parameters as attention layers.<br>The number of parameters does not equate to importance in the model (cf layernorm).<br>Examining and dissecting models are useful skills. |
| 75 | CodeChallenge: GPT2 trained weights distributions | `part2_build_CCweightsDists` | Characteristics of model weights often have smooth transitions across layers, reflecting shifts in representations and calculations.<br>In histograms, use counts for equal sample sizes, and use density (or other scaling) for unequal sample sizes. |
| 76 | CodeChallenge: Do we really need Q? | `part2_build_CClobotomizeQ` | Causal manipulations are easy to implement, although knowing what to manipulate is challenging and non-trivial.<br>Model evaluations are tricky. There are quantitative evaluation methods, but many evaluations are based on qualitative inspection of generated text. |

#### Pretrain LLMs

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 77 | What is "pretraining" and is it necessary? | - | Pretraining is a necessary first step for any LLM. A pretrained model understands the structure and patterns of written language, and can generate text.<br>Pretraining to create a useful modern base model is prohibitively expensive for most individuals and companies.<br>You should learn how pretraining works, but don’t try it at home ;) |
| 78 | Introducing huggingface.co | - | HuggingFace provides resources for downloading pretrained LLMs (and other models), training datasets, and more. We will some some of the free resources in this course. You do not need a HuggingFace login to access materials for this course. |
| 79 | The AdamW optimizer | - | L2 regularization in Adam involves updating and regularizing in one step (i.e., regularizing the update).<br>AdamW updates the weights first, then regularizes (i.e., regularizing the weights).<br>Without regularization, Adam==AdamW.<br>AdamW implements “constant shrinkage” instead of “adaptive shrinkage,” and empirically better in large models. |
| 80 | CodeChallenge: SGD vs. Adam vs. AdamW | `part2_pretrain_CCsgdVsAdams` | Simple mechanisms work better in simple models.<br>Adam is adaptive and more likely to stabilize.<br>Gradient accumulation speeds learning, but risks over-generalizing by pooling across more training data.<br>Gradient accumulation is only used for training very large models, and you probably will always want to reset the gradients. |
| 81 | Train model 1 | `part2_pretrain_model1` | Code to train LLMs has the same basic organization as code to train any deep learning model.<br>Even very simple models with little and limited training quickly learn text structure such as punctuation and line breaks. |
| 82 | CodeChallenge: Add a test set | `part2_pretrain_CCmodel1test` | Creating and evaluating a test set is not so difficult, but requires extra code.<br>Train/test splits are less important for pretraining LLMs, but is still good practice.<br>Additional subtleties about devsets, model vs. researcher overfitting, etc., are not discussed here. |
| 83 | CodeChallenge: Train model 1 with GPT2's embeddings | `part2_pretrain_CCmodel1withEmbeds` | Freezing weights is a common technique in deep learning for transfer learning or when re-training on limited data.<br>Freezing weights is not necessarily advantageous, especially in simple models or if the new training data differ from the previously trained data. |
| 84 | CodeChallenge: Train model 5 with modifications | `part2_pretrain_model5WithMods` | There are several ways to sample data, depending on how meticulous you want the procedure. Some overlap or skipping is less consequential with limitless training data.<br>Models learn to produce “language-looking” text very quickly. |
| 85 | Create a custom loss function | `part2_pretrain_customLoss` | Loss functions are extremely important for training deep learning models.<br>Loss functions should be as simple as possible, both mathematically and in code implementation.<br>Knowing how to create your own loss function gives you more control and flexibility over precise model designs and outcomes. |
| 86 | CodeChallenge: Train a model to like "X" | `part2_pretrain_CCtrainXbias` | KL divergence is a useful loss function for training distributions instead of individual tokens.<br>It is easy to train biases into models. This has major implications for fairness, misuse, persuasion or manipulation, cultural or political biases, marketing, and other AI safety topics. |
| 87 | CodeChallenge: Numerical scaling issues in DL models | `part2_pretrain_CCscaling` | Each matrix multiplication (dot products) increases the variance and numerical range of numbers.<br>This can have a negative impact on softmax probabilities by flattening the distribution.<br>Repeated normalization is very important for the stability of deep learning models including LLMs. |
| 88 | Weight initializations | `part2_pretrain_weightsInits` | Weight initialization is not important in small models, but is crucial for training large models including LLMs.<br>Weights should be initialized to small values, often proportional to the matrix sizes.<br>Bias terms are typically ignored in initializations, because there are so few bias terms.<br>Any initialization is important; the exact details and distribution shape seems to be less relevant. |
| 89 | CodeChallenge: Train model 5 with weight inits | `part2_pretrain_CCmodel5weightInits` | Now you know how to initialize weights :)<br>Examining changes in the model during learning is an approaching in mechanistic interpretability. Weights distributions tend to widen as the models learn more diverse patterns and representations. |
| 90 | Dropout in theory and in Pytorch | `part2_pretrain_dropout` | Dropout involves “switching off” units during training, and is thought to promote distributed representations.<br>LLM pretraining sets are so large and diverse that dropout is less important compared to, e.g., CNNs.<br>Dropout is more useful in fine-tuning when datasets are smaller and overfitting risk is higher. |
| 91 | Should you output logits or log-softmax(logits)? | - | Calculating log-softmax inside the LLM is often fine and convenient during training and classification, but outputting the raw values allows for more flexibility in subsequent applications. |
| 92 | The FineWeb dataset | `part2_pretrain_FineWeb` | HuggingFace provides several high-quality datasets for LLM training, including FineWeb (and several more specific variants). |
| 93 | CodeChallenge: Fine dropout in model 5 (part 1) | `part2_pretrain_CCmodel5dropout` | Dropout is conceptually simple, but adding it to code can be tricky.<br>There are many moving parts and parameters in an LLM; make sure to track variables and transformations.<br>Models can learn very fast on small homogeneous datasets, but take longer to train on datasets with more variability. |
| 94 | CodeChallenge: Fine dropout in model 5 (part 2) | `part2_pretrain_CCmodel5dropout` | Dropout is conceptually simple, but adding it to code can be tricky.<br>There are many moving parts and parameters in an LLM; make sure to track variables and transformations.<br>Models can learn very fast on small homogeneous datasets, but take longer to train on datasets with more variability. |
| 95 | CodeChallenge: What happens to unused tokens? | `part2_pretrain_CCunusedTokens` | Because softmax affects all token logits, not just the ones currently being processed, all tokens embeddings are trained inside the model, not just the ones that appear in the current token sequence.<br>This has implications for model output coherence and accuracy for tokens that are less commonly used in training datasets (e.g., obscure topics, new coding languages). |
| 96 | Optimization options | - | There are many ways to decrease computation time during LLM training.<br>Different strategies focus on the data, the model, or the hardware.<br>LLMs take so long to pretrain that even a few milliseconds improvement per batch can be significant. |

#### Fine-tune pretrained models

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 97 | What does "fine-tuning" mean? | - | Fine-tuning means making targeted adjustments to an existing pretrained model. It takes days or weeks instead of months, and has much smaller computational requirements.<br>There are myriad choices in fine-tuning; the challenge is from knowing what you want to do with the model. |
| 98 | Fine-tune a pretrained GPT2 | `part2_finetune_GPT2gulliver` | Shortcuts are great, but make sure you understand the mechanisms before oversimplifying your code.<br>Fine-tuning is easy to implement; the challenge comes from choosing the appropriate datasets.<br>The learning rate should generally be lower.<br>Model evaluation should rely on multiple qualitative and quantitative metrics. |
| 99 | CodeChallenge: Gulliver's learning rates | `part2_finetune_CClearningRateGulliver` | Smaller learning rates are generally preferred during fine-tuning.<br>There is a trade-off between preserving the base model and tweaking it towards the custom text. That balance depends on the application.<br>Quantitative metrics are important and useful, but obscure a lot. Qualitative assessments are crucial. |
| 100 | On generating text from pretrained models | `part2_finetune_HFgenerate` | The HuggingFace tokenizers and pretrained LLMs have many options for encoding and generating text. Some warning messages can be ignored, depending on your goal. |
| 101 | CodeChallenge: Maximize the "X" factor | `part2_finetune_CCboostX` | Fine-tuning can be tricky, because it is easy make a good base model be less useful.<br>Fine-tuning in practice often involves a lot of testing, explorations, and starting again.<br>KL divergence is a powerful loss function. Use it wisely ;) |
| 102 | Alice in Wonderland and Edgar Allen Poe (with GPT-neo) | `part2_finetune_2models2styles` | There are many pretrained models to use and explore. Different models have slightly different architecture, different parameters, and different training datasets, based on the specific goals of the org.<br>Models can easily adapt the writing style of text they are fine-tuned on. |
| 103 | CodeChallenge: Quantify the Alice/Edgar fine-tuning | `part2_finetune_CC2modelsQuantify` | It would be great to rely entirely on quantitative evaluation scores, but for generative models, numbers don’t capture the success of training or the application value. |
| 104 | CodeChallenge: A chat between Alice and Edgar | `part2_finetune_CCconvoAliceEdgar` | Concatenating token sequences, generating new tokens, and displaying only the newly generated tokens is a simplified implementation of chatbots.<br>Additional training is required to get the models to “understand” the difference between the user and them. |
| 105 | Partial fine-tuning by freezing attention weights | `part2_finetune_freezeAttention` | Freezing trainable parameters is toggled by the boolean requires_grad property of a tensor.<br>There are many options for what and when to freeze, and the specific benefits of each choice are not clear.<br>In practice, freezing involves a lot of testing, exploration, intuition, and mechinterp research. |
| 106 | CodeChallenge: Fine-tuning and targeted freezing (part 1) | `part2_finetune_CCfreezingNeo` | Precision freezing isn’t so difficult to implement; knowing which weights to freeze and why is challenging and uncertain.<br>Freezing has many trade-offs.<br>Hopefully, increased research on mechanistic interpretability will help guide targeted fine-tuning. |
| 107 | CodeChallenge: Fine-tuning and targeted freezing (part 2) | `part2_finetune_CCfreezingNeo` | Precision freezing isn’t so difficult to implement; knowing which weights to freeze and why is challenging and uncertain.<br>Freezing has many trade-offs.<br>Hopefully, increased research on mechanistic interpretability will help guide targeted fine-tuning. |
| 108 | Parameter-efficient fine-tuning (PEFT) | - | PEFT is a set of related techniques that allow you to fine-tune models with limited computational resources.<br>PEFT methods are very restrictive, and tend to work well when the task is well-characterized, e.g., classification.<br>Running and fine-tuning very large models still requires considerable resources! |
| 109 | CodeGen for code completion | `part2_finetune_CodeGen` | Always look for specialized pretrained models! They often come in multiple sizes (parameter counts).<br>You have now seen how code-completion AI tools work. |
| 110 | CodeChallenge: Fine-tune codeGen for calculus | `part2_finetune_CCcodeGenCalc` | To an LLM, any token sequence is just a token sequence; it is humans who make the distinction between spoken languages, code languages, time series, music, etc.<br>Code tends to comprise more short tokens and a smaller vocab, making training more challenging.<br>Before working on your application, look for the most relevant pretrained models. |
| 111 | Fine-tuning BERT for classification | `part2_finetune_bert4classification` | Base models are very versatile, and you can use them in many ways.<br>Procedures for importing datasets and models from 3rd party platforms can change; be prepared to modify your code.<br>Be mindful of the appropriate loss function; it’s not always NLLLoss. |
| 112 | CodeChallenge: IMDB sentiment analysis using BERT | `part2_finetune_CCbertClassifier` | Language models are very versatile and can be incorporated into myriad tasks that are based on processing text, code, or any other temporal sequence that can be tokenized.<br>Accuracy is a commonly used metric; also consider F1 score, ROC measures, etc. |
| 113 | Gradient clipping and learning rate scheduler (part 1) | `part2_finetune_clipScheduler` | Gradient clipping involves scaling down, without fundamentally changing, the gradient magnitude. It slows and stabilizes learning.<br>Learning rate schedulers are also designed to slow down and stabilize learning.<br>Sometimes, going slower means going faster (fewer errors and better solutions). |
| 114 | Gradient clipping and learning rate scheduler (part 2) | `part2_finetune_clipScheduler` | Gradient clipping involves scaling down, without fundamentally changing, the gradient magnitude. It slows and stabilizes learning.<br>Learning rate schedulers are also designed to slow down and stabilize learning.<br>Sometimes, going slower means going faster (fewer errors and better solutions). |
| 115 | CodeChallenge: Clip, freeze, and schedule BERT | `part2_finetune_CCbertClipScheduler` | Each additional step of LLM training is straightforward, but together they can present and overwhelming number of choices and decisions.<br>Consider your choices carefully, and be mindful that some choices (e.g., weight initializations) will impact other choices (e.g., gradient magnitudes). |
| 116 | Saving and loading trained models | `part2_finetune_saveAndLoadModels` | There are several ways to save and load models, depending on the environment and subsequent goals when re-importing. |
| 117 | BERT decides: Alice or Edgar? | `part2_finetune_bertWhichAuthor` | Training language models to classify text is easy — and easier to evaluate compared to generative models.<br>Data should be visualized and analyzed with as little manipulation as possible, but smoothing can help with interpretation. |
| 118 | CodeChallenge: Evolution of Alice and Edgar (part 1) | `part2_finetune_CCbertClassifiesAliceEdgar` | Some seemingly subjective evaluations can be made quantitative by using a separate trained classifier model.<br>However, this approach can produce deceptive results and false confidence, e.g., AI-writing detectors. |
| 119 | CodeChallenge: Evolution of Alice and Edgar (part 2) | `part2_finetune_CCbertClassifiesAliceEdgar` | Some seemingly subjective evaluations can be made quantitative by using a separate trained classifier model.<br>However, this approach can produce deceptive results and false confidence, e.g., AI-writing detectors. |
| 120 | Why fine-tune when you can use AGI? | - | SOTA commercial models are powerful, but cannot be downloaded or fine-tuned.<br>Customized fined-tuned models offer more flexibility and control, and are less resource-intensive.<br>But useful customized models may still require considerable resources to train. |

#### Instruction tuning

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 121 | Instruction tuning | - | Instruction tuning is the same procedure as fine-tuning (which is the same as pretraining). The main difference is the style and content of the datasets.<br>Instruction tuning is like training an “instruction manual” for how LLMs should interact with users, and constraints on their behaviors (e.g., not helping with illegal or immoral requests). |
| 122 | Some datasets for instruction tuning | `part2_instruct_datasets` | Instruction tuning datasets look like a desired chat.<br>The LLM learns keywords and tags like “question” and “answer”, or “system” and “user”.<br>The LLM learns to mimic the patterns in the training data, so if the training data look like a chat, the model will learn to chat. |
| 123 | Training a chatbot with system-user-assistant | - | Yet again: Many ways of training LLMs looks the same: Predict the next token in a sequence.<br>LLMs combine high-level patterns with detailed contextual and world-knowledge information.<br>Many token snippets are excluded (“hidden”) from the user during chats.<br>The system prompt is like an “instruction manual” that describes how the LLM can and should interact with users. |
| 124 | Instruction tuning with GPT2 | `part2_instruct_GPT2` | Training datasets for chatbots comprise human-written Q&A texts. The LLM learns to emulate the style and tone.<br>Chatbots don’t “answer” the user; they simply continue the pattern they recognize in the input.<br>Even GPT2-small can sound authoritative and encyclopedic; this has implications for AI safety. |
| 125 | CodeChallenge: Instruction tuning GPT2-large (part 1) | `part2_instruct_CCinstructingGPT2large` | LLMs train on such a large number of token sequences that imposing structure that helps us (humans) is not necessarily beneficial to the model.<br>LLMs learn to recognize, mix, and complete patterns in sequences, regardless of where those sequences start or end.<br>“Large” models are memory-intensive, not because of the models themselves, but because of all the intermediate activations matrices. |
| 126 | CodeChallenge: Instruction tuning GPT2-large (part 2) | `part2_instruct_CCinstructingGPT2large` | LLMs train on such a large number of token sequences that imposing structure that helps us (humans) is not necessarily beneficial to the model.<br>LLMs learn to recognize, mix, and complete patterns in sequences, regardless of where those sequences start or end.<br>“Large” models are memory-intensive, not because of the models themselves, but because of all the intermediate activations matrices. |
| 127 | Reinforcement learning from human feedback (RLHF) | - | RLHF is designed to align models to human values and human-like behavior, including helpfulness and truthfulness.<br>RLHF is a complicated process that involves gathering a lot of data and training intermediate models.<br>Despite its imperfections, RLHF works very well and is key to the success of modern chatbots. |

</details>

---

### Part 3: Evaluating LLMs

<details>
<summary><b>Show details for Part 3: Evaluating LLMs</b></summary>

#### Quantitative evaluations

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 129 | Promises and challenges of quantitative evaluations | - | Quantitative evals have the goal of assessing basic and advanced language capabilities and world-knowledge.<br>Evals are useful and important, but all tests have limitations.<br>Different methods target different skills or areas of knowledge, and it is useful to combine many tests.<br>Eval methods can be short-lived because LLMs can be fine-tuned to perform well on specific tests. |
| 130 | Numerical issues in logits and softmax | `part3_evals_numericalIssuesLogits` | Always check your intermediate and final results carefully, and be mindful of numerical issues.<br>Simple normalizations are often sufficient.<br>Many PyTorch functions internally safeguard against numerical issues, but confirm rather than assume. |
| 131 | Perplexity | `part3_evals_perplexity` | Perplexity is a simple measure of token prediction that can be calculated without access to model internals (need only final output logits).<br>The score is sensitive to analysis and dataset choices.<br>Perplexity also provides insights into datasets and sequences, not only into models. |
| 132 | CodeChallenge: Perplexing perplexities | `part3_evals_CCperplexities` | Token logits in pretrained models do not follow a uniform distribution.<br>Perplexity values depend on many factors. When evaluating models using perplexity, match as many parameters as possible (e.g., text, sequence length, stride), and interpret relative (not absolute) perplexity values. |
| 133 | Masked word prediction accuracy | `part3_evals_maskedPredictionsBERT` | Token prediction accuracy measures semantic and grammatical capabilities of LLMs.<br>Categorically incorrect responses should be evaluated for relevance; the purpose of language is to express meaning, not specific tokens. |
| 134 | HellaSwag | `part3_evals_hellaswag` | Sentence-level evaluations have advantages over token-level evals, including the ability to assess world knowledge and reasoning.<br>All evaluation methods have advantages and limitations, because both LLMs and human language are complex, ambiguous, and noisy. |
| 135 | Import large models using bitsandbytes | `part3_evals_bitsandbytes` | You can use the bitsandbytes library to import and use lightweight versions of large models.<br>Training should be done on the max-precision models if possible. |
| 136 | CodeChallenge: HellaSwag evals in two models (part 1) | `part3_evals_CChellaModels` | The success of the attention mechanism has led to increasing varieties of the basic GPT architecture.<br>There are many eval methods and none is perfect. It’s good to use several to compare and combine. |
| 137 | CodeChallenge: HellaSwag evals in two models (part 2) | `part3_evals_CChellaModels` | The success of the attention mechanism has led to increasing varieties of the basic GPT architecture.<br>There are many eval methods and none is perfect. It’s good to use several to compare and combine. |
| 138 | KL (Kullback-Leibler) divergence | `part3_evals_KLdivergence` | KL divergence is a measure of the distance between two probability distributions.<br>It opens the door to text-level and discourse-level evaluations of global text qualities (c.f. local token-level text features). |
| 139 | MAUVE | `part3_evals_mauve` | The MAUVE score ranges between 0 and 1, and reflects how much LLM-generated text “looks like” (has similar distribution characteristics to) human-written text. |
| 140 | CodeChallenge: Large and small MAUVE explorations | `part3_evals_CCmauveVariability` | A general statistical principle in science is to average multiple noisy measurements in hopes of more accurately assessing characteristics and capabilities.<br>Aggregated measures still pose challenges, however, particularly in terms of getting a sufficient amount of appropriate model-product text. |
| 141 | SuperGLUE and other amalgamations | - | That the diversity and imperfections of individual evaluation methods can be countered by aggregating results across many evals.<br>That there are many aggregated LLM evaluation aggregations. Some are more general while others are targeted for specific knowledge domains. |
| 142 | Assessing bias and fairness | - | LLMs can exhibit biases against certain groups.<br>These biases can be trained out to some extent via RLHF or data relabeling.<br>LLM biases are not formed “naturally,” but instead are learned from human text. |
| 143 | Non-technical benchmarks | - | Statistical measures of task performance are useful, but there are many other ways to think about the impact of LLMs from commercial and culture perspectives. |

#### Qualitative evaluations

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 144 | Black box evals | - | Black box evaluations are important and have revealed many serious safety risks of LLMs.<br>However, black box evals only identify existing issues that users happen to search for; they cannot predict potential risks or identify underlying mechanisms that create those risks. |
| 145 | Red-teaming | - | “Red-teaming” refers to trained security experts using adversarial attacks to discover security, privacy, or safety risks.<br>The advantages and limitations of red-teaming safety evaluations. |
| 146 | Accuracy, coherence, and relevance | - | Some qualities of language are subjective, yet are desired features of a language model.<br>“Human-in-the-loop” evaluation can be high quality, but increases costs and can be more variable.<br>LLMs can do “subjective evaluations,” but that risks low-quality training. |
| 147 | Distributions of hidden-state activations | `part3_evals_variousVisualizations` | The internals of the models can be extracted and explored. This is key to mechanistic interpretability and technical AI safety.<br>LLMs are sufficiently complex systems that having access to all the numbers (weights and activations) does not necessarily lead to understanding.<br>Mechanistic interpretability is a nascent and exciting field in research and AI safety solutions. |
| 148 | Heatmaps of tokens for qualitative inspection | `part3_evals_textHeatmaps` | Text heatmaps are compelling visualization tools that can facilitate understanding how LLM calculations relate to tokens.<br>Text heatmaps suffer from several issues, including scaling, selection bias, non-representative examples, and overinterpretation. |
| 149 | CodeChallenge: Visualize single-token predictions | `part3_evals_CCheatmapTokenPredictions` | The final output logits of token N predict token N+1.<br>Similarity of final outputs does not imply similarity of internal calculations and representations.<br>Text heatmaps are fun to look at, and can be insightful. |

</details>

---

### Part 4: Overview of AI safety and mechinterp

<details>
<summary><b>Show details for Part 4: Overview of AI safety and mechinterp</b></summary>

#### AI safety

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 151 | AI safety and alignment | - | AI safety involves technical and legal measures to guide AI development towards maximal benefits with minimal risk.<br>AI safety is increasingly important, yet receives relatively little funding and research attention.<br>Aligning AI to human values raises deep philosophical, legal, cultural, and historical questions. |
| 152 | Why can't AI just be safe and moral? | - | LLMs understand human morality and ethics as token sequence patterns, just like they understand how to write LinkedIn posts or any other pattern.<br>LLMs can be misused by bad actors, or can act in immoral ways on their own.<br>Moral decision-making may conflict with profit motives of AI development companies. |
| 153 | In-context and few-shot learning | - | ICL is the remarkable ability for LLMs to learn and accurately perform novel tasks without adjusting weights or fine-tuning.<br>ICL is a challenge for AI safety, because it is difficult to control for unsafe abilities that are not built into the model during training. |
| 154 | Scaling and AI safety | - | Scaling laws describe the past but cannot guarantee the future.<br>Future AI capabilities may vastly exceed current capabilities — or may be only marginally better.<br>Many people have many motivations to hype future AI capabilities beyond what may be realistic. |
| 155 | Hands-on: Hack an AI to steal a password! | - | A fun little demo where you try to convince an LLM to give you a secret password. |
| 156 | How to get involved in AI safety | - | AI safety is increasingly important, but lags far behind AI development.<br>There are many formal and informal ways to get involved in AI safety, all of which start from understanding LLM architectures and mechanisms. |

#### Interpretability

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 157 | What is "mech interp" (mechanistic interpretability)? | - | Mechanistic interpretability is the attempt to understand LLM mechanisms by reverse-engineering.<br>It is a surprising difficult problem due to the enormity and complexity of the models, and the lack of ground truth verification.<br>The field is new and struggles from reproducibility, concrete applications, and a reliance on empirical observations rather than theoretical foundations. |
| 158 | How does mech interp relate to AI safety? | - | Mechanistic interpretability will lead to a more precise understanding of LLMs and AI.<br>Improved interpretability could be beneficial and/or harmful for AI safety, depending on its success and impact on AI development. |
| 159 | Concepts and terms in mech interp | - | Terminology is an important part of learning a new area of knowledge.<br>Some terms and concepts are good to know in advance; others you learn on-the-fly. |
| 160 | Theoretical and empirical approaches in research and teaching | - | Both theory and data analysis are crucial for development of LLM understanding and AI safety.<br>This course is more focused on analysis methods, because they provide a solid and lasting foundation for LLMs and myriad other ML applications.<br>If you want to contribute to the mechanistic interpretability field, you can learn theories and speculative interpretations in blogs and papers. |
| 161 | General criticisms of mechanistic interpretability | - | There are several valid criticisms of mechanistic interpretability.<br>Criticisms should be embraced and taken as an opportunity to do better. The potential impact of mech interp is high, and literally everything that humans do can be criticized.<br>On the other hand, avenues of research that are clearly ineffective or non-reproducible should be learned from and left behind. |

</details>

---

### Part 5: Observation (non-causal) mech interp

<details>
<summary><b>Show details for Part 5: Observation (non-causal) mech interp</b></summary>

#### Investigating token embeddings (part 1)

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 163 | CodeChallenge: Cosine similarity (advanced) (part 1) | `part5_embeddings_CCcosineSimilarity` | Pytorch functions take some getting used to…<br>Cosine similarity matrices are compact representations of pairwise linear interactions. They can be used in bivariate and multivariate analyses (e.g., pattern analysis, clustering, SVD). |
| 164 | CodeChallenge: Cosine similarity (advanced) (part 2) | `part5_embeddings_CCcosineSimilarity` | Pytorch functions take some getting used to…<br>Cosine similarity matrices are compact representations of pairwise linear interactions. They can be used in bivariate and multivariate analyses (e.g., pattern analysis, clustering, SVD). |
| 165 | CodeChallenge: Cosine similarity in word sequences | `part5_embeddings_CCsequentialCosines` | Studying relationships in embeddings matrices can be interesting and thought-provoking, but static embeddings cannot capture the re-interpretations of words that is necessary for realistic communication. |
| 166 | CodeChallenge: Coloring cosine similarity | `part5_embeddings_CCheatmapsCossim` | Tokens that convey less contextual or semantic meaning tend to have shorter embeddings vectors. Shorter vectors are less impactful in dot product calculations.<br>Embeddings and cosine similarity can be used in many ways to reveal the structure of language (as it appears in the training set). |
| 167 | CodeChallenge: Can random embeddings be interpreted? | `part5_embeddings_CCinterpRandomEmbeds` | Interpreting patterns in noise is dangerously easy. Generalizing from patterns in small datasets is one of the most important functions of the nervous system…<br>Be cautious about interpreting findings in complex systems, especially when using small datasets and without proper statistical validation. |
| 168 | T-SNE projection and DBSCAN clustering (theory) | - | T-SNE and clustering can be insightful methods to group and characterize high-dimensional data.<br>Both methods are sensitive to parameter choices.<br>There are several dimension-reduction and clustering methods; it’s a good idea to test several. Don’t rely on one run from one method with one parameter setting. |
| 169 | T-SNE projection and DBSCAN clustering (Python) | `part5_embeddings_TSNEclust` | T-SNE and clustering can be insightful methods to group and characterize high-dimensional data.<br>Both methods are sensitive to parameter choices.<br>There are several dimension-reduction and clustering methods; it’s a good idea to test several. Don’t rely on one run from one method with one parameter setting. |
| 170 | CodeChallenge: cluster the "x" terms | `part5_embeddings_CCtSNE` | Selecting internal model representations by linguistic category or feature is easy and can be interesting.<br>Always inspect dimension-reduction and clustering results carefully before interpretations.<br>DBSCAN can be very to parameter settings, and it is often difficult to know what to set the parameters to. |
| 171 | CodeChallenge: Tokenize, embed, and cluster happy emojis | `part5_embeddings_CCemoji` | Emojis are multitoken characters, and have embeddings just like any other token.<br>Multitoken words bring ambiguity: Is averaging their component token embeddings valid?<br>If analysis results seem puzzling or wrong, it is possible that our assumptions are wrong. |
| 172 | RSA (representational similarity analysis) | `part5_embeddings_RSA` | RSA allows you to compare representations of tokens in different models, layers, or measurements (e.g., LLM vs. human brain).<br>There are many variants and extensions of RSA for specific data types or hypotheses.<br>The results of an RSA are limited to the items (e.g., token embeddings) used in the analysis. |
| 173 | CodeChallenge: Compare embeddings with RSA (part 1) | `part5_embeddings_CCembeddingsRSA` | Your toolkit of analysis methods is growing :)<br>More dimensions seems to be beneficial for increasing category selectivity of word embeddings (based on our small sample). |
| 174 | CodeChallenge: Compare embeddings with RSA (part 2) | `part5_embeddings_CCembeddingsRSA` | Your toolkit of analysis methods is growing :)<br>More dimensions seems to be beneficial for increasing category selectivity of word embeddings (based on our small sample). |
| 175 | CodeChallenge: Word2vec vs. GPT2 | `part5_embeddings_CCword2vecVsGpt2` | Cosine similarity is sensitive to mean offsets, which is a desired feature in some situations but can be a confound in other situations.<br>RSA indicates that relative word embeddings seem reasonably consistent in word2vec and GPT2. |
| 176 | CodeChallenge: Graph representation of cosine similarities | `part5_embeddings_CCgraphs` | Similarity metrics can be visualized in multiple ways.<br>Some visualization methods look nice but are not very insightful.<br>Many graphs can be created without libraries. Writing your own code can be elucidating but not necessarily advantageous. |
| 177 | Embeddings arithmetic and analogies | `part5_embeddings_analogyVectors` | Embeddings vectors are just vectors; any geometric or linear-algebra operation can be implemented.<br>“Semantic axes” based on vector arithmetic is compelling and simple, and seems to work in isolated cases (see also next two videos).<br>Assuming linearity can help with analyses and interpretations, but risks misinterpretation. |
| 178 | CodeChallenge: soft-coded analogies in word2vec | `part5_embeddings_CC300softAnalogies` | Language models are fun to work with ;)<br>Differences in training between word2vec and GloVe.<br>Analogy vectors created from simple arithmetic do capture meaningful semantic relationships. |
| 179 | Creating and interpreting linear "semantic axes" | `part5_embeddings_semanticAxesNorm` | Normalization, regularization, selection, and other (often post-hoc) techniques are often necessary to get satisfactory results.<br>Adding, subtracting, and scaling (normalizing) embeddings vectors is how the transformer attention mechanism works! |
| 180 | kNN for synonym-searching in BERT | `part5_embeddings_knnBERT` | kNN (k-nearest neighbor) is a simple and versatile algorithm that can be used for classification or nuclear sampling.<br>There are many measures of distance (similarity); some are comparable while others are not.<br>BERT embeddings seem to allow for distance-based synonyms. |
| 181 | CodeChallenge: BERT v GPT kNN kompetition | `part5_embeddings_CCknnBertGPT` | Many analyses in machine-learning and linear algebra can be applied to many types of data and for many hypotheses and explorations.<br>Spacing in tokens and normalizations can have a significant impact on results of analyses based on vector calculations (distance, similarity, correlation). |
| 182 | Research on translating embeddings spaces | - | There is active and ongoing research on translating between embeddings vectors in different models.<br>There are both conceptual and statistical challenges, and successful translations is not guaranteed.<br>If successful, it would imply some underlying universal space of language representation. |
| 183 | Singular value spectrum of embeddings submatrices | `part5_embeddings_SVDspectrum` | SVD is an advanced matrix decomposition that can be very insightful (e.g., PCA) but also requires more linear algebra background to fully appreciate.<br>SVD can be used to identify “semantic basis vectors” of a submatrix of embeddings for related tokens.<br>SVD linearly rotates the embeddings subspace; the basis vectors will be interpretable only if the data have structure to identify.<br>SVD is like a “data-informed weighted average” and should perform better than a simple vector average. |
| 184 | CodeChallenge: SVD projections of related embeddings | `part5_embeddings_CCSVDprojections` | SVD, like all other analyses of complex data, can be insightful or confusing (or both), and often raises more questions than answers.<br>LLM embeddings matrices are designed to be starting-points for subsequent modifications, so these analyses are likely to be more insightful when applied to internal and/or final model calculations. |

#### Investigating neurons and dimensions

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 185 | Activation maximization via gradient ascent (theory) | - | Activation maximization is a technique to discover features that activate individual neurons in a DL model.<br>It is built on questionable assumptions, but seems to work for vision models.<br>LLMs may violate the assumptions too strongly to provide meaningful insights into LLM mechanisms. |
| 186 | Activation maximization (code) | `part5_neurons_activationMaximization` | If you can specify a loss function, you can optimize it using gradient descent.<br>Some interpretability method that have proven insightful for some model architectures (e.g., CNNs) may not be useful for other architectures (e.g., LLMs).<br>Even techniques that fail to reveal mechanisms provide important insights and increase “forensic” and coding skills. |
| 187 | Activation maximization via data sampling | `part5_neurons_actMaxSampling` | Data sampling for activation maximization is conceptually simple and likely to be interpretable.<br>It has been used in computer vision models to understand how image features like color, shape, and category are represented.<br>Scaling issues and context-dependency makes this method less promising for LLMs, especially without clear hypotheses to minimize the search space. |
| 188 | CodeChallenge: Reproducibility of activation maximization | `part5_neurons_CCactMaxRepeat` | Individual dimensions in an LLM don’t seem to be “tuned” to specific tokens.<br>Although we haven’t ruled out that this happens at a finer resolution (e.g., inside attention or MLP), lack of single-unit specificity is a common finding in DL.<br>This is consistent with the very large vocab and pretraining procedure (e.g., dropout, weight decay). |
| 189 | Extracting activations using "hooks" | `part5_neurons_PytorchHook` | “Hooks” are functions you can implant into a PyTorch model, that allow you to access its activations during a forward pass.<br>Hooks can slow performance, and so should be used only when accessing internals; they can be removed during inference or when not needed. |
| 190 | Relation between hooks and output.hidden_states | `part5_neurons_hookVsHiddenStates` | The “hidden states” provided by HuggingFace are the final outputs of each transformer block, which can be reconstructed as the previous plus the attention and MLP adjustments.<br>Embeddings vectors pass through the LLM, and are adjusted by each transformer layer. Hooks allow you to access all the adjustments. |
| 191 | Clarification of final hidden_states output | `part5_neurons_hookVsFinalHS` | The final hidden_states layer is the final transformer block pushed through the final layer-norm transformation. For this reason, the final hidden_states layer may show different patterns of activity compared to earlier hidden_states. |
| 192 | CodeChallenge: Grammar tuning in MLP neurons? (part 1) | `part5_neurons_CCtuningNeo` | Identifying category-specific neurons is challenging, due to context-dependency of token processing.<br>Always confirm findings in out-of-sample datasets. |
| 193 | CodeChallenge: Grammar tuning in MLP neurons? (part 2) | `part5_neurons_CCtuningNeo` | Identifying category-specific neurons is challenging, due to context-dependency of token processing.<br>Always confirm findings in out-of-sample datasets. |
| 194 | CodeChallenge: Context-modulated activation in MLP | `part5_neurons_CCcontextComplexities` | LLM activations are highly context-dependent, which poses a challenge That internal activations depend on spaces in words (because those correspond to different tokens).<br>The differences between single-token and in-context token processing.<br>Why context is a crucial challenge in mech interp research. |
| 195 | CodeChallenge: Activation histograms by token length (part 1) | `part5_neurons_CCactivationByLength` | Processing in early transformer blocks is more closely related to “superficial” token features, such as length.<br>Repeating analyses over layers can be insightful.<br>Even in coarse-grained analyses, larger models do not necessarily behave like smaller models, which means the universality assumption needs to be carefully checked before being used in interpretation. |
| 196 | CodeChallenge: Activation histograms by token length (part 2) | `part5_neurons_CCactivationByLength` | Processing in early transformer blocks is more closely related to “superficial” token features, such as length.<br>Repeating analyses over layers can be insightful.<br>Even in coarse-grained analyses, larger models do not necessarily behave like smaller models, which means the universality assumption needs to be carefully checked before being used in interpretation. |
| 197 | CodeChallenge: Activation histograms by token length (part 3) | `part5_neurons_CCactivationByLength` | Processing in early transformer blocks is more closely related to “superficial” token features, such as length.<br>Repeating analyses over layers can be insightful.<br>Even in coarse-grained analyses, larger models do not necessarily behave like smaller models, which means the universality assumption needs to be carefully checked before being used in interpretation. |
| 198 | Dealing with multitoken word embeddings | `part5_neurons_multitokenWords` | For multitoken target words, analyze the final token in the word.<br>The first token of a multitoken word contains no information about the rest of the tokens, whereas the final token is modulated by all previous tokens.<br>Identifying the final token position in a sequence can be tricky. Check your code for accuracy! |
| 199 | CodeChallenge: Category-tuned MLP projections (part 1) | `part5_neurons_CCmultiTokenTtests` | Multitoken words require additional considerations; taking the final token is generally the best approach.<br>The multitude of category-tuned neurons implies distributed circuits of representations.<br>Confirming results in unseen (test) data is generally easier in LLMs compared to other domains, and should be done in research. |
| 200 | CodeChallenge: Category-tuned MLP projections (part 2) | `part5_neurons_CCmultiTokenTtests` | Multitoken words require additional considerations; taking the final token is generally the best approach.<br>The multitude of category-tuned neurons implies distributed circuits of representations.<br>Confirming results in unseen (test) data is generally easier in LLMs compared to other domains, and should be done in research. |
| 201 | Classification via logistic regression: theory and code | `part5_neurons_logisticRegression` | Logistic regression allows you to predict a binary category label using data such as activation values.<br>Logistic regression is a cornerstone analysis in machine-learning, including mechanistic interpretability.<br>Logistic regression can be implemented in several libraries including PyTorch (using gradient descent), although statsmodels provides the most information. |
| 202 | Logistic regression vs. t-test: assumptions and applications | `part5_neurons_logregTtest` | T-tests and logistic regressions can provide similar results, but are based on different assumptions and formulas.<br>Use a t-test to compare two groups, and a logistic regression to make single-sample predictions about category labels.<br>Data simulations and visualizations are powerful tools to understand statistics and machine-learning analyses. |
| 203 | Proper noun tuning in GPT2-medium | `part5_neurons_classifyProperNouns` | Logistic regression is a common tool in mechanistic interpretability to identifying “tuning” in neurons, circuits, and dimensions.<br>Multi-dimensional visualizations can help interpret and contextualize results — and identify problems or subsequent analyses.<br>Whenever possible, confirm findings in new samples to avoid improperly interpreting overfitting. |
| 204 | CodeChallenge: Negation tuning in MLP neurons (part 1) | `part5_neurons_CCclassifyNegationsMLP` | How to identify words in categories.<br>How to run mass-univariate classification analyses using logistic regression.<br>More ways of creating hooks.<br>That numbers (statistical results) don’t tell the whole story — multiple forms of visualization are necessary for proper interpretation. |
| 205 | CodeChallenge: Negation tuning in MLP neurons (part 2) | `part5_neurons_CCclassifyNegationsMLP` | How to identify words in categories.<br>How to run mass-univariate classification analyses using logistic regression.<br>More ways of creating hooks.<br>That numbers (statistical results) don’t tell the whole story — multiple forms of visualization are necessary for proper interpretation. |
| 206 | CodeChallenge: Negation tuning in MLP neurons (part 3) | `part5_neurons_CCclassifyNegationsMLP` | How to identify words in categories.<br>How to run mass-univariate classification analyses using logistic regression.<br>More ways of creating hooks.<br>That numbers (statistical results) don’t tell the whole story — multiple forms of visualization are necessary for proper interpretation. |
| 207 | CodeChallenge: Negation tuning in QVK neurons | `part5_neurons_CCclassifyNegationsQKV` | Taking the time to understand code is a wise investment, because you will likely reuse the same code many times.<br>Summary statistics and relations to token types are often similar in QVK and MLP neurons. |

#### Investigating layers

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 208 | Token-related similarities within and across Q, K, V matrices (part 1) | `part5_layer_cossimsQKV` | Cosine similarity and correlation coefficient can give very different results, depending on mean offsets.<br>There are lots of curious patterns inside LLMs! Keep a curious and courageous attitude when exploring DL models.<br>Q and K vectors tend to be more similar, and V tends to be more distinct. That’s consistent with their computational organization in the attention algorithm. |
| 209 | Token-related similarities within and across Q, K, V matrices (part 2) | `part5_layer_cossimsQKV` | Cosine similarity and correlation coefficient can give very different results, depending on mean offsets.<br>There are lots of curious patterns inside LLMs! Keep a curious and courageous attitude when exploring DL models.<br>Q and K vectors tend to be more similar, and V tends to be more distinct. That’s consistent with their computational organization in the attention algorithm. |
| 210 | CodeChallenge: Token-related similarities across layers | `part5_layer_CCcossimsQKVlayers` | Coarse-grained analyses can facilitate a “holistic” understanding of LLMs, but are less able amenable to targeted insights and hypothesis-testing.<br>Token embeddings vectors transition from the current token to a prediction about the next token, as it passes through the LLM.<br>Multiple visualizations can yield different insights. |
| 211 | Grouping and RSA in Q and K matrices | `part5_layer_semanticGroupsRSA` | RSA is a useful analysis for comparing similarities of token embeddings across matrices or layers.<br>Query vectors represent the current token (here, different in each sequence) while key vectors represent contextual tokens (here, identical in each sequence). |
| 212 | CodeChallenge: Laminar profile of RSA and category selectivity | `part5_layer_CClayerGroupingRSA` | Token selectivity varies by attention matrix and layer.<br>Different analyses of the same data (e.g., RSA vs. selectivity) can identify different patterns and reveal distinct insights. |
| 213 | "Effective dimensionality" analysis with PCA | `part5_layer_effectiveDimensionality` | “Effective dimensionality” is a versatile and generic analysis method that can be applied to many multivariate datasets.<br>Dimensionality provides insights into the complexity of the processing, because more complex calculations require more dimensions.<br>“Linear” must be interpreted cautiously in DL, because all linear interactions take place between a multitude of nonlinear transformations. |
| 214 | CodeChallenge: Dimensionalities in Pythia 2.3B | `part5_layer_CCdimensionalityPythia` | Effective dimensionality fluctuates over layers, as the models need more or less complex representations.<br>Many analyses in LLMs can give very different results when different texts are used. This needs to be carefully considered in mech interp research.<br>Deeper insights can be obtained in follow-up research to determine the “width” of the dimensions, and the nature of the calculations in those dimensions. |
| 215 | Mutual information: theory and code | `part5_layer_mutualInformation` | Mutual information is a non-negative measure that quantifies statistical dependencies between two data sources.<br>The mutual information value on its own does not distinguish between linear vs. non-linear dependencies, or between positive vs. negative relationships.<br>Mutual information is easy to implement in Python, but can be slow for many interaction pairs. |
| 216 | Pairwise mutual information through the LLM | `part5_layer_MIhiddenStates` | Mutual information can be easily integrated into a mechanistic interpretability analysis pipeline.<br>The parameter-sensitivity of MI means that relative values are more safely interpreted.<br>MI over tokens per dimension-pair is not necessarily the best approach for practical and interpretational reasons. |
| 217 | Mutual information vs. covariance | `part5_layer_MIvsCovariance` | Mutual information and covariance are related but distinct measures of bivariate dependencies.<br>Covariance is often preferred because it is the basis for many additional analyses including PCA, dimensional analysis, and clustering.<br>MI can be used when the relationships are nonlinear or the sign won’t be interpreted.<br>Translating the math into code introduces biases and risks numerical issues, but the bias is constant and it’s faster than Sklearn’s version. |
| 218 | CodeChallenge: Attention to coffee: MI and token distances (part 1) | `part5_layer_CCMIcoffee` | Token-pair-based MI is faster and more interpretable, and addresses some statistical issues like sample sizes.<br>Extreme activation values may be important for internal calculations, but create statistical estimation problems in MI and many other analyses.<br>MI is higher in early MLP layers and higher in later attention layers, reflecting a “passing off” of context-based processing.<br>Tokens further apart in the text generally have less common context, and therefore lower MI. |
| 219 | CodeChallenge: Attention to coffee: MI and token distances (part 2) | `part5_layer_CCMIcoffee` | Token-pair-based MI is faster and more interpretable, and addresses some statistical issues like sample sizes.<br>Extreme activation values may be important for internal calculations, but create statistical estimation problems in MI and many other analyses.<br>MI is higher in early MLP layers and higher in later attention layers, reflecting a “passing off” of context-based processing.<br>Tokens further apart in the text generally have less common context, and therefore lower MI. |
| 220 | CodeChallenge: Clusters in internal vs. terminal punctuation (part 1) | `part5_layer_CCpunctuationMI` | Well-done data science asks more questions than it answers ;)<br>Covariance and mutual information can give qualitatively different results and lead to different insights.<br>The MI calculation removes the scale of the data while covariance preserves scale. Covariances should be normalized (i.e., Pearson correlation) if comparisons are made across variables with different variances. |
| 221 | CodeChallenge: Clusters in internal vs. terminal punctuation (part 2) | `part5_layer_CCpunctuationMI` | Well-done data science asks more questions than it answers ;)<br>Covariance and mutual information can give qualitatively different results and lead to different insights.<br>The MI calculation removes the scale of the data while covariance preserves scale. Covariances should be normalized (i.e., Pearson correlation) if comparisons are made across variables with different variances. |
| 222 | The Logit Lens | `part5_layer_logitlens` | The Logit Lens analysis involves examining the predicted token from each transformer layer output.<br>It can be used to explore how LLMs transition from the current to the subsequent token.<br>Even when the prediction is categorically incorrect, the prediction can be context-appropriate.<br>There are several additional metrics that can be extracted and visualized, including token rank and KL divergence. |
| 223 | CodeChallenge: Logit Lens in BERT (part 1) | `part5_layer_CClogitBertZ` | Analyses often need minor modifications for different models with different architectures or naming conventions.<br>Transitions into the final token can be sudden, indicating a sharp “phase transition” in internal processing.<br>The Logit Lens approach can be adapted to other internal calculations and metrics. Check the original post for more ideas and applications. |
| 224 | CodeChallenge: Logit Lens in BERT (part 2) | `part5_layer_CClogitBertZ` | Analyses often need minor modifications for different models with different architectures or naming conventions.<br>Transitions into the final token can be sudden, indicating a sharp “phase transition” in internal processing.<br>The Logit Lens approach can be adapted to other internal calculations and metrics. Check the original post for more ideas and applications. |

#### Investigating token embeddings (part 2)

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 225 | Calculating rotations of embeddings vectors | `part5_token_embeddingsAngleChange` | Many concepts in linear algebra and geometry can be useful in mechanistic interpretability. The field of mech interp is still young and there is room for growth!<br>Random pairs of embeddings vectors tend to be roughly orthogonal.<br>Various features and metrics suggest smooth or discontinuous changes across the transformer blocks. How these all fit together remains a major challenge. |
| 226 | CodeChallenge: Laminar evolution of sequential angular adjustments | `part5_token_CCpairwiseAngles` | Angle changes between tokens are less interpretable than changes of the same token across layers.<br>When you see a curious or unexpected finding, check for coding bugs or data issues, then investigate by separating the data and lots of visualizations!<br>LLMs take a few tokens to “load in” a context. Therefore, the first few tokens should not be interpreted in analyses. |
| 227 | Path length and logit token prediction | `part5_token_pathlengthLogits` | Path length is an interesting metric that provides insights into how much the embeddings vectors change through the model.<br>Generalization can be difficult in LLMs. Universality is not guaranteed, and should not be assumed. |
| 228 | CodeChallenge: Residual stream decomposition of path lengths  (part 1) | `part5_token_CCpathlengthDecomposition` | Attention and MLP subblock adjustments are largely orthogonal. They provide unique information to adjusting the token embeddings vectors.<br>Both contribute to the embeddings adjustments, although MLP makes a stronger contribution (somewhat tautological).<br>The first token in a sequence “loads in” the context, and behaves differently from all subsequent tokens. |
| 229 | CodeChallenge: Residual stream decomposition of path lengths  (part 2) | `part5_token_CCpathlengthDecomposition` | Attention and MLP subblock adjustments are largely orthogonal. They provide unique information to adjusting the token embeddings vectors.<br>Both contribute to the embeddings adjustments, although MLP makes a stronger contribution (somewhat tautological).<br>The first token in a sequence “loads in” the context, and behaves differently from all subsequent tokens. |
| 230 | State-space trajectories through embedding space | `part5_token_trajectories` | High-dimensional vectors can be visualized and quantified using low-dimensional projections.<br>PCA projections are fast, linear, and reproducible, and based on the assumption “variance = relevance.” Other compression methods (e.g., tSNE) might be useful.<br>State-space trajectories can reveal dynamics and differences that might be otherwise obscured. |
| 231 | Parts of speech with SpaCy library | `part5_token_SpaCy` | The SpaCy library is fast and easy to use to detect part of speech (and many other applications).<br>Some text processing is necessary when categorizing POS from LLM-tokenized text.<br>Using SpaCy for POS identification in LLM mech interp may require additional considerations due to discrepancies between tokenizers. |
| 232 | CodeChallenge: Do nouns or adjectives have longer trajectories? (part 1) | `part5_token_CCadjNounTrajectories` | POS can be challenging to identify at scale (minimal human oversight) with different tokenizers.<br>Nouns tend to require longer-range contextual dependencies to interpret, whereas adjectives provide local context to neighboring words. |
| 233 | CodeChallenge: Do nouns or adjectives have longer trajectories? (part 2) | `part5_token_CCadjNounTrajectories` | POS can be challenging to identify at scale (minimal human oversight) with different tokenizers.<br>Nouns tend to require longer-range contextual dependencies to interpret, whereas adjectives provide local context to neighboring words. |

#### Identifying circuits and components

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 234 | What is a "circuit" in a DL model? | - | There is no single widely agreed-upon definition of a “circuit” in complex systems.<br>Working definitions allow for organizing data investigation techniques, but should not be interpreted too restrictively. |
| 235 | Isolating and investigating attention heads | `part5_circuits_attentionHeads` | Attention is highly selective; selectivity is created by having many negative attention scores.<br>Isolating and studying LLM components (e.g., attention heads) opens doors for insights and additional analyses.<br>KDE is a useful technique for estimating smooth pdf’s from sparse data, with applications in ML, signal processing, finance, Gaussian/stochastic processes. |
| 236 | CodeChallenge: Laminar profile of attention head weights | `part5_circuits_CCattentionDists` | Attention weights tend to be sparse, with most token pairs having low attention weights and few large weights.<br>More token pairs have higher attention weights towards later layers, as the model transitions to its final token predictions.<br>KDE is a non-parametric frequency estimator with many applications in many domains, and facilitates clear data visualization. |
| 237 | Are circuits clustered in low-dimensional space? | `part5_circuits_clustering` | Alexander Graham Bell: "there are no unsuccessful experiments; every experiment contains a lesson."<br>Research and theory development require knowing what works and what doesn’t work.<br>Dimension compression and clustering methods are useful for many analyses. |
| 238 | Sparse probing: theory and code | `part5_circuits_sparseLinearProbing` | Sparse probing is an interesting statistical approach to a small number of variables that predict a label.<br>There are several issues with logistic regression in large datasets with strong inter-variable correlations (multicollinearity); see next video.<br>Sparse probing is probably best done with a small subset of neurons. |
| 239 | Challenges with sparse logistic regression in large datasets | `part5_circuits_logisticRegressionChallenges` | Sparse logistic regression predicts labels; redundant information across variables may be suppressed even if those variables are individually meaningfully.<br>Selecting a smaller number of variables for the analysis can ameliorate the situation, but data selection can also introduce biases and other limitations. |
| 240 | "Latent" vs. "manifest" variables | - | Latent variables are often the most important to understand, yet cannot be directly observed.<br>Manifest variables can be combined in many ways to estimate latent factors.<br>Latent estimation is statistical and therefore involves uncertainty. Ground truth can be difficult or impossible to verify. |
| 241 | Sparse autoencoders: theory and code | `part5_circuits_SAEdemo` | Autoencoders are nonlinear compression methods that can find patterns distributed across manifest variables.<br>Interpreting autoencoder results can be challenging without ground truth, careful experiments with good controls, and statistical confirmations.<br>Simulating data is an effective way to explore the accuracy and robustness of analysis methods, because they provide ground-truth comparisons and ability to manipulate noise, sample size, and other parameters. |
| 242 | SAE in GPT2 learns about Hungarian Palinka | `part5_circuits_SAEinGPT2` | Autoencoders are a powerful architecture in DL, with applications in compression, denoising, computer vision — and interpretability.<br>Choosing a latent component for interpretation is challenging, because there are so many and they are not intrinsically sortable (cf PCA). Quantitative analyses and test datasets are necessary. |
| 243 | CodeChallenge: Laminar profile of autoencoder sparsity | `part5_circuits_CCSAElaminarProfile` | Laminar profiles of coarse-grained characteristics can be insightful, and complement in-depth analyses.<br>Activation density increases with transformer layer, indicating increasing complexity and distributed representations, consistent with dimensionality analyses.<br>Sparse autoencoders can be sensitive to parameter settings. |
| 244 | Non-orthogonal latent components via eigendecomposition (theory and demo) | `part5_circuits_GEDdemo` | Generalized eigendecomposition is a powerful and method for revealing subtle patterns embedded in multivariate data. It is used in signal processing, neuroscience, and machine-learning (e.g., Fisher discriminant analysis).<br>The application of GED to LLM data require additional considerations, but can lead to network discovery and novel insights. |
| 245 | Generalized eigendecomposition separates "him" from "her" in MLP | `part5_circuits_GEDinGPT` | GED is a powerful and targeted method to identify latent components (weighted combinations of neurons) in data with distributed representations.<br>Some considerations are required for very large datasets, such as preselection or PCA-compression.<br>GED risks overfitting training data; additional statistical confirmations are necessary. |
| 246 | CodeChallenge: GED for category isolation across layers (part 1) | `part5_circuits_CClaminarOutOfSampleGED` | Analyzing complex systems can be tricky; take it one step at a time.<br>GED is a powerful, flexible, and targeted method of identifying patterns embedded in multidimensional datasets.<br>Some results (e.g., effective dimensionality) emerge from other analyses, and provide opportunities for replication, extension, and linking to other literatures. |
| 247 | CodeChallenge: GED for category isolation across layers (part 2) | `part5_circuits_CClaminarOutOfSampleGED` | Analyzing complex systems can be tricky; take it one step at a time.<br>GED is a powerful, flexible, and targeted method of identifying patterns embedded in multidimensional datasets.<br>Some results (e.g., effective dimensionality) emerge from other analyses, and provide opportunities for replication, extension, and linking to other literatures. |

</details>

---

### Part 6: Intervention (causal) mech interp

<details>
<summary><b>Show details for Part 6: Intervention (causal) mech interp</b></summary>

#### How to modify activations

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 249 | Introduction to causal mech interp | - | Causal mechanistic interpretability involves altering model activations during inference or training.<br>It can be used to replace activations with another value or add noise, and evaluate counterfactuals (how the model would have behaved if the internals were different).<br>There are so ways to manipulate and measure activation patterns; having clear hypotheses is helpful.<br>Null effects in causality experiments can be difficult to interpret, because of distributed representations and compensation. |
| 250 | Activation editing: Code implementations | `part6_how2modify_activationManipulation` | Manipulating model activations is easy and only a minor change from measuring activations.<br>Some care needs to be taken to ensure code accuracy.<br>Incorporating print statements and other sanity-checks during development is recommended. |
| 251 | CodeChallenge: replacing attention, MLP, and hidden states | `part6_how2modify_CCchangingActivations` | Always visualize, print, and anything else to sanity-check that your code works as intended, before using it for research.<br>Different layers of the model require different code approaches because variable types can differ.<br>Changing activations in one layer impacts later layers, which can impact the final model output. |

#### Editing hidden states

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 252 | Downstream impact of early layer scaling | `part6_hiddenStates_scalingImpact` | Hooking is fun :D<br>The activations without any interference can be used as comparisons for qualitative and quantitative evaluations of causal interventions.<br>Early modifications can have compounding impacts on downstream layers, though they are somewhat corrected for in the final transformer block. |
| 253 | CodeChallenge: Hidden-state scaling and token loss | `part6_hiddenStates_CCscalingTargetLoss` | Having the unedited internal activations provides a baseline reference to evaluate the impact of manipulations.<br>Interpreting model output behavior (token selection) has high application value and validity, and should complement more detailed circuit-level analyses.<br>Global scaling has minimal impact on next-token prediction, and can make the model less variable (akin to temperature in softmax). |
| 254 | CodeChallenge: Noisy and shuffled BERT predictions | `part6_hiddenStates_CCbertNoisyPredictions` | Randomly shuffling dimensions is catastrophic to model performance.<br>Noisifying activations reduces their fidelity and increases stochasticity (token selection uncertainty).<br>Added noise from a different distribution might be valid. The signal vs. noise characteristics of LLM activations are not fully understood. |
| 255 | CodeChallenge: Measure and correct BERT's bias | `part6_hiddenStates_CCbertBiases` | Complex systems like LLMs are easy to adjust in specific examples, but difficult to adjust for all possible examples (including generative text that doesn’t yet exist).<br>Averaging embeddings vectors assumes linearity, but internal representations might be nonlinear — and the geometry might be layer-specific.<br>The success of linear mixing does imply that linear approximations are valid and useful. |
| 256 | Activation patching with indirect object identification | `part6_hiddenStates_patchingIOI` | “Activation patching” involves replacing activations from one sequence with those from another.<br>The indirect object identification test assesses a model’s ability to “copy” grammatical and direct-object information into a next-token prediction.<br>IOI results reveal a sudden transition in context processing around halfway through the model. |
| 257 | Skip a layer | `part6_hiddenStates_skipAlayer` | The input into transformer block t is the output of block t-1. You can replace the output with the input to skip an entire transformer layer.<br>This is a rather crude manipulation that is unlikely to reveal detailed insights, but does help understand the mechanics of working with hook functions. |

#### Interfering with attention

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 258 | Head ablation and token prediction | `part6_attention_ablateHeadTokenPrediction` | Yet again: manipulating LLM dynamics (here, layer-specific attention head activations) during inference is easy.<br>The target logits were suppressed, although the model still produced the categorically correct answer. In a complex, stochastic system, categorical accuracy may be too insensitive to be the sole metric.<br>Replacing an entire head with zeros may seem like it is out-of-distribution, but that needs to be empirically evaluated. |
| 259 | CodeChallenge: Token prediction after head ablations (part 1) | `part6_attention_CCtokenPredictionHeads` | There are many terms for the action of changing an activation value during forward pass (e.g., ablate, edit, impute, intervene, replace). Terminology tends to settle as fields mature and develop.<br>Small changes in manipulations (e.g., replacing with zeros vs. mean) can have inconsequential or significant downstream impacts. Such “deterministic chaos” is a hallmark of complex systems. |
| 260 | CodeChallenge: Token prediction after head ablations (part 2) | `part6_attention_CCtokenPredictionHeads` | There are many terms for the action of changing an activation value during forward pass (e.g., ablate, edit, impute, intervene, replace). Terminology tends to settle as fields mature and develop.<br>Small changes in manipulations (e.g., replacing with zeros vs. mean) can have inconsequential or significant downstream impacts. Such “deterministic chaos” is a hallmark of complex systems. |
| 261 | Impact of head-silencing on cosine similarity | `part6_attention_silencingCosineSimilarity` | Attention adjustments can have complicated downstream impacts at multiple scales of LLM calculations and representations.<br>Deeper insights into LLM mechanisms will come from incorporating multiple measurements into research. |
| 262 | CodeChallenge: Does GPT2 like pineapple pizza? | `part6_attention_CCpineapplePizza` | Causal mechanistic interpretability research can be extremely precise and detailed. It’s a good idea to start with hypotheses to help guide your manipulations and measurements. |
| 263 | Attention head patching in IOI | `part6_attention_IOI` | You can create and remove hooks on the fly as needed, not only when importing the model.<br>The IOI experiment is amenable to precise manipulations, including combinations of manipulations at different scales and locations of the model.<br>Attention heads provide only a minor adjustment to the embeddings vectors, and therefore their contribution to complex language tasks is important but subtle. |
| 264 | CodeChallenge: Head and token patching in IOI | `part6_attention_CCperHeadIOI` | Hooks can be created, implanted, and removed anywhere in the experiment. Let your hypothesis and experiment design guide the best coding strategy, not the other way around.<br>The more targeted the manipulation, the smaller the effect is likely to be. Make sure to have enough data (100s or 1000s of samples) to perform statistics and be confident about the results. |

#### Modifying MLP

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 265 | Successive median-replacement of MLP neurons | `part6_mlp_replacement` | The MLP expansion layer is very high-dimensional; statistical strategies are necessary to avoid dimensional-explosion problems.<br>The similarity between 10% vs. 90% replacement suggests that only a few high-activation neurons carry a lot of information. |
| 266 | Statistics-based lesioning MLP neurons | `part6_mlp_ttest2ablateMLP` | Using independent tests to select model features (neurons or dimensions) to manipulate is a useful way to deal with the “dimensionality explosion” problem.<br>Hook functions are versatile and can incorporate analyses — but keep in mind that complicated analyses will slow down the forward pass. |
| 267 | CodeChallenge: Laminar profile of MLP t-lesions | `part6_mlp_CCstatsMLPlaminar` | Using independent data to select neurons (or other model components) for manipulation can help prevent biased results.<br>Having good hypotheses, a controlled experiment, and targeted analyses does not guarantee positive results, especially with small sample sizes. |
| 268 | Explorations in subspace removal | `part6_mlp_removeSubspace` |  |

</details>

---

### Part 7: Python tutorial

<details>
<summary><b>Show details for Part 7: Python tutorial</b></summary>

#### Python intro: Colab and notebooks

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 270 | Creating, working with, and saving Colab notebooks | - |  |

#### Python intro: Data types

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 271 | Arithmetic and comments | `datatypes` |  |
| 272 | Variables | `datatypes` |  |
| 273 | Lists | `datatypes` |  |
| 274 | Booleans | `datatypes` |  |
| 275 | Dictionaries | `datatypes` |  |

#### Python intro: Indexing and slicing

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 276 | Indexing | `indexSlice` |  |
| 277 | Slicing | `indexSlice` |  |

#### Python intro: Functions

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 278 | Inputs and outputs | `functions` |  |
| 279 | The numpy library | `functions` |  |
| 280 | Getting help on functions | `functions` |  |
| 281 | Creating functions | `functions` |  |
| 282 | Copying (duplicating) variables | `functions` |  |
| 283 | Generating random numbers | `functions` |  |

#### Python intro: Flow control

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 284 | For loops | `flowControl` |  |
| 285 | If-else statements | `flowControl` |  |
| 286 | List comprehension (single-line loops) | `flowControl` |  |
| 287 | Initializing variables | `flowControl` |  |
| 288 | Enumerate iterables | `flowControl` |  |
| 289 | Zip multiple iterables | `flowControl` |  |

#### Python intro: Data visualization

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 290 | Plotting dots and lines | `visualization` |  |
| 291 | Subplot geometry | `visualization` |  |
| 292 | Making graphs look nice | `visualization` |  |

#### Python intro: Strings and texts

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 293 | String interpolation and f-strings | `text` |  |
| 294 | Importing text from the web | `text` |  |
| 295 | Processing text | `text` |  |

#### Python intro: Pytorch

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 296 | Working with classes | `pytorch` |  |
| 297 | Creating custom classes | `pytorch` |  |
| 298 | Datatypes, tensors, and dimensions | `pytorch` |  |
| 299 | Reshaping tensors | `pytorch` |  |
| 300 | Random numbers | `pytorch` |  |

</details>

---

### Part 8: Deep learning intro

<details>
<summary><b>Show details for Part 8: Deep learning intro</b></summary>

#### Math of deep learning

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 302 | Terms and datatypes in math and computers | - | Some important terms in linear algebra and python include scalar, vector, matrix, and tensor.<br>There are "types" of numbers and variables, which can have different meanings in math vs. coding. |
| 303 | Vector and matrix transpose | `part8_math_transpose` | "Transpose" is an operation in linear algebra that involves swapping rows and columns in a vector or matrix.<br>It's used a lot in linear algebra and deep learning, as you'll see in the next several videos. |
| 304 | Linear weighted combinations | `part8_math_linWeightedCombo` | Linear weighted combinations are simple yet foundational in deep learning (and countless other applications!).<br>Non-zero means do not necessarily bias the results.<br>Simulating data and examining distributions can be very insightful when learning math. |
| 305 | The dot product | `part8_math_dotProduct` | There are several notations of the dot product.<br>“Dot product” is just a fancy term for linear weighted combination.<br>Libraries have built-in functions to calculate the dot product.<br>PyTorch is more sensitive to datatypes than is numpy (a common source of errors!). |
| 306 | Matrix multiplication | `part8_math_matrixMultiplication` | Matrix multiplication is valid only when “inner” matrix dimensions match.<br>Matrix multiplication is just an organized collection of dot products.<br>There are multiple ways to calculate matrix multiplication in numpy and pytorch.<br>PyTorch functions can be sensitive to datatype. |
| 307 | Softmax | `part8_math_softmax` | "Softmax" is a mathematical operation used in deep learning to transform a set of numbers (often called "logits") into probabilities.<br>Softmax is a nonlinear transformation. Inputs can be negative, but all outputs are positive.<br>Language models use softmax to help select tokens (subwords). |
| 308 | Logarithms | `part8_math_log` | The logarithm is a nonlinear transformation of positive-valued numbers.<br>It is often used in machine-learning and optimization to "stretch out" small numbers such as p-values. |
| 309 | Entropy and cross-entropy | `part8_math_entropy` | Entropy is a mathematical measures of the "surprise" or unexpectedness of a system. It can be used as a measure of nonlinear variability.<br>In deep learning, entropy and cross-entropy are used as error signals to train language models to select words that are consistent with human text. |
| 310 | Min/max and argmin/argmax | `part8_math_argmin` | Min and max refer to the smallest (most negative) and largest values in a vector or matrix.<br>The "arg" prefix refers to the index in the matrix where the extreme value was observed. |
| 311 | Mean and variance | `part8_math_meanvar` | The mean is the average of a collection of numbers (sum divided by count). It is used as a measure of the most typical or representative number.<br>Variance is the amount of dispersion in the data. Larger variance means the data values are more spread out around the mean. |
| 312 | Random sampling and sampling variability | `part8_math_sampling` | Repeatedly taking data from a population will give different results. This is called "sampling variability."<br>Sampling variability is a foundation of statistics, and is used in language modeling to probabilistically select from among the most likely next tokens. |
| 313 | The t-test | `part8_math_ttest` | The t-test is a statistical evaluation that indicates whether the average of one group is different from the average of another group.<br>The t-test has countless applications; in this course, we'll use the t-test when evaluating model performance and investigating circuits and mechanisms in attention layers. |
| 314 | Derivatives: intuition and polynomials | `part8_math_derivatives1` | A derivative describes how a function changes. There are analytic (pure math) techniques for finding the derivative of a function, but you can also approximate it by calculating differences between neighboring points of a function.<br>Among other uses, derivatives are used to find the minimum values of a function.<br>Derivatives are central to training a language model: Represent the model's error as a loss function, calculate its derivative, and then use the derivative to find the model parameters that minimize the error. |
| 315 | Derivatives find minima | - | Because the derivative indicates whether a function is increasing or decreasing, it can be used to point towards function minima.<br>In deep learning including large language models, the functions and their derivatives are simple, but have so many variables that they cannot be fully calculated analytically. Therefore, estimation methods like gradient descent are used to find minima.<br>There are some potentially difficult issues like local minima and vanishing gradients; modern techniques have some ways of avoiding these kinds of issues to help obtain good models. |
| 316 | Derivatives: product and chain rules | `part8_math_derivatives2` | This video introduces two differentiation techniques for multiplied (product rule) and compound (chain rule) functions. Pytorch handles the implementational details of differentiation, but understanding more about derivatives will help you understand backprop. |

#### How models learn: gradient descent

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 317 | Overview of gradient descent | - | Gradient descent is the core algorithm that powers deep learning, by iteratively finding weights (parameters) that allow the model to learn structure in text data to classify and generate new text.<br>A "gradient" is another word for the derivative, and thus, gradient descent simply involves calculating the local derivative of the loss function, and nudging the weights in the direction that decreases the loss.<br>There are some modern techniques to help gradient descent perform well in large models, but the core method hasn't really changed in decades. |
| 318 | What about local minima? | - | The optimization goal of training deep learning models is to find the *global* minimum of the error function, but gradient descent can get caught in *local* minima.<br>Although this can be problematic in relatively small models, large network models rarely suffer from local minima, in part because there are so few (most apparent local minima are actually saddle points, where the function can go down a different direction).<br>Modern optimizers such as Adam can adaptively navigate other potential issues such as vanishing gradients. |
| 319 | Gradient descent in 1D | `part8_gradientDescent_1D` | Gradient descent in a 1D (single-variable) function provides a great visual intuition for how and why gradient decent works.<br>This demo also reveals why gradient descent is not guaranteed to provide the global optimal solution (although it will be a good approximation given some assumptions). |
| 320 | Gradient descent in 2D | `part8_gradientDescent_2D` | This videos shows how gradient descent is extended from 1D (single-variable function) to 2D (two-variable function). The principles and the math are the same, but the implementation and possibile outcomes are slightly more complicated. |
| 321 | CodeChallenge: fixed vs. dynamic learning rate | `part8_gradientDescent_CClearningrate` | Here you will explore the impact of learning rate regimes on model performance. |

#### Essence of deep learning modeling

| Video | Lecture Title | Code File | Key Take-home Points |
| :---: | :--- | :--- | :--- |
| 322 | The perceptron and ANN architecture | - | All deep learning models are built on the same foundational principles, which are many decades old.<br>New advances in hardware, datasets, and some architectural and regularization methods have allowed modern deep learning networks like LLMs to become impressively capable.<br>But understanding the "essence" of deep learning models as a perceptron will really help you learn about LLM mechanics. |
| 323 | A geometric view of ANNs | - | Deep learning models can be conceptualized using geometry. This helps build intuition, although the visualizations don't always generalize to really large models. |
| 324 | ANN math part 1 (forward prop) | - | Learn the math of the forward pass (hint: it's just a linear weighted combination + nonlinear activation). |
| 325 | ANN math part 2 (errors, loss, cost) | - | In this video, you'll learn about losses and costs -- the functions that deep learning models try to minimize during training. |
| 326 | ANN math part 3 (backprop) | - | The math of backprop. |
| 327 | Forward pass in Pytorch | `part8_essence_forwardpass` | Even the most complicated language models are simply dot product-calculators.<br>Models are created using classes and PyTorch methods.<br>“Tokenization” is the process of transforming text into numbers.<br>Models are useless without properly defined weights! |
| 328 | Backprop in Pytorch | `part8_essence_backprop` | PyTorch handles the low-level details of back-propagation.<br>Small, simple models that have analytic solutions and can be visualized are great for building intuition. |

</details>

---
