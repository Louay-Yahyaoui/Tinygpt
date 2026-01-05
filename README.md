# Tiny Transformer for Research Paper Text Modeling

## Overview

This repository contains a single Jupyter notebook `tinygpt.ipynb` implementing a small Transformer-based language model trained to generate short segments of research paper-style text. The project is exploratory and is intended to gain practical familiarity with Transformer architectures, tokenization pipelines, and end-to-end language model training in TensorFlow.

The model is deliberately minimal. Design decisions favor simplicity and inspectability over performance, scale, or completeness. The notebook combines data preparation, model definition, training, and basic text generation in one place.

## Data Sources

Two datasets are used for distinct purposes.

### Vocabulary Construction

- **Brown Corpus (NLTK)**  
  The Brown corpus is used only to construct a general English vocabulary. Tokens are extracted, cleaned, and indexed, and a small set of special tokens is added. The Brown corpus is not used for training or evaluation.

### Training and Validation Data

- **ACL-OCL Dataset (Hugging Face)**  
  Sentence-level text from NLP and computational linguistics research papers is used for model training. Sentences are tokenized, converted to integer sequences, padded or truncated to a fixed length, and split into training and validation sets.

The training objective is standard next-token prediction on fixed-length sequences.

## Model Architecture

The model is a small, decoder-only Transformer resembling a simplified GPT architecture. It consists of:

- Token embedding layer
- Positional encoding
- `n` Transformer blocks, each containing:
  - Multi-head self-attention
  - Residual connections
  - Layer normalization
  - Feed-forward projection
  - Dropout
- Final dense layer projecting to vocabulary size

## Parameters and Configuration

Key parameters defined in the notebook include:

### Tokenization and Vocabulary

- Vocabulary size derived from the Brown corpus
- Special tokens for padding, sentence start, and unknown words
- Fixed maximum sequence length

### Model Hyperparameters

- Embedding dimension
- Number of attention heads
- Hidden dimensions of attention and feed-forward layers
- Number of Transformer blocks
- Dropout rate

### Training Parameters

- Batch size
- Number of epochs
- Optimizer (Adam)
- Initial learning rate
- Number of warmup steps ( for learning rate decay )

### Generation Parameters

- Sampling temperature
- Simple stopping condition based on punctuation (`.`) similar to how training was conducted.

## Scope

This project is intended as a compact, self-contained example of implementing and training a Transformer language model on real academic text.
