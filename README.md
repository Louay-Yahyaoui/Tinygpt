# Tiny Transformer for Short-Form Story Modeling

## Overview

This repository contains a single Jupyter notebook `tinygpt.ipynb` implementing a small Transformer-based language model trained to generate short-form narrative text. The project is exploratory and is intended to gain practical familiarity with Transformer architectures, tokenization pipelines, and end-to-end language model training in TensorFlow.

The model is deliberately minimal. Design decisions favor simplicity and inspectability over performance, scale, or completeness. The notebook combines data preparation, model definition, training, and basic text generation in one place.

## Data Sources

A single dataset is used for both vocabulary construction and model training, with an explicit separation between training and validation data.

### Vocabulary Construction and Training Data

- **TinyStories Dataset**  
  The TinyStories dataset is used to construct the model vocabulary and to train the language model. The training set contains approximately 2.7 million short stories. A separate validation set is used for evaluation and contains approximately 1 percent of the training set size, or around 22,000 stories.

Text is tokenized and indexed to build a fixed vocabulary, with special tokens added for padding, unknown words, and sentence termination. The dataset is not randomly split within the notebook.

The training objective is standard next-token prediction on fixed-length sequences. Generation is terminated using a special end-of-sentence `<EOS>` token.

## Model Architecture

The model is a small, decoder-only Transformer resembling a simplified GPT architecture. It consists of:

- Token embedding layer
- Positional encoding
- `n = 6` Transformer blocks, each containing:
  - Multi-head self-attention
  - Residual connections
  - Layer normalization
  - Feed-forward projection
  - Dropout
- Final dense layer projecting to vocabulary size

The total parameter count of the model is approximately 58 million.

## Parameters and Configuration

Key parameters defined in the notebook include:

### Tokenization and Vocabulary

- Vocabulary size of 38,783 unique words plus special tokens
- Special tokens for padding, end-of-sentence (`<EOS>`), and unknown words
- Fixed maximum sequence length of 511 tokens  
  Stories exceeding this length are truncated

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
- Number of warmup steps (for learning rate decay)

### Generation Parameters

- Sampling temperature
- Generation stopping condition based on the `<EOS>` token

## Scope

This project is intended as a compact, self-contained example of implementing and training a Transformer language model on a narrative dataset.
