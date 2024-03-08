# Introduction to Deep Learning and TensorFlow for NLP

## Overview

This project is an introduction to Deep Learning and Natural Language Processing (NLP) using TensorFlow. 
It demonstrates how to build a simple neural network for sentiment analysis on the IMDb movie review dataset. 
The goal is to classify movie reviews as positive or negative based on their content.

## Features

- Utilizes TensorFlow, a powerful machine learning library.
- Employs the IMDb movie review dataset for binary classification.
- Demonstrates data preprocessing, model architecture definition, model compilation, training, and evaluation.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your system.
- TensorFlow library installed. You can install it using pip:

```bash
pip install tensorflow
```

## Dataset

The IMDb movie review dataset is automatically downloaded via the TensorFlow Keras datasets module. 
It contains 50,000 movie reviews from the Internet Movie Database, split into 25,000 reviews for training and 25,000 reviews for testing. 
Each review is labeled as positive (1) or negative (0).

## Model Architecture

The model is a Sequential model with the following layers:

- **Embedding Layer**: Maps each word to a fixed-size vector of 16 dimensions.
- **GlobalAveragePooling1D**: Averages over the sequence dimension to reduce the dimensionality of the tensors.
- **Dense Layer**: Fully connected layer with 16 units and ReLU activation function.
- **Output Layer**: Single unit with a sigmoid activation function for binary classification.

## Usage

To run this project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the prerequisites installed.
3. Run the script using Python:

```bash
python path/to/script.py
```
