Here is a clean, professional README designed to look like a standard developer's documentation.

NLU Assignment 2: Word Embeddings and Sequence Modeling
Name: Nithin Manoj

Roll Number: B22CS066

Course: CSL 7640 – Natural Language Understanding

Instructor: Dr. Anand Mishra

Project Overview
This repository contains the implementation for Assignment 2 of the NLU course. The project is split into two distinct parts: building word embeddings from scratch using local data and generating names using character-level recurrent networks.

Problem 1: Word Embeddings
The goal was to learn vector representations for words using a corpus derived from IIT Jodhpur sources.

Model: Word2Vec (CBOW and Skip-gram architectures).

Implementation: Built from scratch using NumPy to handle forward passes, backpropagation, and weight updates.

Analysis: Includes semantic similarity tests, analogy solving, and PCA-based clusters for visualization.

Problem 2: Character-Level Name Generation
This task focuses on generating Indian names by predicting the next character in a sequence.

Architectures: Vanilla RNN, Bidirectional LSTM (BLSTM), and RNN with an Attention mechanism.

Framework: Implemented using PyTorch.

Evaluation: Models are compared based on Novelty Rate (uniqueness) and Diversity Score (variety of generated outputs).

Directory Structure
Plaintext
NLU-Assignment-2-B22CS066/
├── Problem1/
│   ├── Problem1.py          # Word2Vec source code
│   ├── Corpus.txt           # Processed text data
│   ├── WordCloud.png        # Corpus frequency visual
│   └── Visualization.png    # PCA embedding plot
├── Problem2/
│   ├── Problem2.py          # RNN/LSTM/Attention source code
│   └── TrainingNames.txt    # Dataset of 1000 Indian names
└── README.md                # Project documentation
Setup and Requirements
To run the scripts, ensure you have Python 3 installed along with the following libraries:

Bash
pip install torch numpy matplotlib wordcloud scikit-learn
Running the Code
To run Word2Vec training and analysis:

Bash
python Problem1/Problem1.py
To run the name generation models:

Bash
python Problem2/Problem2.py
Implementation Details
Problem 1: Focuses on the mathematical foundation of word embeddings. No high-level deep learning libraries were used for the training logic to ensure a full understanding of the Skip-gram and CBOW gradients.

Problem 2: Utilizes PyTorch for sequence modeling. The Attention mechanism was integrated into the RNN to improve the coherence of longer name generations.

Preprocessing: Standard NLP pipelines including tokenization, lowercasing, and special character removal were applied to both datasets.
