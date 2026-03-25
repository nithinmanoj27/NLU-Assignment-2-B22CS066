# CSL 7640 – Natural Language Understanding
## Assignment 2 | Spring 2026

**Student Information**
* **Name:** Nithin Manoj
* **Roll Number:** B22CS066
* **Course:** CSL 7640 – Natural Language Understanding
* **Instructor:** Dr. Anand Mishra

---

### 1. Overview
This repository contains the implementation for Assignment 2 of the Natural Language Understanding course. The project focuses on custom Word2Vec implementation and character-level sequence modeling for name generation.

---

### 2. Project Modules

| Module | Task | Key Techniques |
| :--- | :--- | :--- |
| **Problem 1** | Word Embeddings | CBOW, Skip-gram, PCA Visualization, Analogies |
| **Problem 2** | Name Generation | Vanilla RNN, Bidirectional LSTM, Attention Mechanism |

---

### 3. Problem 1: Word Embeddings (IIT Jodhpur Data)
In this task, textual data was collected from IIT Jodhpur sources and preprocessed to create a clean corpus. Word2Vec models were then trained and evaluated.

**Key Steps:**
* Data collection and text preprocessing.
* Manual implementation of CBOW and Skip-gram architectures using NumPy.
* Evaluation via semantic similarity and analogy experiments.
* Visualization of high-dimensional vectors using PCA.

**Files:**
* `Problem1/Problem1.py`: Implementation of Word2Vec and analysis.
* `Problem1/Corpus.txt`: Cleaned dataset.
* `Problem1/WordCloud.png`: Word frequency visualization.
* `Problem1/Visualization.png`: Embedding visualization.

---

### 4. Problem 2: Character-Level Name Generation
This task involves generating Indian names using various recurrent neural network architectures.

**Models Implemented:**
* **Vanilla RNN:** Standard recurrent model.
* **Bidirectional LSTM (BLSTM):** Captures bidirectional dependencies.
* **RNN with Attention:** Uses attention to focus on relevant characters in the sequence.

**Evaluation Metrics:**
* **Novelty Rate:** Measures the percentage of unique, non-training names generated.
* **Diversity Score:** Evaluates the variety in the generated output.

**Files:**
* `Problem2/Problem2.py`: Model implementation and evaluation.
* `Problem2/TrainingNames.txt`: Dataset containing 1000 Indian names.

---

### 5. Repository Structure
```text
NLU-Assignment-2-B22CS066
│
├── Problem1/
│   ├── Problem1.py          
│   ├── Corpus.txt           
│   ├── WordCloud.png        
│   └── Visualization.png    
│
├── Problem2/
│   ├── Problem2.py          
│   └── TrainingNames.txt    
│
└── README.md
```

## 6. Setup and Usage

### Requirements

Install the necessary dependencies using pip:

```bash
pip install torch numpy matplotlib wordcloud scikit-learn
```

Running the Code

To execute the word embedding tasks:
```bash
python Problem1/Problem1.py
```
To execute the name generation models:
```bash
python Problem2/Problem2.py
```
