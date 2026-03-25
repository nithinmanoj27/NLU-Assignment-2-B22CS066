# CSL 7640 – Natural Language Understanding  
## Assignment 2 | Spring 2026

**Name:** Nithin Manoj  
**Roll Number:** B22CS066  
**Course:** CSL 7640 – Natural Language Understanding  
**Instructor:** Dr. Anand Mishra  

---

## 1. Overview

This repository contains the implementation for Assignment 2 of the Natural Language Understanding course.  
The assignment focuses on:

- Learning Word Embeddings from IIT Jodhpur data  
- Character-level Name Generation using Recurrent Neural Networks  

Both tasks are implemented following the assignment instructions.

---

## 2. Project Modules

| Problem | Task | Techniques |
|---------|------|------------|
| Problem 1 | Word Embeddings | CBOW, Skip-gram, PCA, Analogies |
| Problem 2 | Name Generation | Vanilla RNN, BLSTM, Attention |

---

## 3. Problem 1: Word Embeddings (IIT Jodhpur Data)

Textual data was collected from IIT Jodhpur sources and preprocessed to create a clean corpus.  
Word2Vec models were implemented from scratch and evaluated.

### Steps Performed

- Data collection from IIT Jodhpur sources  
- Text preprocessing and corpus creation  
- Implementation of CBOW and Skip-gram  
- Semantic similarity and analogy experiments  
- PCA visualization  

### Files

- `Problem1/Problem1.py` – Word2Vec implementation  
- `Problem1/Corpus.txt` – Cleaned corpus  
- `Problem1/WordCloud.png` – Word cloud visualization  
- `Problem1/Visualization.png` – Embedding visualization  

---

## 4. Problem 2: Character-Level Name Generation

Indian names were generated using recurrent neural network models.

### Models Implemented

- Vanilla RNN  
- Bidirectional LSTM (BLSTM)  
- RNN with Attention  

### Evaluation Metrics

- Novelty Rate  
- Diversity Score  

### Files

- `Problem2/Problem2.py` – Model implementation  
- `Problem2/TrainingNames.txt` – Dataset of 1000 names  

---

## 5. Repository Structure


```bash
NLU-Assignment-2-B22CS066
│
├── Problem1
│ ├── Problem1.py
│ ├── Corpus.txt
│ ├── WordCloud.png
│ └── Visualization.png
│
├── Problem2
│ ├── Problem2.py
│ └── TrainingNames.txt
│
└── README.md
```

## 6. Setup and Usage

### Requirements

Install required libraries:

```bash
pip install torch numpy matplotlib wordcloud scikit-learn
```
Running the Code
Run Problem 1:
```bash
python Problem1/Problem1.py
```

Run Problem 2:
```bash
python Problem2/Problem2.py
```

Notes
- Word2Vec models in Problem 1 are implemented from scratch using NumPy
- RNN models in Problem 2 are implemented using PyTorch
- All preprocessing and evaluation steps are followed as in assignment guidelines
