import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import random

# TASK-1 : DATASET PREPARATION
# Collect data from IIT Jodhpur website and preprocess it

def collect_iitj_data():

    # These are the three sources that I have selected from IIT Jodhpur Portal website
    urls = [
        "https://www.iitj.ac.in/m/Index/main-departments?lg=en",
        "https://www.iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility?",
        "https://iitj.ac.in/office-of-academics/en/academic-regulations#4.%20Teaching%20and%20Evaluation"
    ]

    raw_documents = []
    print("Step 1: Scraping IIT Jodhpur data...")

    # Scraping text content from each webpage
    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')

            # Extract text from paragraph, list and headings
            content = " ".join([tag.get_text() for tag in soup.find_all(['p','li','h2'])])
            raw_documents.append(content)

        except:
            print("Error scraping:", url)

    clean_corpus = []
    all_tokens = []

    # Words that don't add much meaning so that they can be be removed
    boilerplate = ['home','contact','about','login','copyright']

    # Preprocessing each document
    for doc in raw_documents:

        # Convert text to lowercase
        doc = doc.lower()

        # Remove punctuation, numbers and non-text characters
        doc = re.sub(r'[^a-z\s]', ' ', doc)

        # Remove extra spaces
        doc = re.sub(r'\s+', ' ', doc).strip()

        # Tokenization (splitting into words)
        tokens = doc.split()

        # Remove unwanted boilerplate words
        tokens = [t for t in tokens if t not in boilerplate]

        if tokens:
            clean_corpus.append(" ".join(tokens))
            all_tokens.extend(tokens)

    # Creating vocabulary
    vocab = sorted(list(set(all_tokens)))

    # Display dataset statistics
    print("\n--- DATASET STATISTICS ---")
    print("Documents:", len(clean_corpus))
    print("Tokens:", len(all_tokens))
    print("Vocabulary:", len(vocab))

    # Save corpus : corpus file is obtained here below
    with open("Corpus.txt","w",encoding="utf-8") as f:
        f.write("\n".join(clean_corpus))

    # Generate word cloud to visualize frequent words
    wordcloud = WordCloud(width=800,height=400,background_color="white").generate(" ".join(all_tokens))
    wordcloud.to_file("WordCloud.png")

    return all_tokens, vocab

# TASK-2 : WORD2VEC MODEL IMPLEMENTATION FROM SCRATCH
# Includes both Skip-gram and CBOW models

class Word2Vec:

    def __init__(self, vocab, embed_dim=100):

        # Stores the vocabulary and embedding size
        self.vocab = vocab
        self.v_size = len(vocab)
        self.d_size = embed_dim

        # Mapping the words to indices
        self.word_to_id = {w:i for i,w in enumerate(vocab)}
        self.id_to_word = {i:w for i,w in enumerate(vocab)}

        # Initialize weight matrices randomly
        self.W1 = np.random.uniform(-0.5,0.5,(self.v_size,self.d_size))
        self.W2 = np.random.uniform(-0.5,0.5,(self.v_size,self.d_size))

    # Sigmoid activation function is implemted here
    def sigmoid(self,x):
        return 1/(1+np.exp(-np.clip(x,-10,10)))


    # Skip-gram training with negative sampling
    def train_skipgram(self, center, context, neg_samples, lr):

        c = self.word_to_id[center]
        o = self.word_to_id[context]

        v = self.W1[c]
        u = self.W2[o]

        # Forward pass
        z = np.dot(v,u)
        p = self.sigmoid(z)

        # Compute gradient
        grad = (p-1)

        # Update output weights
        self.W2[o] -= lr*grad*v

        # Negative sampling updates
        for n in neg_samples:
            u_n = self.W2[n]
            p_n = self.sigmoid(np.dot(v,u_n))
            grad_n = p_n
            self.W2[n] -= lr*grad_n*v

        # Update input weights
        self.W1[c] -= lr*grad*u


    # CBOW training
    def train_cbow(self, context_words, target_word, lr):

        target_idx = self.word_to_id[target_word]
        context_indices = [self.word_to_id[w] for w in context_words]

        # Average context vectors
        context_vec = np.mean(self.W1[context_indices], axis=0)

        u = self.W2[target_idx]

        # Forward pass
        z = np.dot(context_vec, u)
        p = self.sigmoid(z)

        grad = (p - 1)

        # Update weights
        self.W2[target_idx] -= lr * grad * context_vec

        for idx in context_indices:
            self.W1[idx] -= lr * grad * u


# MAIN EXECUTION

tokens, vocab = collect_iitj_data()

# Model hyperparameters
EMBED = 100
WINDOW = 2
NEG = 5
LR = 0.025

# Training Skip-gram model
model_skip = Word2Vec(vocab,EMBED)

print("\nTraining Skip-gram...")

for epoch in range(5):

    for i in range(WINDOW,len(tokens)-WINDOW):

        center = tokens[i]
        context = tokens[i-WINDOW:i] + tokens[i+1:i+WINDOW+1]

        for c in context:

            neg = np.random.randint(0,len(vocab),NEG)
            model_skip.train_skipgram(center,c,neg,LR)

# Training CBOW model
model_cbow = Word2Vec(vocab,EMBED)

print("\nTraining CBOW...")

for epoch in range(5):

    for i in range(WINDOW,len(tokens)-WINDOW):

        context = tokens[i-WINDOW:i] + tokens[i+1:i+WINDOW+1]
        target = tokens[i]

        model_cbow.train_cbow(context,target,LR)


# TASK-3 : SEMANTIC ANALYSIS

# Cosine similarity between two words
def cosine(model,w1,w2):

    if w1 not in model.word_to_id:
        return 0

    if w2 not in model.word_to_id:
        return 0

    v1 = model.W1[model.word_to_id[w1]]
    v2 = model.W1[model.word_to_id[w2]]

    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


targets = ["research","student","phd","exam"]

# Show nearest neighbors using Skip-gram model
print("\n--- Skipgram Results ---")

for t in targets:

    sims = []

    for w in vocab:

        if w!=t:
            sims.append((w,cosine(model_skip,t,w)))

    sims = sorted(sims,key=lambda x:x[1],reverse=True)[:5]

    print(t,":",sims)


# Show nearest neighbors using CBOW method
print("\n--- CBOW Results ---")

for t in targets:

    sims = []

    for w in vocab:

        if w!=t:
            sims.append((w,cosine(model_cbow,t,w)))

    sims = sorted(sims,key=lambda x:x[1],reverse=True)[:5]

    print(t,":",sims)

# ANALOGY EXPERIMENTS

def analogy(model,a,b,c):

    if a not in model.word_to_id:
        return []

    if b not in model.word_to_id:
        return []

    if c not in model.word_to_id:
        return []

    v = model.W1[model.word_to_id[b]] - model.W1[model.word_to_id[a]] + model.W1[model.word_to_id[c]]

    sims = []

    for w in vocab:
        vec = model.W1[model.word_to_id[w]]
        sim = np.dot(v,vec)/(np.linalg.norm(v)*np.linalg.norm(vec))
        sims.append((w,sim))

    return sorted(sims,key=lambda x:x[1],reverse=True)[:5]


print("\n--- Analogy Skipgram ---")

print("student : exam :: phd : ?")
print(analogy(model_skip,"student","exam","phd"))

print("research : lab :: student : ?")
print(analogy(model_skip,"research","lab","student"))


print("\n--- Analogy CBOW ---")

print("student : exam :: phd : ?")
print(analogy(model_cbow,"student","exam","phd"))

print("research : lab :: student : ?")
print(analogy(model_cbow,"research","lab","student"))

# TASK-4 : VISUALIZATION USING PCA

words = ["research","student","phd","faculty","exam"]

words = [w for w in words if w in model_skip.word_to_id]

vectors = np.array([model_skip.W1[model_skip.word_to_id[w]] for w in words])

# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.figure(figsize=(10,8))

# Plots the  each word
for i,word in enumerate(words):

    plt.scatter(reduced[i,0],reduced[i,1])
    plt.annotate(word,(reduced[i,0],reduced[i,1]))

plt.title("Word Embedding Visualization")

plt.savefig("Visualization.png")

print("Visualization saved")
