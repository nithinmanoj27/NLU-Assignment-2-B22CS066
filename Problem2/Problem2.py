import torch
import torch.nn as nn
import numpy as np
import random

# TASK-0 : LOAD DATASET

# Reading the list of Indian names generated using ChatGPT
# Each line contains one name,, Total = 1000 names
with open("TrainingNames.txt","r",encoding="utf-8") as f:
    names = [line.strip() for line in f.readlines()]

# Creating character vocabulary from dataset
# Since this is character level generation, we take all unique characters
chars = sorted(list(set("".join(names))))

# Mapping characters to numbers and numbers back to characters
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

# Total unique characters
vocab_size = len(chars)

print("Total Names:",len(names))
print("Vocabulary Size:",vocab_size)


# DATA PREPARATION

# Convert name into index values
# Example : ram → [17,5,9]
def encode(name):
    return [char_to_idx[ch] for ch in name]

# Convert index values back to characters
def decode(indices):
    return "".join([idx_to_char[i] for i in indices])


# MODEL-1 : VANILLA RNN

# Basic RNN model for character level name generation
class VanillaRNN(nn.Module):

    def __init__(self,vocab_size,hidden_size):

        super(VanillaRNN,self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size,hidden_size)

        # Simple RNN layer
        self.rnn = nn.RNN(hidden_size,hidden_size,batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size,vocab_size)

    def forward(self,x):

        x = self.embedding(x)
        out,_ = self.rnn(x)
        out = self.fc(out)

        return out


# MODEL-2 : BIDIRECTIONAL LSTM

# BLSTM captures both forward and backward context
class BLSTM(nn.Module):

    def __init__(self,vocab_size,hidden_size):

        super(BLSTM,self).__init__()

        self.embedding = nn.Embedding(vocab_size,hidden_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            bidirectional=True,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size*2,vocab_size)

    def forward(self,x):

        x = self.embedding(x)
        out,_ = self.lstm(x)
        out = self.fc(out)

        return out


# MODEL-3 : RNN WITH ATTENTION

# RNN model with basic attention mechanism
class AttentionRNN(nn.Module):

    def __init__(self,vocab_size,hidden_size):

        super(AttentionRNN,self).__init__()

        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.RNN(hidden_size,hidden_size,batch_first=True)

        # Attention layer
        self.attn = nn.Linear(hidden_size,hidden_size)

        self.fc = nn.Linear(hidden_size,vocab_size)

    def forward(self,x):

        x = self.embedding(x)
        out,_ = self.rnn(x)

        # Applying attention
        attn_weights = torch.softmax(self.attn(out),dim=1)
        out = out * attn_weights

        out = self.fc(out)

        return out


# TRAINING FUNCTION

# Training function for all models
def train_model(model,epochs=20,lr=0.003):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):

        total_loss = 0

        # Training using each name
        for name in names:

            encoded = encode(name)

            inp = torch.tensor(encoded[:-1]).unsqueeze(0)
            target = torch.tensor(encoded[1:]).unsqueeze(0)

            optimizer.zero_grad()

            output = model(inp)

            loss = criterion(output.squeeze(),target.squeeze())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch:",epoch,"Loss:",total_loss)


# COUNT TRAINABLE PARAMETERS

# Function to count total trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# GENERATE NAMES

# Generate names character by character
def generate(model,max_len=10):

    # Start from random character
    idx = random.randint(0,vocab_size-1)
    result = [idx]

    for _ in range(max_len):

        inp = torch.tensor(result).unsqueeze(0)
        output = model(inp)

        next_char = torch.argmax(output[0,-1]).item()

        result.append(next_char)

    return decode(result)


# HYPERPARAMETERS

hidden_size = 128
learning_rate = 0.003
epochs = 20

print("\nHyperparameters")
print("Hidden Size:",hidden_size)
print("Learning Rate:",learning_rate)
print("Epochs:",epochs)


# TRAIN MODEL-1

print("\nTraining Vanilla RNN")

model_rnn = VanillaRNN(vocab_size,hidden_size)

print("Trainable Parameters:",count_parameters(model_rnn))

train_model(model_rnn,epochs,learning_rate)


# TRAIN MODEL-2

print("\nTraining BLSTM")

model_blstm = BLSTM(vocab_size,hidden_size)

print("Trainable Parameters:",count_parameters(model_blstm))

train_model(model_blstm,epochs,learning_rate)


# TRAIN MODEL-3

print("\nTraining Attention RNN")

model_attn = AttentionRNN(vocab_size,hidden_size)

print("Trainable Parameters:",count_parameters(model_attn))

train_model(model_attn,epochs,learning_rate)


# GENERATE SAMPLE NAMES

# Generate 20 names from each model
def generate_samples(model,name):

    print("\nGenerated names using",name)

    samples = []

    for _ in range(20):

        n = generate(model)
        samples.append(n)
        print(n)

    return samples


samples_rnn = generate_samples(model_rnn,"Vanilla RNN")
samples_blstm = generate_samples(model_blstm,"BLSTM")
samples_attn = generate_samples(model_attn,"Attention RNN")


# TASK-2 : QUANTITATIVE EVALUATION

# Calculate diversity and novelty
def evaluate(samples):

    unique = len(set(samples))
    total = len(samples)

    diversity = unique / total

    novelty = len([s for s in samples if s not in names]) / total

    return diversity,novelty


print("\nEvaluation Results")

print("Vanilla RNN:",evaluate(samples_rnn))
print("BLSTM:",evaluate(samples_blstm))
print("Attention:",evaluate(samples_attn))
