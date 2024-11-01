# Fetch ML Assessment
link to assessment [here](https://app3.greenhouse.io/tests/dc9d7860c17da3281ab024e4ef3d51d2?utm_medium=email&utm_source=TakeHomeTest&utm_source=Automated)

# How to Run?
* `result.ipynb` contains the explaination, implementation and results of all the tasks.

## Base Model
Base Model Architecture:
 - Embedding Layer:
    - It will generate the embedding vector for each word. Give more context to the model about the word. 
 - Positional Encoding:
    - It will add positional information to the embedding vector. Which helps the model to understand the order of the words in the sentence.
 - Transformer Encoder Layer:
    - It will generate the embedding vector for each word, with same sequence length.

This is the base model which is used for all the tasks. Used Transformer Encoder Layer for all the tasks. Reason for this is using attention between words give better understanding of the sentence, and use of each word, resulting in good embeddings.

## Task1
Sentence Embedding Transformer Model, architecture:
 - Base Model
 - Mean Pooling: Average all the sequence output from the base model, to get a single vector. 
 - Linear Layer (300 dimension): Added another dense layer with 300 output neuron, generating embedding vector of shape (1, 300), for each sentence.

## Task2
Sentiment Classification Model, architecture:
 - Base Model
 - Mean Pooling : Average all the sequence output from the base model, to get a single vector.
 - Linear Layer (3 classes, Positive, Negative, Neutral) : Added another dense layer with 3 output neuron, each neuron representing a class.
 - Final Softmax Layer can be used to get the probability of each class, and choose the class with highest probability.

## Task 3 and 4
File is present in both formats:
* [pdf](Fetch-assessment-Task3.pdf)
* [docs](https://docs.google.com/document/d/1zNavmyjJhrhIhzkUU1A5l_5o62VSlwJX_xnaQYcy8G4/edit?usp=sharing)




