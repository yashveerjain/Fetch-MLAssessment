# %% [markdown]
# # Fetch Assessment

# %%
from utils import *
from dataset import *

# %%
dataset = TaskDataset(data=None)

# %% [markdown]
# # Task 1
# Sentence Embedding Model, architecture:
#  - Embedding Layer 
#  - Positional Encoding
#  - Transformer Encoder Layer
#  - Mean Pooling
#  - Linear Layer (generates embedding)

# %%
# initialize models
n_layers = 1 # The number of transformer encoder layers
d_model = 512 # The input embedding dimension
embed_size = 300 # The output embedding size
nhead = 8 # The number of attention heads in encoder layer
task_type = "se" # The type of task (sentence embedding or sentiment classification)
vocab_size = dataset.vocab_size # The size of the vocabulary (using hugging face bert tokenizer)
base_model, se_task_model = init_model(vocab_size, n_layers, d_model, nhead, embed_size, task_type)

# %%
# Generate sample input
input_data = torch.randint(0, dataset.vocab_size, (2, 100))

# Forward pass
base_output = base_model(input_data)
task_output = se_task_model(base_output)

print("Base model output:", base_output.shape)
print("Task model output:", task_output.shape)

# Generate input based on text
text1 = "How, are you?"
text2 = "I am good"

input_data1 = dataset.encode(text1)
input_data2 = dataset.encode(text2)

# Forward pass
base_output1 = base_model(input_data1)
base_output2 = base_model(input_data2)
task_output1 = se_task_model(base_output1)
task_output2 = se_task_model(base_output2)

print("Base model output 1:", base_output1.shape)
print("Base model output 2:", base_output2.shape)
print("Task model output 1:", task_output1.shape)   
print("Task model output 2:", task_output2.shape)
print("Task model output 2:", task_output2)

# %% [markdown]
# # Task 2
# Sentiment Classification Model, architecture:
#  - Embedding Layer
#  - Positional Encoding
#  - Transformer Encoder Layer
#  - Mean Pooling
#  - Linear Layer (3 classes, Positive, Negative, Neutral) ( can be trained using cross entropy loss)

# %%
task_type = "sc"
base_model, sc_task_model = init_model(vocab_size, n_layers, d_model, nhead, embed_size, task_type)

# %%
# Generate sample input
input_data = torch.randint(0, dataset.vocab_size, (2, 100))

# Forward pass
base_output = base_model(input_data)
task_output = sc_task_model(base_output)
task_output = nn.Softmax(dim=1)(task_output).detach().numpy()

print("Base model output:", base_output.shape)
print("Task model output:", task_output.shape)
print("Task model output:", task_output)

# Generate input based on text
text1 = "How, are you?"
text2 = "I am good"

input_data1 = dataset.encode(text1)
input_data2 = dataset.encode(text2)

# Forward pass
base_output1 = base_model(input_data1)
base_output2 = base_model(input_data2)
task_output1 = sc_task_model(base_output1)
task_output2 = sc_task_model(base_output2)

task_output1 = nn.Softmax(dim=1)(task_output1).detach().numpy()
task_output2 = nn.Softmax(dim=1)(task_output2).detach().numpy()

print("Base model output 1:", base_output1.shape)
print("Base model output 2:", base_output2.shape)
print("Task model output 1:", task_output1.shape)   
print("Task model output 2:", task_output2.shape)

# %%


# %% [markdown]
# # Task 4
# 
# Assuming we are training the sentence embedding model, and the sentiment classification model, simulatenously. Then each model can have different learning rates, as well for embedding in transformer will add different learning rate.

# %% [markdown]
# * Embedding Layer: Set a learning rate of 1e-3.
#   
#   -  This is a low-level layer responsible for capturing foundational features, so a finer learning rate helps prevent drastic changes that could drift away from good initial representations.
# * Transformer Layer: Use a learning rate of 5e-3.
#    
#    - As a middle layer, this part of the model focuses on generalizing patterns from the dataset. A slightly higher learning rate allows it to adapt faster, helping it develop better generalizations over time.
# * Sentence Embedding and Sentiment Classification Layers: Set a learning rate of 1e-2.
#   - These layers capture task-specific, high-level features crucial for task understanding. A higher learning rate here allows faster learning of these details, which is important to avoid long convergence times.

# %%
import torch.optim as optim

# %%
parameters = []
base_model_transformer_lr = 5e-3
base_model_embedding_lr = 1e-3

for name, param in base_model.named_parameters():
    if "embed" in name:
        parameters.append({"params": param, "lr": base_model_embedding_lr})
    else:
        parameters.append({"params": param, "lr": base_model_transformer_lr})

se_lr = 1e-2
sc_lr = 1e-2

parameters.append(
    {"params" : se_task_model.parameters(), "lr" : se_lr}
)

parameters.append(
    {"params" : sc_task_model.parameters(), "lr" : sc_lr}
)


optimizer = optim.Adam(parameters)


