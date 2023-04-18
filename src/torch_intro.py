import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
print(torch.__version__)


def softmax(z):
    z_exp = torch.exp(z - torch.max(z, axis=-1, keepdims=True)[0])
    return z_exp / torch.sum(z_exp, axis=-1, keepdims=True)


sentence = "Life is short, eat dessert first"

dc = {s: i for i, s in enumerate(sorted(sentence.replace(",", "").split()))}

sentence_int = torch.tensor([dc[s] for s in sentence.replace(",", "").split()])

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

# print(embedded_sentence)

# Init weight matrices

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28

W_q = torch.rand(d_q, d)
W_k = torch.rand(d_k, d)
W_v = torch.rand(d_v, d)

# Creating the Q, V, K matrices
x_2 = embedded_sentence[1]
query_2 = W_q.matmul(x_2)
print(query_2)
key_2 = W_k.matmul(x_2)
value_2 = W_v.matmul(x_2)

# print(query_2.shape)
# print(key_2.shape)
# print(value_2.shape)

# computing the embedding for the whole input

query = (W_q @ embedded_sentence.T).T
keys = (W_k @ embedded_sentence.T).T
values = (W_v @ embedded_sentence.T).T


omega_2 = query_2 @ keys.T

attention_weights_2 = softmax(omega_2 / d_k**0.5)
# print(attention_weights_2)

context_vector_2 = attention_weights_2 @ values
# print(context_vector_2)

# introducing multihead attention

h = 3
multihead_W_query = torch.rand(h, d_q, d)
multihead_W_key = torch.rand(h, d_k, d)
multihead_W_value = torch.rand(h, d_v, d)

multihead_query_2 = multihead_W_query @ x_2
print(multihead_query_2[0])
multihead_key_2 = multihead_W_key @ x_2
multihead_value_2 = multihead_W_value @ x_2

stacked_inputs = embedded_sentence.T.repeat(3, 1, 1)
print(stacked_inputs[1])
multihead_queries = torch.bmm(multihead_W_key, stacked_inputs)

query = (W_q @ embedded_sentence.T).T
keys = (W_k @ embedded_sentence.T).T
values = (W_v @ embedded_sentence.T).T
