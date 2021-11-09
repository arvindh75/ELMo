#!/usr/bin/env python
# coding: utf-8

# ## Imports and Global Variables

# In[1]:


import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numpy import dot
from torch import optim
from collections import Counter
from numpy.linalg import norm
from tqdm.auto import tqdm
from torch.autograd import Variable
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset, DataLoader


# In[2]:


EPOCHS = 50
MODE = "TEST"
LR = 1e-5
BATCH_SIZE = 32


# In[3]:


CONV_FILTERS = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
NUM_HIGHWAY = 2
CHAR_EMBED_DIM = 50
WORD_EMBED_DIM = 300
MAX_CHAR = 50
MIN_COUNT = 5
MAX_LEN = 256
OUTPUT_DIM = 150
NUM_UNITS = 256
NUM_LAYERS = 2


# In[4]:


if MODE == "TRAIN":
    with open("./data/corpus.json") as f:
        corpus = json.load(f)


# In[5]:


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print("device - " + str(device))


# In[6]:


get_ipython().system('nvidia-smi')


# ## Classes and Functions

# In[7]:


class Tokenizer:
    def __init__(self, word2id, char2id):
        self.word2id = word2id
        self.char2id = char2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.id2ch = {i: char for char, i in char2id.items()}
    
    @classmethod
    def from_corpus(cls, corpus, min_count=5):
        word_count = Counter()
        for sentence in corpus:
            word_count.update(sentence.lower().split())
        word_count = list(word_count.items())
        word_count.sort(key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(word_count):
            if count < min_count:
                break
        vocab = word_count[:i]
        vocab = [v[0] for v in vocab]
        word_lexicon = {}
        for special_word in ['<oov>', '<pad>']:
            if special_word not in word_lexicon:
                word_lexicon[special_word] = len(word_lexicon)
        for word in vocab:
            if word not in word_lexicon:
                word_lexicon[word] = len(word_lexicon)
        char_lexicon = {}
        for special_char in ['<oov>', '<pad>']:
            if special_char not in char_lexicon:
                char_lexicon[special_char] = len(char_lexicon)
        for sentence in corpus:
            for word in sentence.split():
                for ch in word:
                    if ch not in char_lexicon:
                        char_lexicon[ch] = len(char_lexicon)
        return cls(word_lexicon,char_lexicon)
    
    def tokenize(self,text,max_length=512,max_char=50):
        oov_id, pad_id = self.word2id.get("<oov>"), self.word2id.get("<pad>")
        w = torch.LongTensor(max_length).fill_(pad_id)
        words = text.lower().split()
        for i, wi in enumerate(words[:max_length]):
            w[i] = self.word2id.get(wi, oov_id)
        oov_id, pad_id = self.char2id.get("<oov>"), self.char2id.get("<pad>")
        c = torch.LongTensor(max_length,max_char).fill_(pad_id)
        for i, wi in enumerate(words[:max_length]):
            for j,wij in enumerate(wi[:max_char]):
                c[i][j]=self.char2id.get(wij, oov_id)
        return w, c, len(words[:max_length])


# In[9]:


if MODE == "TRAIN":
    TOKENIZER = Tokenizer.from_corpus(corpus, MIN_COUNT)
else:
    with open(f"./checkpoints/tokenizer.json") as f:
        d = json.load(f)
    TOKENIZER = Tokenizer(d["word2id"], d["ch2id"])


# In[10]:


N_CONV_FILTERS = sum(z[1] for z in CONV_FILTERS)
FINAL_EMBED_DIM = WORD_EMBED_DIM + N_CONV_FILTERS
char2id_len = len(TOKENIZER.char2id)
word2id_len = len(TOKENIZER.word2id)
PADDING_INDX_CHAR = TOKENIZER.char2id.get("<pad>")
PADDING_INDX_WORD = TOKENIZER.word2id.get("<pad>")


# In[11]:


class elmoDS(Dataset):
    def __init__(self, corpus, tokenizer):
        self.corpus = corpus
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        text = self.corpus[idx]
        word, char, i = self.tokenizer.tokenize(text, max_length = MAX_LEN, max_char = MAX_CHAR)
        return char, word, i
    
    def __len__(self):
        return len(self.corpus)


# In[12]:


if MODE == "TRAIN":
    data = elmoDS(corpus, TOKENIZER)
    data_loader = DataLoader(data, batch_size = BATCH_SIZE)


# In[21]:


# Inspirations form https://gist.github.com/Redchards/65f1a6f758a1a5c5efb56f83933c3f6e
class Highway(nn.Module):
    def __init__(self, inp_dim, n_layers, activation):
        super(Highway, self).__init__()
        self.inp_dim = inp_dim
        self.layers = nn.ModuleList([nn.Linear(inp_dim, inp_dim * 2) for i in range(n_layers)])
        self.activation = activation
        for layer in self.layers:
            layer.bias[inp_dim:].data.fill_(1)
    
    def forward(self, inp):
        cur_inp = inp
        for l in self.layers:
            proj_inp = l(cur_inp)
            l_p = cur_inp
            nl_p = self.activation(proj_inp[:, 0 : self.inp_dim])
            g = torch.sigmoid(proj_inp[:, self.inp_dim : (2 * self.inp_dim)])
            cur_inp = g * l_p + (1 - g) * nl_p
        return cur_inp


# In[23]:


class elmo(nn.Module):
    def __init__(self):
        super(elmo, self).__init__()
        self.char_e = nn.Embedding(char2id_len, CHAR_EMBED_DIM, padding_idx = PADDING_INDX_CHAR)
        self.word_e = nn.Embedding(word2id_len, WORD_EMBED_DIM, padding_idx = PADDING_INDX_WORD)
        self.activation = nn.ReLU()
        conv_list = []
        for w, h in CONV_FILTERS:
            conv_list.append(nn.Conv1d(in_channels = CHAR_EMBED_DIM, out_channels = h, kernel_size = w, bias = True))
        self.convs = nn.ModuleList(conv_list)
        self.highway = Highway(N_CONV_FILTERS, NUM_HIGHWAY, activation = self.activation)
        self.proj = nn.Linear(FINAL_EMBED_DIM, OUTPUT_DIM, bias = True)
        fLSTM = []
        bLSTM = []
        for x in range(NUM_LAYERS):
            if x == 0:
                fLSTM.append(nn.LSTM(input_size = OUTPUT_DIM, hidden_size = NUM_UNITS, batch_first = True))
                bLSTM.append(nn.LSTM(input_size = OUTPUT_DIM, hidden_size = NUM_UNITS, batch_first = True))
            else:
                fLSTM.append(nn.LSTM(input_size = NUM_UNITS, hidden_size = NUM_UNITS, batch_first = True))
                bLSTM.append(nn.LSTM(input_size = NUM_UNITS, hidden_size = NUM_UNITS, batch_first = True))
        self.fLSTM = nn.ModuleList(fLSTM)
        self.bLSTM = nn.ModuleList(bLSTM)
        self.linL = nn.Linear(in_features = NUM_UNITS, out_features = word2id_len)
    
    def forward(self, char_input, word_input):
        embeddings = []
        batch_size = word_input.size(0)
        seq_len = word_input.size(1)
        word_em = self.word_e(Variable(word_input))
        embeddings.append(word_em)
        char_input = char_input.view(batch_size * seq_len, -1)
        char_em = self.char_e(Variable(char_input)).transpose(1, 2)
        conv_list = []
        for x in range(len(self.convs)):
            c, temp = torch.max(self.convs[x](char_em), dim = -1)
            conv_list.append(self.activation(c))
        char_em = self.highway(torch.cat(conv_list, dim = -1))
        embeddings.append(char_em.view(batch_size, -1, N_CONV_FILTERS))
        embeddings = self.proj(torch.cat(embeddings, dim = 2))
        forward = []
        backward = []
        forward.append(embeddings)
        backward.append(embeddings)
        for f_layer, b_layer in zip(self.fLSTM, self.bLSTM):
            forward.append(f_layer(forward[-1])[0])
            backward.append(torch.flip(b_layer(torch.flip(backward[-1], dims = [1, ]))[0], dims = [1, ]))
        return forward, backward


# ## Training

# In[ ]:


model = elmo()
model.to(device)


# In[ ]:


opt = optim.Adam(model.parameters(), lr = LR)
loss_func = torch.nn.NLLLoss()


# In[ ]:


with open(f"./checkpoints/tokenizer.json", "w") as f:
    json.dump({"word2id": TOKENIZER.word2id, "char2id": TOKENIZER.char2id}, f, indent=4)


# In[ ]:


for epoch in range(EPOCHS):
    total_loss = 0
    print(f"Epoch - {epoch}")
    for batch in tqdm(data_loader):
        total_loss = 0
        chars, words, i = batch
        chars = chars.to(device)
        words = words.to(device)
        forward, backward = model(chars, words)
        forward = forward[-1]
        backward = backward[-1]
        max_k = torch.max(i)
        loss = 0
        for k in range(1, max_k):
            fp = forward[:, k-1, :]
            bp = backward[:, k-1, :]
            f_layer = model.linL(fp).squeeze()
            b_layer = model.linL(bp).squeeze()
            f_loss = torch.nn.functional.log_softmax(f_layer, dim = 1).squeeze()
            b_loss = torch.nn.functional.log_softmax(b_layer, dim = 1).squeeze()
            loss += loss_func(f_loss, words[:, k]) + loss_func(b_loss, words[:, k])
        loss.backward()
        opt.step()
        opt.zero_grad()
        model.zero_grad()
        total_loss += loss.detach().item()
    
    if epoch%2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': total_loss,
        }, f"./checkpoints/model.pt")
    print(f"Total Loss - {total_loss}")


# ## Testing

# In[24]:


model = elmo()
model.load_state_dict(torch.load(f"./checkpoints/model.pt"), strict=False)
model.to(device)


# In[25]:


def euc_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


# In[26]:


def cos_sim(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1) * norm(v2))


# In[37]:


def gen_embeddings(sentence, word):
    indx = sentence.split().index(word)
    model.eval()
    with torch.no_grad():
        words, chars, i = TOKENIZER.tokenize(sentence, max_length = MAX_LEN)
        chars = chars.unsqueeze(0).to(device)
        words = words.unsqueeze(0).to(device)
        forward, backward = model(chars, words)
        en_embedding = forward[0][0][indx].cpu().detach().numpy()
        h = list()
        for x in range(1, len(forward)):
            h.append(torch.cat((forward[x][0][indx], backward[x][0][indx])).cpu().detach().numpy())
    return np.mean(h, axis = 0) # Can be modified as required


# In[28]:


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# In[29]:


sentence1 = "I love the fall season"
sentence2 = "Try not to fall down while balancing"
word = "fall"
fall1 = gen_embeddings(sentence1, word)
fall2 = gen_embeddings(sentence2, word)


# In[30]:


sentence = "I love the climate in October"
word = "climate"
climate = gen_embeddings(sentence, word)


# In[31]:


cos_sim(fall1, climate)


# In[32]:


cos_sim(fall2, climate)


# In[38]:


sentence1 = "He is the king of this region"
sentence2 = "In chess queen is the strongest player"
word1 = "king"
word2 = "queen"
e11 = gen_embeddings(sentence1, word1)
e12 = gen_embeddings(sentence2, word2)


# In[40]:


sentence1 = "I like coffee on rainy evenings"
sentence2 = "The British love tea parties"
word1 = "coffee"
word2 = "tea"
e21 = gen_embeddings(sentence1, word1)
e22 = gen_embeddings(sentence2, word2)


# In[48]:


cos_sim(e11, e12), euc_dist(e11, e12)


# In[49]:


cos_sim(e21, e22), euc_dist(e21, e22)


# In[50]:


cos_sim(e11, e21), euc_dist(e11, e21)


# In[51]:


cos_sim(e11, e22), euc_dist(e11, e22)


# In[52]:


cos_sim(e12, e21), euc_dist(e12, e21)


# In[53]:


cos_sim(e12, e22), euc_dist(e12, e22)

