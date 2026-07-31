from dataclasses import dataclass

from matplotlib.pyplot import step
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import urllib.request
import os
# from datasets import load_dataset

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

#------------------------------------------------- MODEL ARCHITECTURE
@dataclass
class GPT2Config:
    block_size: int = 256 #1024
    vocab_size: int = 50304 
    n_layer: int = 6 #12
    n_head: int = 6 #12
    n_embed: int = 384 #768
    dropout: float = 0.0  
    bias: bool = True

config = GPT2Config()

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.c_proj  = nn.Linear(4 * config.n_embed, config.n_embed)
        self.act     = nn.GELU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    # later, add dropouts and regularization for training
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        #dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        #initializing a mask
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # batch size, num_tokens, n_embed
        B, T, C = x.size() 
        #splitting, projecting and rearranging
        q, k, v = self.c_attn(x).split(self.n_embed, dim = 2) # (B, T, C) @ (C, 3 * C) --> (B, T, 3*C) --> splittin
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) --> (B, T, h, d_h) --> (B, h, T, d_h)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) --> (B, T, h, d_h) --> (B, h, T, d_h)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) --> (B, T, h, d_h) --> (B, h, T, d_h)
        #Self-attention matrix
        s = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // self.n_head)) # (B, h, T, d_h) @ (B, h, d_h, T) --> (B, h, T, T)
        s = s.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        #Softmax
        s = F.softmax(s, dim = -1)
        s = self.attn_dropout(s)
        #match with vals
        o = s @ v # (B, h, T, T) @ (B, h, T, d_h) --> (B, h, T, d_h)
        #rearranging and concatinating heads
        o = o.transpose(1, 2).reshape(B, T, C) # (B, h, T, d_h) --> (B, T, h, d_h) --> (B, T, C)
        #final projection
        x = self.residual_dropout(self.c_proj(o))
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embed) # (vocab_size, embeding_size)
        self.wpe = nn.Embedding(config.block_size, config.n_embed) # (max_seq_length, embedding_size)
        self.drop = nn.Dropout(config.dropout)
        self.pipe = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed,  bias = config.bias)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False) 
        #initializing weights
        self.wte.weight = self.lm_head.weight # tying de/embedder weights
        #standart initialization of weights
        self.apply(self._init_weights)
        # special initialization for residuals
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean = 0.0, std = 0.02 / math.sqrt(2 * config.n_layer))

    #basic weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, target = None):
        B, T = idx.size() # our input
        # getting token_embeddings
        te  = self.wte(idx) # look-up в словаре 
        pos = torch.arange(T, device = idx.device)
        pe = self.wpe(pos) # positional embedings
        x = self.drop(te + pe) #embeddings
        # main pipe
        for block in self.pipe:
            x = block(x)
        #final norm
        x = self.ln_f(x) # (B, T, C)
        #projecting to logits
        logits = self.lm_head(x) # (B, T, C) @ (C, vocab_size) --> (B, T, vocab_size)
        loss = None

        if target is not None:
            loss = F.cross_entropy(input=logits.view(B*T, config.vocab_size), target=target.view(B*T,))
            

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperatre = 1.0, top_k = 50):
        B, T = idx.size() # starting point
        for _ in range(max_new_tokens):
            # cutting the context
            idx_cut = idx[:, -self.config.block_size:]
            # forward pass
            logits, _ = self(idx_cut) # (B, T, vocab)
            logits = logits[:, -1, :] / temperatre # (B, vocab)

            #selecting top-k (O(n))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

            #softmax
            probs = F.softmax(logits, dim = -1)

            #sample
            next_token = torch.multinomial(probs, num_samples = 1)
            #concat
            idx = torch.cat((idx, next_token), dim = 1)
            #streaming
            print(enc.decode([next_token[0].item()]), end = '', flush = 'True')

        print()
        return idx

@torch.no_grad()
def complete(model, prompt, max_new_tokens = 250, temperature = 0.9, top_k = 50):
    model.eval()
    print(f'Prompt: {prompt}')
    idx = enc.encode(prompt)
    idx = torch.tensor(idx, dtype = torch.long)[None, :].to(device)
    idx = model.generate(idx, max_new_tokens, temperature, top_k)
    completion = enc.decode(idx[0].tolist())
    return completion

#------------------------------------------------- LOADING TRAINING DATA

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = urllib.request.urlopen(url).read().decode('utf-8')
enc = tiktoken.get_encoding('gpt2')
data = torch.tensor(enc.encode(text), dtype = torch.long)

def get_batch(data, batch_size, block_size):
    ix = torch.randint(0, len(data) - config.block_size - 1, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x, y
#------------------------------------------------- TRAINING CYCLE
model = GPT2(config)
model.to(device)
chkp_path = 'ckpt.path'
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
start_step = 0

#resuming from checkpoint if it exists
if os.path.exists(chkp_path):
    checkpoint = torch.load(chkp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    print(f'Checkpoint loaded. Resuming from step {start_step}.')
else:
    print('No checkpoint found. Starting training from scratch.')

num_steps = 0

for epoch in range(start_step, start_step + num_steps):
    x, y = get_batch(data, 16, 256)
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    _, loss = model(x, y)
    if epoch % 100 == 0:
        print(f'"Epoch: {epoch + 1} | Loss: {loss:.4f}')
    
    # saving the state of the model 
    if epoch % 1000 == 0:
        torch.save({
            'step': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, chkp_path)
        print(f'Checkpoint saved at step {epoch}.')
    loss.backward()
    optimizer.step()

complete(model, "I went to the church ", max_new_tokens = 250, temperature = 0.9, top_k = 50)

