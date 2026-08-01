from dataclasses import dataclass
from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import urllib.request
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel



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

    @staticmethod
    def gpt2_124m(dropout = 0.0):
        # ровно HF gpt2
        return GPT2Config(block_size = 1024, vocab_size = 50257,
                          n_layer = 12, n_head = 12, n_embed = 768,
                          dropout = dropout, bias = True)

config = GPT2Config()

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.c_proj  = nn.Linear(4 * config.n_embed, config.n_embed)
        self.act     = nn.GELU(approximate = 'tanh') # gelu_new в HF, точный GELU даст другие логиты
    
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

        # имена (transformer.wte / wpe / h.{i} / ln_f, lm_head) повторяют HF GPT2LMHeadModel,
        # чтобы state_dict копировался ключ в ключ, см. load_pretrained
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # (vocab_size, embeding_size)
            wpe = nn.Embedding(config.block_size, config.n_embed), # (max_seq_length, embedding_size)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed, bias = config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)
        #initializing weights
        self.transformer.wte.weight = self.lm_head.weight # tying de/embedder weights
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
        te  = self.transformer.wte(idx) # look-up в словаре
        pos = torch.arange(T, device = idx.device)
        pe = self.transformer.wpe(pos) # positional embedings
        x = self.transformer.drop(te + pe) #embeddings
        # main pipe
        for block in self.transformer.h:
            x = block(x)
        #final norm
        x = self.transformer.ln_f(x) # (B, T, C)
        #projecting to logits
        logits = self.lm_head(x) # (B, T, C) @ (C, vocab_size) --> (B, T, vocab_size)
        loss = None

        if target is not None:
            loss = F.cross_entropy(input=logits.view(B*T, self.config.vocab_size), target=target.view(B*T,))
            

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = 50):
        B, T = idx.size() # starting point
        for _ in range(max_new_tokens):
            # cutting the context
            idx_cut = idx[:, -self.config.block_size:]
            # forward pass
            logits, _ = self(idx_cut) # (B, T, vocab)
            logits = logits[:, -1, :] / temperature # (B, vocab)

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

def pad_to(data, seq_len, pad_token = 50256):
    return data + [pad_token] * (seq_len - len(data)) 

# processing single example
def format_example(example):
    inp = example['input']
    inst = example['instruction']
    output = example['output']
    if inp:
        promt = f"###Instruction:\n{inst}\n\n###Input:\n{inp}\n\n###Response:\n"
    else:
        promt = f"###Instruction:\n{inst}\n\n###Response:\n"
    full = promt + output
    return promt, full

# processing whole dataset
def process_dataset(data):
    data = data['train']
    examples = []
    for ex in data:
        promt, full = format_example(ex)
        full_ids = enc.encode(full)
        promt_len = len(enc.encode(promt))
        examples.append((full_ids, promt_len)) # tokenized sentence and a boarder
    return examples

# returning sft batches
def get_batch_sft(data, batch_size, block_size):
    ix = torch.randint(0, len(data), (batch_size,))
    batch = [data[i] for i in ix]
    max_len = max(len(i[0]) for i in batch) - 1
    xs, ys = [], []
    for seq, promt_len in batch:
        x = seq[:-1]
        x = pad_to(x, max_len)
        y = seq[1:]
        y[:promt_len-1] = [-100] * (promt_len - 1) #masking promt tokens
        y = pad_to(y, max_len, -100) # not training model on pad tokens
        xs.append(torch.tensor(x, dtype = torch.long))
        ys.append(torch.tensor(y, dtype = torch.long))
    
    x = torch.stack(xs)
    y = torch.stack(ys)
    return x, y


# returning pretrained batches
def get_batch(data, batch_size, block_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x, y

    
#------------------------------------------------- TRAINING CYCLE
model = GPT2(config)
model.to(device)
chkp_path = 'ckpt.path'
start_step = 0

#resuming from checkpoint if it exists
def load_checkpoint(model, optimizer, chkp_path):
    if os.path.exists(chkp_path):
        checkpoint = torch.load(chkp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f'Checkpoint loaded. Resuming from step {start_step}.')
        return start_step
    else:
        print('No checkpoint found. Starting training from scratch.')
        return 0

# full training loop (pretrained, sft, lora) with checkpointing
def train_model(num_steps, mode, model, data, device, chkp_path = None, use_checkpoint = True):
    if mode == 'pretrained':
        learning_rate = 1e-3
    elif mode == 'sft':
        learning_rate = 3e-5
    else:
        learning_rate = 1e-4
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = learning_rate)
    start_step = load_checkpoint(model, optimizer, chkp_path) if use_checkpoint else 0
    for epoch in range(start_step, start_step + num_steps):
        if mode == 'pretrained':
            x, y = get_batch(data, 16, 256)
        else:
            x, y = get_batch_sft(data, 16, 256)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        if epoch % 10 == 0:
            print(f'"Epoch: {epoch + 1} | Loss: {loss:.4f}')
        
        # saving the state of the model 
        if epoch % 200 == 0 and use_checkpoint and chkp_path is not None:
            torch.save({
                'step': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, chkp_path)
            print(f'Checkpoint saved at step {epoch}.')
        loss.backward()
        optimizer.step()

#-------------------------------------------------- copying prettrained weights from gpt2

@torch.no_grad()
def load_pretrained(model_type = 'gpt2', dropout = 0.0):
    """Собирает GPT2 с конфигом HF-модели и копирует в него веса."""
    hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = hf.state_dict()

    hf_cfg = hf.config
    config = GPT2Config(block_size = hf_cfg.n_positions, vocab_size = hf_cfg.vocab_size,
                        n_layer = hf_cfg.n_layer, n_head = hf_cfg.n_head,
                        n_embed = hf_cfg.n_embd, dropout = dropout, bias = True)
    model = GPT2(config)
    sd = model.state_dict()

    # у HF это Conv1D, там веса лежат как (in, out), у nn.Linear наоборот
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                  'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # каузальная маска у нас в state_dict, у HF её нет (или лежит как attn.bias)
    skip = ('.attn.bias', '.attn.masked_bias')

    copied = set()
    for k, v in sd_hf.items():
        if k.endswith(skip):
            continue
        assert k in sd, f'нет ключа {k} в нашей модели'
        if any(k.endswith(t) for t in transposed):
            v = v.t()
        assert sd[k].shape == v.shape, f'{k}: {tuple(sd[k].shape)} vs {tuple(v.shape)}'
        sd[k].copy_(v)
        copied.add(k)

    # wte и lm_head.weight связаны, так что незакрытыми должны остаться только маски
    missed = [k for k in sd if k not in copied and not k.endswith('.mask')]
    assert not missed, f'не заполнены веса: {missed}'
    return model

#-------------------------------------------------- testing pretrained model
base = load_pretrained('gpt2').to(device)

# using Lora to speed up training
lora_config = LoraConfig(r = 8,
                         lora_alpha = 16,
                         target_modules = ["c_attn", "c_proj", "c_fc"],
                         lora_dropout = 0.05,
                         bias = "none")
#checkpoint load
if os.path.exists('lora_sft_ckpt'):
    gpt = PeftModel.from_pretrained(base, 'lora_sft_ckpt', is_trainable = True)
else:
    gpt = get_peft_model(base, lora_config)

# sft data
df = load_dataset('tatsu-lab/alpaca')
examples = process_dataset(df)


train_model(num_steps=1000,
            mode = 'lora',
            model = gpt,
            data = examples,
            device = device,
            use_checkpoint=False)

gpt.save_pretrained('lora_sft_ckpt')