from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


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
            # -100 это промпт и паддинг, см. get_batch_sft
            loss = F.cross_entropy(input=logits.view(B*T, self.config.vocab_size), target=target.view(B*T,),
                                   ignore_index = -100)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eot_token, temperature = 1.0, top_k = 50, stream_decode = None):
        B, T = idx.size() # starting point
        seen = torch.zeros((B, self.config.vocab_size), dtype = torch.bool, device = idx.device) # mask for repetition penalty
        for _ in range(max_new_tokens):
            # cutting the context
            idx_cut = idx[:, -self.config.block_size:]
            # forward pass
            logits, _ = self(idx_cut) # (B, T, vocab)
            logits = logits[:, -1, :] / temperature # (B, vocab)

            #applying repetition penalty
            logits[seen & (logits > 0)] /= 1.2
            logits[seen & (logits < 0)] *= 1.2
            #selecting top-k (O(n))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

            #softmax
            probs = F.softmax(logits, dim = -1)

            #sample
            next_token = torch.multinomial(probs, num_samples = 1)
            # operations with new token
            if next_token.item() == eot_token:
                break # end of sequence

            # marking repetition penalty
            seen.scatter_(1, next_token, True)

            #concat
            idx = torch.cat((idx, next_token), dim = 1)
            #streaming
            if stream_decode is not None:
                print(stream_decode(next_token[0].item()), end = '', flush = True)

        if stream_decode is not None:
            print()
        return idx


#-------------------------------------------------- copying pretrained weights from gpt2
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
