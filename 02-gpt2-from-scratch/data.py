import urllib.request
import torch
import tiktoken

enc = tiktoken.get_encoding('gpt2')

SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def load_shakespeare():
    text = urllib.request.urlopen(SHAKESPEARE_URL).read().decode('utf-8')
    return torch.tensor(enc.encode(text), dtype = torch.long)


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
        full_ids = enc.encode(full) + [enc.eot_token] # end of sequence token, чтобы модель не генерировала дальше
        full_ids = full_ids[:256] # обрезаем, чтобы не превышать block_size
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
        y = list(seq[1:]) # копия, иначе маска ниже портит сам examples
        y[:promt_len-1] = [-100] * (promt_len - 1) #masking promt tokens
        y = pad_to(y, max_len, -100) # not training model on pad tokens
        xs.append(torch.tensor(x, dtype = torch.long))
        ys.append(torch.tensor(y, dtype = torch.long))

    x = torch.stack(xs)
    y = torch.stack(ys)
    return x, y


# returning pretrain batches
def get_batch(data, batch_size, block_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x, y
