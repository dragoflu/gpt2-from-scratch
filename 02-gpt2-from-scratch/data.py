import urllib.request
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

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


# pretrain: каждый элемент это стартовая позиция окна block_size в общей ленте токенов
class PretrainDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# sft/lora: элемент это уже готовая пара (full_ids, promt_len) из process_dataset,
# паддинг тут не делаем, потому что max_len известен только внутри батча
class SFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RLDataSet(Dataset):
    def __init__(self, examples): # сюда подаем отформатированый, закодированный датасет --> падим в колейте в даталоадере
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# динамический паддинг по максимальной длине конкретного батча, а не всего датасета
def sft_collate(batch):
    max_len = max(len(seq) for seq, _ in batch) - 1
    xs, ys = [], []
    for seq, promt_len in batch:
        x = pad_to(seq[:-1], max_len)
        y = list(seq[1:]) # копия, иначе маска ниже портит сам examples
        y[:promt_len-1] = [-100] * (promt_len - 1) # masking promt tokens
        y = pad_to(y, max_len, -100) # not training model on pad tokens
        xs.append(torch.tensor(x, dtype = torch.long))
        ys.append(torch.tensor(y, dtype = torch.long))
    return torch.stack(xs), torch.stack(ys)


def get_dataloader(dataset, batch_size, shuffle = True, collate_fn = None):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn)


# train_model работает по шагам, а не по эпохам, поэтому лоадер должен уметь отдавать
# батчи бесконечно, перемешиваясь заново на каждом новом проходе по датасету
def infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


def process_dataset_rl(data):
    data = data['train']
    examples = []
    for example in data:
        formated = format_example_rl(example)
        buff = {}
        for key, val in formated.items():
            full_ids = enc.encode(val[1]) + [enc.eot_token]
            full_ids = full_ids[:1024]
            promt_len = len(enc.encode(val[0]))
            buff[key] = (full_ids, promt_len)
        examples.append(buff)
    return examples # {chosen: (tokenized, len), rejected: (tokenized, len)}



def collate_fn_rl(batch):
    xs_chosen = [ex['chosen'] for ex in batch]
    xs_rejected = [ex['rejected'] for ex in batch]
    xs_chosen, ys_chosen = sft_collate(xs_chosen)
    xs_rejected, ys_rejected = sft_collate(xs_rejected)
    return xs_chosen, ys_chosen, xs_rejected, ys_rejected



def format_example_rl(example):
    inp = example['input']
    inst = example['instruction']
    output1 = example['output1']
    output2 = example['output2']
    pref = example['preference']
    if inp:
        promt = f"###Instruction:\n{inst}\n\n###Input:\n{inp}\n\n###Response:\n"
    else:
        promt = f"###Instruction:\n{inst}\n\n###Response:\n"
    if pref == 1:
        return {'chosen': (promt, promt + output1), 'rejected': (promt, promt + output2)}
    else:
        return {'chosen': (promt, promt + output2), 'rejected': (promt, promt + output1)}

            
