import math
import json
import os

from config import device, PRETRAIN_CKPT_DIR
from model import GPT2, GPT2Config
from data import load_shakespeare
from train import train_model, complete

config = GPT2Config()
small = GPT2(config).to(device) # рандомная инициализация
loading = False # хотим ли подгружать чекппоинт

data = load_shakespeare()
val_start = int(len(data) * 0.8)
val_end = int(len(data) * 0.9)
train_data = data[:val_start]
val_data = data[val_start:val_end]
test_data = data[val_end:]

train_losses, val_losses, best_loss = train_model(num_steps = 5000, 
            mode = 'pretrained',
            model = small,
            train_data = train_data,
            val_data = val_data,
            device = device,
            loading = loading,
            ckpt_path = PRETRAIN_CKPT_DIR,
            patience = 20)

ppl_before = math.exp(val_losses[0])
ppl_after = math.exp(val_losses[-1])
print(f'Perplexity before training: {ppl_before:.2f}') # sanity check: должно быть ~vocab_size
print(f'Perplexity after training: {ppl_after:.2f}')

history = {'train_losses': train_losses, 'val_losses': val_losses}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrain_history.json'), 'w') as f:
    json.dump(history, f)

small.eval()
complete(small, "ROMEO:", device, max_new_tokens = 100) # TODO: подобрать промпт под то, что реально видел на обучении
