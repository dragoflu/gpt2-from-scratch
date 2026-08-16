import math
import json
import os
from datasets import load_dataset
from peft import LoraConfig

from config import CKPT_DIR, device
from model import load_pretrained
from data import process_dataset
from train import load_checkpoint_lora, train_model, complete

base = load_pretrained('gpt2').to(device)

# using Lora to speed up training
lora_config = LoraConfig(r = 8,
                         lora_alpha = 16,
                         target_modules = ["c_attn", "c_proj", "c_fc"],
                         lora_dropout = 0.05,
                         bias = "none")

gpt = load_checkpoint_lora(base, CKPT_DIR, lora_config, loading = False).to(device)

# sft data
df = load_dataset('tatsu-lab/alpaca')
examples = process_dataset(df)

train_losses, val_losses = train_model(num_steps = 50,
            mode = 'lora',
            model = gpt,
            train_data = examples[:int(len(examples)*0.9)],
            val_data = examples[int(len(examples)*0.9):],
            device = device,
            use_checkpoint = False)

ppl_before = math.exp(val_losses[0])
ppl_after = math.exp(val_losses[-1])
print(f'Perplexity before training: {ppl_before:.2f}')
print(f'Perplexity after training: {ppl_after:.2f}')

#-------------------------------------------------- saving results
gpt.save_pretrained(CKPT_DIR)
history = {'train_losses': train_losses, 'val_losses': val_losses}
with open(os.path.join(CKPT_DIR, 'history.json'), 'w') as f:
    json.dump(history, f)

#-------------------------------------------------- testing lora model
gpt.eval()
prompt1 = "###Instruction:\n Give three tips for staying healthy\n\n###Response:\n"
prompt2 = "###Instruction:\n Classify the sentiment of the sentence.\n\n###Input:\nThe movie was way too long and I fell asleep.\n\n###Response:\n"
complete(gpt, prompt1, device, max_new_tokens = 100)
complete(gpt, prompt2, device, max_new_tokens = 100)
