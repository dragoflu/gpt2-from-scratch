import os
import torch

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora_sft_ckpt')
PRETRAIN_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrain_ckpt', 'best.pt')

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

if device == 'mps':
    torch.mps.empty_cache()
