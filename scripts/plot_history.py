import json
import os
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hist = json.load(open(os.path.join(root, '02-gpt2-from-scratch/lora_sft_ckpt/history.json')))
out = os.path.join(root, 'assets/sft_loss.png')

steps = [i * 10 for i in range(len(hist['train_losses']))] # логировали раз в 10 шагов

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize = (7, 4), dpi = 160)
# первая точка train это среднее по одному шагу, а не по десяти, поэтому она заниженная
ax.plot(steps[1:], hist['train_losses'][1:], marker = 'o', ms = 4, color = '#6ea8ff', label = 'train')
ax.plot(steps, hist['val_losses'], marker = 'o', ms = 4, color = '#ff9e64', label = 'val')
ax.set_xlabel('step')
ax.set_ylabel('cross-entropy loss')
ax.set_title('LoRA SFT on alpaca, GPT-2 124M', fontsize = 11)
ax.grid(alpha = 0.15)
ax.legend(frameon = False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok = True)
fig.savefig(out, facecolor = fig.get_facecolor())
print('saved', out)
